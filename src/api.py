"""
RegisterABot Interface Tool — Multi-Tenant

FastAPI server providing:
- ConnectionManager for multiple simultaneous bot connections to RegisterABot relay
- Per-slug WebSocket connections, each routing through gatekeeper with bot_slug
- API for dynamic connect/disconnect (called by gatekeeper on profile launch/stop)
- Legacy single-bot mode via env vars for backward compatibility

Message flow:
  relay envelope in → extract messages → gatekeeper /process (with bot_slug)
  → runner responds via sync bridge → wrap in envelope → relay out
"""

import os
import json
import asyncio
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import JSONResponse
import uvicorn
import httpx
import websockets
import structlog


# =============================================================================
# LOGGING
# =============================================================================

structlog.configure(
    processors=[
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.dev.ConsoleRenderer(colors=True)
    ],
    wrapper_class=structlog.stdlib.BoundLogger,
    context_class=dict,
    logger_factory=structlog.PrintLoggerFactory(),
    cache_logger_on_first_use=True,
)

log = structlog.get_logger()


# =============================================================================
# GLOBALS
# =============================================================================

GATEKEEPER_URL = os.environ.get('GATEKEEPER_URL', 'http://gatekeeper:8080')
BOT_SERVICE_URL = os.environ.get('BOT_SERVICE_URL', 'http://bot:8080')

RELAY_URL = os.environ.get('REGISTERABOT_RELAY_URL', '')
# Legacy single-bot env vars (backward compat)
LEGACY_BOT_SLUG = os.environ.get('REGISTERABOT_BOT_SLUG', '')
LEGACY_API_KEY = os.environ.get('REGISTERABOT_API_KEY', '')

# Host mode (preferred): ONE socket to /ws/host/:account carries all our bots. When these are
# set, connect(slug)/disconnect(slug) become attach/detach on that socket instead of a per-bot
# WS. See registerabot docs/HOST-CONNECTION-PROTOCOL.md.
ACCOUNT_ID = os.environ.get('REGISTERABOT_ACCOUNT_ID', '')
ACCOUNT_KEY = os.environ.get('REGISTERABOT_ACCOUNT_KEY', '')
HOST_MODE = bool(RELAY_URL and ACCOUNT_ID and ACCOUNT_KEY)

# Application-level ping on the host control channel. The WS library's own ping proves the
# socket is open; this proves the RELAY is still answering us. Stale pong => routing suspect.
HOST_PING_SECONDS = int(os.environ.get('REGISTERABOT_HOST_PING_SECONDS', '30'))

# Upper bound the connector waits for a bot turn. The REAL budget is per-profile and
# enforced by the gatekeeper (which returns within it); this ceiling just has to exceed
# the largest profile timeout so httpx never throws ReadTimeout before the gatekeeper replies.
BOT_RESPONSE_CEILING = float(os.environ.get('INSTAR_BOT_TIMEOUT_CEILING', '900'))

RECONNECT_DELAY = 5  # seconds

# Conversations marked persistent by gatekeeper (conversation_id -> True)
# Once a persistence-granting passphrase is matched, the conversation stays persistent
_persistent_conversations: dict[str, bool] = {}


def _log_task_exception(task: asyncio.Task) -> None:
    """Log exceptions from fire-and-forget tasks."""
    if not task.cancelled() and task.exception():
        log.error("task_exception", error=str(task.exception()))


# =============================================================================
# ENVELOPE HANDLING
# =============================================================================

def extract_messages_from_envelope(envelope: dict) -> tuple[list[dict], str]:
    """
    Extract the message list from an inbound envelope.

    Returns (messages, source_info) where messages is the array of
    {role, content} dicts and source_info is a string for logging.

    Handles both formats:
    - WebSocket service clients: payload is a JSON string with messages inside
    - HTTP /ask path: messages array at top level of envelope
    """
    sender = envelope.get('from', {})
    source_info = f"{sender.get('kind', '?')}:{sender.get('slug', '?')}"

    # Try payload field first (WebSocket service clients send this way)
    payload_raw = envelope.get('payload')
    if payload_raw and isinstance(payload_raw, str):
        try:
            payload = json.loads(payload_raw)
            if isinstance(payload, dict) and 'messages' in payload:
                return payload['messages'], source_info
        except (json.JSONDecodeError, TypeError):
            pass

    # Fall back to top-level messages (HTTP /ask path spreads body into envelope)
    messages = envelope.get('messages')
    if isinstance(messages, list):
        return messages, source_info

    return [], source_info


def extract_user_context_from_envelope(envelope: dict) -> dict:
    """Pull the generic user_context blob an interface may attach (e.g.
    {operator_id, user_id, permissions}). Opaque to the connector — it is
    forwarded verbatim to the gatekeeper, which forwards it to the bot. The
    bot's expert layer decides what the keys mean."""
    payload_raw = envelope.get('payload')
    if isinstance(payload_raw, str):
        try:
            payload = json.loads(payload_raw)
        except json.JSONDecodeError:
            payload = None
        if isinstance(payload, dict) and isinstance(payload.get('user_context'), dict):
            return payload['user_context']
    elif isinstance(payload_raw, dict) and isinstance(payload_raw.get('user_context'), dict):
        return payload_raw['user_context']
    # HTTP /ask path spreads the body into the envelope
    uc = envelope.get('user_context')
    return uc if isinstance(uc, dict) else {}


def build_response_envelope(
    original_envelope: dict,
    bot_slug: str,
    response_text: str,
    spoken: str = '',
    attachments: list[dict] = None,
    audio_base64: str = '',
) -> dict:
    """
    Build an outbound envelope to send back through the relay.

    Mirrors the original envelope's routing: from=bot, to=original sender.
    Optionally includes file attachments and inline voice audio in the payload.
    Embedding audio here (rather than a separate frame) lets it reach clients on
    the synchronous /ask path, e.g. the platform's in-browser assistant.
    """
    payload = {'reply': response_text}
    if spoken:
        payload['spoken'] = spoken
    if attachments:
        payload['attachments'] = attachments
    if audio_base64:
        payload['audio'] = {
            'base64': audio_base64,
            'format': 'mp3',
            'mime_type': 'audio/mpeg',
        }

    return {
        'v': 1,
        'type': 'final_response',
        'session_id': original_envelope.get('session_id', ''),
        'timestamp': int(time.time() * 1000),
        'from': {'kind': 'bot', 'slug': bot_slug},
        'to': original_envelope.get('from', {}),
        'encrypted': False,
        'payload': json.dumps(payload),
    }


def build_status_envelope(original_envelope: dict, bot_slug: str, text: str) -> dict:
    """A lightweight progress frame ('working on it…') routed back to the sender.
    Optional/additive — clients that don't understand 'status' just ignore it."""
    return {
        'v': 1,
        'type': 'status',
        'session_id': original_envelope.get('session_id', ''),
        'timestamp': int(time.time() * 1000),
        'from': {'kind': 'bot', 'slug': bot_slug},
        'to': original_envelope.get('from', {}),
        'encrypted': False,
        'payload': json.dumps({'status': text}),
    }


def build_video_envelope(original_envelope: dict, bot_slug: str, video_base64: str) -> dict:
    """A talking-head video frame (MP4) routed back to the sender. Optional/additive —
    clients without a <VideoAvatar> ignore it. Arrives well after text (render latency)."""
    return {
        'v': 1,
        'type': 'video',
        'session_id': original_envelope.get('session_id', ''),
        'timestamp': int(time.time() * 1000),
        'from': {'kind': 'bot', 'slug': bot_slug},
        'to': original_envelope.get('from', {}),
        'encrypted': False,
        'payload': json.dumps({'format': 'mp4', 'base64': video_base64}),
    }


def build_idle_envelope(original_envelope: dict, bot_slug: str, video_base64: str) -> dict:
    """A looping mouth-closed idle clip for the avatar circle (browser loops it between
    turns so she's always present). Optional/additive; clients ignore unknown types."""
    return {
        'v': 1,
        'type': 'idle',
        'session_id': original_envelope.get('session_id', ''),
        'timestamp': int(time.time() * 1000),
        'from': {'kind': 'bot', 'slug': bot_slug},
        'to': original_envelope.get('from', {}),
        'encrypted': False,
        'payload': json.dumps({'format': 'mp4', 'base64': video_base64}),
    }


_idle_b64_cache: dict = {}  # avatar_id -> base64 idle mp4 (fetched once from the face-renderer)


async def _send_idle_frame(ws, envelope: dict, bot_slug: str, avatar_id: str) -> None:
    """Fetch (and cache) the avatar's idle loop and send it so the browser can loop it
    between turns. First fetch triggers a one-time render on the face-renderer."""
    import base64
    face_url = os.environ.get('FACE_RENDERER_URL', 'http://instar-face-renderer:8088')
    try:
        b64 = _idle_b64_cache.get(avatar_id)
        if not b64:
            async with httpx.AsyncClient(timeout=120.0) as client:
                r = await client.get(f"{face_url}/avatars/{avatar_id}/idle")
            if r.status_code != 200:
                return
            b64 = base64.b64encode(r.content).decode()
            _idle_b64_cache[avatar_id] = b64
        await ws.send(json.dumps(build_idle_envelope(envelope, bot_slug, b64)))
    except Exception as e:
        log.debug("idle_frame_error", error=str(e), bot=bot_slug)


async def _get_face_config(voice_group_id: str) -> dict:
    """Fetch {face_enabled, face_avatar_id} for a voice group (empty on any failure)."""
    admin = os.environ.get('ADMIN_API_URL', 'http://gatekeeper:8090')
    try:
        async with httpx.AsyncClient(timeout=8.0) as client:
            r = await client.get(f"{admin}/api/voice-groups/{voice_group_id}/face")
            if r.status_code == 200:
                return r.json()
    except Exception:
        pass
    return {}


async def _render_and_send_video(ws, envelope: dict, bot_slug: str, audio_b64: str, avatar_id: str) -> None:
    """Decoupled talking-head avatar: render the turn's audio through the avatar (~3-4s at
    FPS=12) and send a 'video' frame carrying the lip-synced clip (with audio baked in).
    Fire-and-forget; any failure is swallowed so it never affects the chat."""
    import base64
    face_url = os.environ.get('FACE_RENDERER_URL', 'http://instar-face-renderer:8088')
    try:
        async with httpx.AsyncClient(timeout=600.0) as client:
            resp = await client.post(
                f"{face_url}/render",
                files={'audio': ('speech.wav', base64.b64decode(audio_b64), 'audio/mpeg')},
                data={'avatar_id': avatar_id},
            )
            if resp.status_code != 200:
                log.debug("video_render_failed", status=resp.status_code, bot=bot_slug)
                return
            await ws.send(json.dumps(build_video_envelope(envelope, bot_slug, base64.b64encode(resp.content).decode())))
            log.info("video_frame_sent", bytes=len(resp.content), bot=bot_slug)
    except Exception as e:
        log.debug("video_render_error", error=str(e), bot=bot_slug)


# Friendly progress wording per tool (tweak freely). Falls back to a cleaned name.
_FRIENDLY_TOOL = {
    'vgx_workspace_snapshot': 'loading your workspace',
    'vgx_snapshot_query': 'checking your data',
    'vendgogh_api': 'querying VendGogh',
    'vgx_read_result': 'reading the results',
    'vgx_set_operator': 'switching operator',
    'vgx_expert_search': 'checking what I know',
    'vgx_expert_read': 'checking what I know',
    'vgx_list_backups': 'looking up backups',
    'vgx_get_backup': 'loading a backup',
    'ToolSearch': 'thinking',
}


def _friendly_api_call(method: str, path: str) -> str:
    """Narrate a vendgogh_api call by its method+path — so the plan dry-run, the
    template fetch, reads, etc. each get their own phrase instead of a generic one."""
    p = (path or '').lower().split('?')[0]
    m = (method or '').upper()
    if '/ai/plan/template' in p:
        return 'getting the plan form'
    if p.rstrip('/').endswith('/ai/plan') and m == 'POST':
        return 'dry-running your plan'
    if '/ai/plan' in p:
        return 'checking the plan'
    if '/ai/apply' in p:
        return 'applying the plan'
    if '/operators' in p:
        return 'loading operators'
    # generic: last non-numeric path segment as the "resource"
    segs = [s for s in p.strip('/').split('/') if s and not s.isdigit()]
    tail = (segs[-1] if segs else 'data').replace('-', ' ')
    if m == 'GET':
        return f'reading {tail}'
    if m in ('POST', 'PUT', 'PATCH', 'DELETE'):
        return f'updating {tail}'
    return 'querying VendGogh'


def _friendly_step(raw: str, tool_input: dict) -> str:
    """Map a tool_use (name + input) to a human progress phrase."""
    name = raw.split('__')[-1]  # strip mcp__instar__ prefix
    if name == 'vendgogh_api':
        ti = tool_input or {}
        return _friendly_api_call(str(ti.get('method', '')), str(ti.get('path', '')))
    if name in _FRIENDLY_TOOL:
        return _FRIENDLY_TOOL[name]
    return name.replace('vgx_', '').replace('_', ' ').strip() or 'working'


def _find_tool_uses(event) -> list[tuple[str, dict]]:
    """Walk a claude stream-json event for (name, input) of tool_use blocks."""
    uses: list[tuple[str, dict]] = []

    def walk(o):
        if isinstance(o, dict):
            if o.get('type') == 'tool_use' and o.get('name'):
                uses.append((o['name'], o.get('input') or {}))
            for v in o.values():
                walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)

    walk(event)
    return uses


async def _stream_status_frames(bot_slug: str, conversation_id: str, relay_ws, envelope: dict) -> None:
    """Subscribe to the gatekeeper's terminal output stream and forward the bot's
    tool-use as 'status' frames to the sender, in parallel with the blocking /process.
    Purely additive: wrapped so it can never disturb the final response. Cancelled by
    the caller once the turn's final is in hand.
    """
    admin = os.environ.get('ADMIN_API_URL', 'http://gatekeeper:8090')
    ws_url = admin.replace('https://', 'wss://').replace('http://', 'ws://') + f'/ws/terminal/{bot_slug}'
    # Dedupe only CONSECUTIVE identical phrases — so each distinct step advances the
    # narration (a second snapshot query, the plan dry-run, etc.), but we don't spam
    # the same phrase twice in a row.
    last_phrase: str | None = None
    try:
        async with websockets.connect(ws_url, open_timeout=5) as tws:
            async for raw in tws:
                try:
                    msg = json.loads(raw)
                except Exception:
                    continue
                if msg.get('type') != 'output' or msg.get('conversation_id') != conversation_id:
                    continue
                for name, tool_input in _find_tool_uses(msg.get('event', {})):
                    # Prefer the bot's own stated intent ('why', <=15 words) — real
                    # narration. Fall back to the per-tool phrase only when absent.
                    why = ''
                    if isinstance(tool_input, dict):
                        why = str(tool_input.get('why') or '').strip()
                    phrase = why or _friendly_step(name, tool_input)
                    if phrase == last_phrase:
                        continue
                    last_phrase = phrase
                    try:
                        await relay_ws.send(json.dumps(build_status_envelope(envelope, bot_slug, phrase)))
                    except Exception:
                        return
    except asyncio.CancelledError:
        raise
    except Exception as e:
        log.debug("status_stream_ended", error=str(e), bot=bot_slug)


# =============================================================================
# MESSAGE HANDLER
# =============================================================================

async def handle_message(envelope: dict, ws, bot_slug: str) -> None:
    """
    Handle incoming RegisterABot message — route through gatekeeper and bot.

    Flow:
    1. Strip envelope, extract messages
    2. Flatten messages into text for gatekeeper (hints included as-is)
    3. Call gatekeeper /process with bot_slug for profile routing
    4. Runner responds via sync bridge (or legacy bot /chat fallback)
    5. Wrap response in envelope, send back through relay
    6. Record response with gatekeeper
    """
    session_id = envelope.get('session_id', '')
    messages, source_info = extract_messages_from_envelope(envelope)
    user_context = extract_user_context_from_envelope(envelope)

    if not messages:
        log.warning("empty_message", session=session_id, source=source_info, bot=bot_slug)
        return

    log.info(
        "message_received",
        session=session_id,
        source=source_info,
        bot=bot_slug,
        message_count=len(messages),
    )

    # Flatten messages into a single string for the gatekeeper.
    # System messages (including [HINTS: ...] from chat link passphrases)
    # are included — gatekeeper's sanitizer will strip the passphrases.
    # Also extract attachments from content blocks or explicit field.
    text_parts = []
    attachments = []
    for m in messages:
        content = m.get('content', '')
        if isinstance(content, str):
            text_parts.append(content)
        elif isinstance(content, list):
            # Content-block format: [{type: "text", text: "..."}, {type: "file", ...}]
            for block in content:
                btype = block.get('type', '')
                if btype == 'text':
                    text_parts.append(block.get('text', ''))
                elif btype == 'file':
                    attachments.append(block)
        # Also check explicit attachments field on the message
        for a in m.get('attachments', []):
            attachments.append(a)

    full_message = '\n'.join(text_parts)

    # Use session_id as conversation_id (groups multi-turn conversations)
    conversation_id = f"registerabot:{session_id}"
    user_id = source_info

    try:
        async with httpx.AsyncClient() as client:
            # 1. Process through gatekeeper — include bot_slug for profile routing
            gk_body = {
                'conversation_id': conversation_id,
                'source': 'registerabot',
                'bot_slug': bot_slug,
                'user_id': user_id,
                'message': full_message,
            }
            if user_context:
                gk_body['user_context'] = user_context
            if attachments:
                gk_body['attachments'] = attachments

            # Forward the bot's tool-use as 'status' progress frames while the turn
            # runs. Additive side-channel — the final response below is unaffected.
            status_task = asyncio.create_task(
                _stream_status_frames(bot_slug, conversation_id, ws, envelope)
            )
            try:
                gk_response = await client.post(
                    f"{GATEKEEPER_URL}/process",
                    json=gk_body,
                    timeout=BOT_RESPONSE_CEILING,
                )
            finally:
                status_task.cancel()

            if gk_response.status_code != 200:
                log.error("gatekeeper_error", status=gk_response.status_code, body=gk_response.text, bot=bot_slug)
                await send_response(ws, envelope, bot_slug, "I had trouble processing that. Please try again.")
                return

            gk_data = gk_response.json()
            log.debug("gatekeeper_response", data=gk_data, bot=bot_slug)

            # Track persistence — once granted, stays granted for this conversation
            if gk_data.get('persistent'):
                _persistent_conversations[conversation_id] = True
                log.info("persistence_granted", conversation=conversation_id)
            is_persistent = _persistent_conversations.get(conversation_id, False)

            # If message was only passphrases (no actual content)
            if gk_data.get('is_empty'):
                if gk_data.get('newly_unlocked'):
                    tools = ', '.join(gk_data['newly_unlocked'])
                    await send_response(ws, envelope, bot_slug, f"Tools now available: {tools}. What would you like to do?")
                return

            # Gatekeeper short-circuit (runner response or no-tools message)
            if gk_data.get('response'):
                text_attachments = [a for a in gk_data.get('attachments', []) if a.get('type') != 'audio']
                spoken = gk_data.get('spoken', '')
                voice_group_id = gk_data.get('voice_group_id', '')

                # Is a talking-head avatar enabled? If so, the rendered video carries the
                # audio (baked in) — so we must NOT also send standalone audio, or it plays
                # twice. Check up front so we can decide before sending anything.
                face_cfg = await _get_face_config(voice_group_id) if voice_group_id else {}
                face_on = bool(face_cfg.get('face_enabled') and face_cfg.get('face_avatar_id'))

                # Option A: generate TTS inline and embed it in the response envelope so
                # audio reaches synchronous /ask clients (the platform assistant). Also emit
                # the legacy separate frame below for persistent WS clients (the website).
                audio_b64 = ''
                if spoken and voice_group_id:
                    audio_b64 = await _generate_tts(spoken, voice_group_id)

                # Suppress standalone audio when the avatar is on (the video is the audio).
                embed_audio = audio_b64 if (audio_b64 and not face_on) else ''

                await send_response(
                    ws, envelope, bot_slug, gk_data['response'],
                    spoken,
                    attachments=text_attachments or None,
                    audio_base64=embed_audio,
                )

                if embed_audio:
                    await _send_audio_frame(ws, envelope, bot_slug, embed_audio)

                # Talking-head avatar (face_enabled). Send the idle loop first (browser
                # loops it while she "thinks"), then the talking clip when it renders — both
                # fire-and-forget so neither blocks the text.
                if face_on:
                    avatar_id = face_cfg['face_avatar_id']
                    asyncio.create_task(_send_idle_frame(ws, envelope, bot_slug, avatar_id))
                    if audio_b64:
                        asyncio.create_task(
                            _render_and_send_video(ws, envelope, bot_slug, audio_b64, avatar_id)
                        )

                return

            # 2. Legacy fallback: call bot service /chat directly
            bot_response = await client.post(
                f"{BOT_SERVICE_URL}/chat",
                json={
                    'session_id': gk_data.get('session_id'),
                    'message': gk_data.get('cleaned_message'),
                    'available_tools': gk_data.get('available_tools', []),
                    'notifications': gk_data.get('notifications', []),
                    'mcp_token': gk_data.get('mcp_token'),
                    'conversation_id': conversation_id if is_persistent else None,
                    'persistent': is_persistent,
                },
                timeout=BOT_RESPONSE_CEILING,
            )

            if bot_response.status_code != 200:
                log.error("bot_error", status=bot_response.status_code, body=bot_response.text, bot=bot_slug)
                await send_response(ws, envelope, bot_slug, "I encountered an error processing that.")
                return

            bot_data = bot_response.json()
            response_text = bot_data.get('response', '')

            # 3. Send response back through relay
            if response_text:
                await send_response(ws, envelope, bot_slug, response_text)
                log.debug("response_sent", session=session_id, length=len(response_text), bot=bot_slug)

                # 4. Record response to gatekeeper history
                await client.post(
                    f"{GATEKEEPER_URL}/record-response",
                    json={
                        'session_id': gk_data.get('session_id'),
                        'response': response_text,
                    },
                    timeout=10.0,
                )

    except httpx.ConnectError as e:
        log.error("connection_error", error=str(e), bot=bot_slug)
        try:
            await send_response(ws, envelope, bot_slug, "I'm having trouble connecting to my internal services. Please try again.")
        except Exception:
            log.warning("failed_to_send_error_response")
    except Exception as e:
        log.error("message_handler_error", error=str(e), bot=bot_slug)
        import traceback
        traceback.print_exc()
        try:
            await send_response(ws, envelope, bot_slug, "I encountered an error. Please try again.")
        except Exception:
            log.warning("failed_to_send_error_response")


ADMIN_API_URL = os.environ.get('ADMIN_API_URL', 'http://gatekeeper:8090')


async def _generate_tts(spoken: str, voice_group_id: str) -> str:
    """Generate TTS audio via the gatekeeper admin API. Returns base64 MP3, '' on failure.

    Requests MP3 so the blob stays under the relay's 1 MiB WS frame cap (raw WAV blows it
    on replies longer than ~16s, taking the whole envelope down with it)."""
    try:
        import base64
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"{ADMIN_API_URL}/api/tts/generate",
                json={"voice_group_id": voice_group_id, "text": spoken, "format": "mp3"},
                timeout=60.0,
            )
            if resp.status_code != 200:
                log.warning("tts_generate_failed", status=resp.status_code)
                return ''
            return base64.b64encode(resp.content).decode()
    except Exception as e:
        log.warning("tts_generate_error", error=str(e))
        return ''


async def _send_audio_frame(ws, original_envelope: dict, bot_slug: str, audio_b64: str) -> None:
    """Send audio as a separate WS frame — for persistent WS clients (the website)."""
    try:
        audio_envelope = {
            'v': 1,
            'type': 'audio_response',
            'session_id': original_envelope.get('session_id', ''),
            'timestamp': int(time.time() * 1000),
            'from': {'kind': 'bot', 'slug': bot_slug},
            'to': original_envelope.get('from', {}),
            'encrypted': False,
            'payload': json.dumps({
                'audio_base64': audio_b64,
                'format': 'mp3',
                'mime_type': 'audio/mpeg',
            }),
        }
        await ws.send(json.dumps(audio_envelope))
        log.info("audio_frame_sent", bot=bot_slug)
    except websockets.ConnectionClosed:
        log.warning("audio_frame_ws_closed", bot=bot_slug)
    except Exception as e:
        log.warning("audio_frame_error", error=str(e), bot=bot_slug)


async def send_response(ws, original_envelope: dict, bot_slug: str, text: str, spoken: str = '', attachments: list[dict] = None, audio_base64: str = '') -> None:
    """Wrap text in an envelope and send it back through the relay WebSocket."""
    try:
        response_envelope = build_response_envelope(original_envelope, bot_slug, text, spoken, attachments=attachments, audio_base64=audio_base64)
        await ws.send(json.dumps(response_envelope))
    except websockets.ConnectionClosed:
        log.warning("send_on_closed_ws", bot=bot_slug)


# =============================================================================
# CONNECTION MANAGER
# =============================================================================

class SlugConnection:
    """State for a single bot slug's relay connection."""

    def __init__(self, slug: str, api_key: str, relay_url: str):
        self.slug = slug
        self.api_key = api_key
        self.relay_url = relay_url
        self.ws: websockets.WebSocketClientProtocol | None = None
        self.task: asyncio.Task | None = None
        self.connected: bool = False
        self.connected_since: float | None = None
        self._shutdown: bool = False

    def to_dict(self) -> dict:
        uptime = None
        if self.connected_since:
            uptime = int(time.time() - self.connected_since)
        return {
            'slug': self.slug,
            'connected': self.connected,
            'connected_since': self.connected_since,
            'uptime_seconds': uptime,
        }


class HostConnection:
    """ONE WebSocket to the relay /ws/host/:account carrying MANY of our owned bots.

    connect(slug)/disconnect(slug) become attach/detach control frames; inbound envelopes are
    demuxed by to.slug into handle_message (which is already slug-keyed). Auto-reconnects and
    re-attaches everything it believes it's serving. See docs/HOST-CONNECTION-PROTOCOL.md.
    """

    def __init__(self, account_id: str, account_key: str, relay_url: str):
        self.account_id = account_id
        self.account_key = account_key
        self.relay_url = relay_url
        self.ws: websockets.WebSocketClientProtocol | None = None
        # DESIRED vs CONFIRMED. `attached` is what we've been asked to serve; `confirmed` is
        # what the relay actually ACKed. They can diverge — the relay can DENY a slug — and
        # when that happened we reported connected:true for 17 hours while every message for
        # that bot was black-holed. Never report health from `attached` alone.
        self.attached: set[str] = set()
        self.confirmed: set[str] = set()
        self.connected: bool = False
        self.connected_since: float | None = None
        self.last_pong: float | None = None
        self._task: asyncio.Task | None = None
        self._ping_task: asyncio.Task | None = None
        self._shutdown: bool = False

    def start(self) -> None:
        if self._task is None:
            self._task = asyncio.create_task(self._loop())
            self._task.add_done_callback(_log_task_exception)

    async def stop(self) -> None:
        self._shutdown = True
        if self._task and not self._task.done():
            self._task.cancel()
        if self.ws:
            try:
                await self.ws.close()
            except Exception:
                pass

    def attach(self, slug: str) -> bool:
        """Ask the relay to route `slug` to us.

        Re-sends the attach frame whenever the slug isn't CONFIRMED, even if we already
        wanted it. That's what makes the gatekeeper's 60s keepalive self-healing: previously
        this returned early on `slug in self.attached`, so once an attach was denied nothing
        ever retried it and the bot stayed silently unreachable.
        """
        already = slug in self.attached
        self.attached.add(slug)
        if already and slug in self.confirmed:
            return False
        if self.connected and self.ws:
            asyncio.create_task(self._send({'type': 'attach', 'slugs': [slug]}))
        log.info("host_attach", slug=slug, retry=already)
        return not already

    def detach(self, slug: str) -> bool:
        if slug not in self.attached:
            return False
        self.attached.discard(slug)
        self.confirmed.discard(slug)
        if self.connected and self.ws:
            asyncio.create_task(self._send({'type': 'detach', 'slugs': [slug]}))
        log.info("host_detach", slug=slug)
        return True

    def is_attached(self, slug: str) -> bool:
        """Routing-truth, not intent: only a relay-CONFIRMED slug counts as reachable."""
        return slug in self.confirmed

    async def _send(self, obj: dict) -> None:
        try:
            if self.ws:
                await self.ws.send(json.dumps(obj))
        except Exception as e:
            log.debug("host_send_failed", error=str(e))

    async def _ping_loop(self) -> None:
        """Account-channel heartbeat + per-slug ROUTING reconciliation.

        The ping alone proves only that the relay answers us on the host socket — it would
        have stayed green through the whole 2026-08-09 outage, because that socket was fine
        and it was the per-slug binding that was broken. So the useful half is `slugs`: the
        relay reports which slugs it can actually serve, and we compare that to what we
        believe we're serving. Any slug we want but the relay doesn't list gets re-attached
        automatically — the exact repair that previously needed a manual detach/attach.
        """
        while not self._shutdown and self.connected:
            try:
                await asyncio.sleep(HOST_PING_SECONDS)
                if not (self.connected and self.ws):
                    return
                await self._send({'type': 'ping', 'ts': int(time.time() * 1000)})
                await self._send({'type': 'slugs'})
                if self.pong_age() is not None and self.pong_age() > HOST_PING_SECONDS * 3:
                    log.warning("host_pong_stale", seconds=int(self.pong_age()),
                                note="relay not answering pings; routing is suspect")
            except asyncio.CancelledError:
                return
            except Exception as e:
                log.debug("host_ping_failed", error=str(e))

    async def _reconcile(self, relay_slugs: list) -> None:
        """Relay's view is authoritative. Re-attach anything we want that it isn't serving."""
        relay = set(relay_slugs or [])
        self.confirmed = self.attached & relay
        missing = self.attached - relay
        if missing:
            log.warning("host_routing_drift", missing=sorted(missing),
                        note="relay is not serving slugs we believe we serve; re-attaching")
            await self._send({'type': 'attach', 'slugs': sorted(missing)})

    def pong_age(self) -> float | None:
        return None if self.last_pong is None else time.time() - self.last_pong

    def healthy(self) -> bool:
        """Connected AND the relay is still answering on the control channel."""
        age = self.pong_age()
        return bool(self.connected) and (age is None or age <= HOST_PING_SECONDS * 3)

    async def _loop(self) -> None:
        while not self._shutdown:
            try:
                ws_url = f"{self.relay_url}/ws/host/{self.account_id}?key={self.account_key}"
                log.info("host_connecting", url=self.relay_url, account=self.account_id)
                async with websockets.connect(
                    ws_url, ping_interval=20, ping_timeout=90, close_timeout=10,
                ) as ws:
                    self.ws = ws
                    self.connected = True
                    self.connected_since = time.time()
                    self.last_pong = time.time()
                    # A new socket confirms NOTHING until the relay ACKs the re-attach.
                    self.confirmed.clear()
                    if self.attached:  # re-attach after a reconnect
                        await self._send({'type': 'attach', 'slugs': sorted(self.attached)})
                    log.info("host_connected", serving=len(self.attached))
                    if self._ping_task is None or self._ping_task.done():
                        self._ping_task = asyncio.create_task(self._ping_loop())
                        self._ping_task.add_done_callback(_log_task_exception)

                    async for raw in ws:
                        if self._shutdown:
                            break
                        try:
                            data = json.loads(raw)
                        except json.JSONDecodeError:
                            log.warning("host_invalid_json", raw=raw[:200])
                            continue
                        mtype = data.get('type', '')
                        if mtype == 'chat_request':
                            slug = (data.get('to') or {}).get('slug')
                            if not slug:
                                log.warning("host_msg_no_slug")
                                continue
                            task = asyncio.create_task(handle_message(data, ws, slug))
                            task.add_done_callback(_log_task_exception)
                        elif mtype == 'attached':
                            ok = data.get('ok') or []
                            denied = data.get('denied') or []
                            # ACT on the reply, don't just log it. A denied slug must leave
                            # `confirmed` so /status stops claiming it's reachable, and must
                            # stay in `attached` so the keepalive keeps retrying it.
                            self.confirmed.update(ok)
                            self.confirmed.difference_update(denied)
                            if denied:
                                log.warning("host_attach_denied", denied=denied,
                                            note="relay refused these slugs; keepalive will retry")
                            log.info("host_attached", ok=ok, denied=denied,
                                     confirmed=sorted(self.confirmed))
                        elif mtype == 'error':
                            # An error naming a slug means we can no longer trust its routing.
                            slug = data.get('slug')
                            if slug:
                                self.confirmed.discard(slug)
                            log.warning("host_error", code=data.get('code'), slug=slug)
                        elif mtype == 'pong':
                            self.last_pong = time.time()
                        elif mtype == 'slugs':
                            # Routing truth from the relay — reconcile against our intent.
                            self.last_pong = time.time()
                            await self._reconcile(data.get('attached') or [])
                        elif 'error' in data:
                            log.warning("host_relay_error", error=data.get('error'))
            except asyncio.CancelledError:
                break
            except websockets.ConnectionClosed as e:
                log.warning("host_disconnected", code=e.code, reason=e.reason)
            except Exception as e:
                log.error("host_error_unexpected", error=str(e))
                import traceback
                traceback.print_exc()
            finally:
                self.ws = None
                self.connected = False
                self.connected_since = None
            if not self._shutdown:
                log.info("host_reconnecting", delay=RECONNECT_DELAY)
                await asyncio.sleep(RECONNECT_DELAY)


class ConnectionManager:
    """
    Manages WebSocket connection(s) to the RegisterABot relay.

    HOST_MODE (preferred): one shared host socket carries all bots — connect/disconnect a slug
    become attach/detach. Legacy: each bot_slug gets its own WS. Gatekeeper calls
    connect/disconnect on profile launch/stop either way.
    """

    def __init__(self):
        self._connections: dict[str, SlugConnection] = {}
        self._host: HostConnection | None = (
            HostConnection(ACCOUNT_ID, ACCOUNT_KEY, RELAY_URL) if HOST_MODE else None
        )

    def start_host(self) -> None:
        if self._host:
            self._host.start()

    def connect(self, slug: str, api_key: str, relay_url: str) -> bool:
        """
        Bring a bot slug online. Host mode: attach on the shared socket. Legacy: new WS.
        Returns False if already online.
        """
        if self._host:
            return self._host.attach(slug)   # account key authenticates; per-bot api_key unused

        if slug in self._connections and not self._connections[slug]._shutdown:
            return False

        conn = SlugConnection(slug=slug, api_key=api_key, relay_url=relay_url)
        conn.task = asyncio.create_task(self._relay_loop(conn))
        conn.task.add_done_callback(_log_task_exception)
        self._connections[slug] = conn

        log.info("slug_connecting", slug=slug)
        return True

    def disconnect(self, slug: str) -> bool:
        """
        Take a bot slug offline. Host mode: detach. Legacy: close its WS. False if not online.
        """
        if self._host:
            return self._host.detach(slug)

        conn = self._connections.pop(slug, None)
        if not conn:
            return False

        conn._shutdown = True
        if conn.task and not conn.task.done():
            conn.task.cancel()
        if conn.ws:
            asyncio.create_task(self._close_ws(conn.ws))

        log.info("slug_disconnected", slug=slug)
        return True

    def list_connections(self) -> list[dict]:
        """List all bots and their status.

        `connected` now means RELAY-CONFIRMED routing, not 'we asked for it' — the old
        version read self._host.attached, which is intent, so a slug the relay had denied
        still showed connected:true. `attach_requested` keeps the intent visible, and
        pong_age_seconds distinguishes a live control channel from a merely-open socket.
        """
        if self._host:
            h = self._host
            up = int(time.time() - h.connected_since) if h.connected_since else None
            age = h.pong_age()
            return [{
                'slug': s,
                'connected': h.connected and s in h.confirmed,
                'attach_requested': True,
                'relay_confirmed': s in h.confirmed,
                'socket_connected': h.connected,
                'via': 'host',
                'uptime_seconds': up,
                'pong_age_seconds': None if age is None else int(age),
                'healthy': h.healthy() and s in h.confirmed,
            } for s in sorted(h.attached)]
        return [conn.to_dict() for conn in self._connections.values()]

    def is_connected(self, slug: str) -> bool:
        if self._host:
            return self._host.connected and self._host.is_attached(slug)
        conn = self._connections.get(slug)
        return conn.connected if conn else False

    async def shutdown_all(self):
        """Disconnect all bots (on app shutdown)."""
        if self._host:
            await self._host.stop()
            return
        slugs = list(self._connections.keys())
        for slug in slugs:
            self.disconnect(slug)

    async def _close_ws(self, ws):
        try:
            await ws.close()
        except Exception:
            pass

    async def _relay_loop(self, conn: SlugConnection) -> None:
        """
        Persistent WebSocket connection to relay for one bot slug.
        Auto-reconnects on disconnect.
        """
        while not conn._shutdown:
            try:
                ws_url = f"{conn.relay_url}/ws/bot/{conn.slug}?key={conn.api_key}"
                log.info("relay_connecting", slug=conn.slug, url=conn.relay_url)

                async with websockets.connect(
                    ws_url,
                    ping_interval=20,   # keep the relay WS alive during long bot turns
                    ping_timeout=90,    # tolerate slow (tool-heavy) processing
                    close_timeout=10,
                ) as ws:
                    conn.ws = ws
                    conn.connected = True
                    conn.connected_since = time.time()
                    log.info("relay_connected", slug=conn.slug)

                    async for raw in ws:
                        if conn._shutdown:
                            break
                        try:
                            data = json.loads(raw)
                        except json.JSONDecodeError:
                            log.warning("invalid_json", raw=raw[:200], slug=conn.slug)
                            continue

                        msg_type = data.get('type', '')
                        if msg_type == 'chat_request':
                            task = asyncio.create_task(handle_message(data, ws, conn.slug))
                            task.add_done_callback(_log_task_exception)
                        elif 'error' in data:
                            log.warning("relay_error", error=data.get('error'), slug=conn.slug)
                        else:
                            log.debug("unhandled_message_type", type=msg_type, slug=conn.slug)

            except asyncio.CancelledError:
                break
            except websockets.ConnectionClosed as e:
                log.warning("relay_disconnected", code=e.code, reason=e.reason, slug=conn.slug)
            except (OSError, websockets.InvalidURI, websockets.InvalidHandshake) as e:
                log.error("relay_connection_failed", error=str(e), slug=conn.slug)
            except Exception as e:
                log.error("relay_unexpected_error", error=str(e), slug=conn.slug)
                import traceback
                traceback.print_exc()
            finally:
                conn.ws = None
                conn.connected = False
                conn.connected_since = None

            if not conn._shutdown:
                log.info("relay_reconnecting", delay=RECONNECT_DELAY, slug=conn.slug)
                await asyncio.sleep(RECONNECT_DELAY)


# Singleton
connection_manager = ConnectionManager()


# =============================================================================
# LIFESPAN
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown logic."""
    log.info("registerabot_api_starting")

    if HOST_MODE:
        # One shared socket for all bots; gatekeeper attaches slugs via /slugs/{slug}/connect
        # as profiles launch. Start it now (idle until the first attach).
        connection_manager.start_host()
        log.info("host_mode_started", account=ACCOUNT_ID)
        if LEGACY_BOT_SLUG:
            connection_manager.connect(LEGACY_BOT_SLUG, LEGACY_API_KEY, RELAY_URL)
    elif RELAY_URL and LEGACY_BOT_SLUG and LEGACY_API_KEY:
        # Legacy single-bot mode: one WS per bot
        connection_manager.connect(
            slug=LEGACY_BOT_SLUG,
            api_key=LEGACY_API_KEY,
            relay_url=RELAY_URL,
        )
        log.info("legacy_bot_connected", slug=LEGACY_BOT_SLUG)
    elif not RELAY_URL:
        log.warning("no_relay_url", message="REGISTERABOT_RELAY_URL not set — waiting for /slugs/{slug}/connect calls")

    yield

    # Shutdown all connections
    await connection_manager.shutdown_all()
    log.info("registerabot_api_stopped")


# =============================================================================
# APP
# =============================================================================

app = FastAPI(title="RegisterABot Interface Tool", lifespan=lifespan)


# =============================================================================
# SLUG MANAGEMENT API
# =============================================================================

@app.post("/slugs/{slug}/connect")
async def connect_slug(slug: str, request: dict | None = None):
    """
    Connect to the relay as a specific bot slug.

    Called by gatekeeper on profile launch.

    Body:
        {
            "api_key": "sk_xxx",
            "relay_url": "wss://relay.registerabot.com"  (optional, falls back to env)
        }
    """
    if request is None:
        request = {}

    api_key = request.get('api_key', '')
    relay_url = request.get('relay_url', RELAY_URL)

    if not api_key:
        return JSONResponse({"error": "api_key required"}, status_code=400)
    if not relay_url:
        return JSONResponse({"error": "relay_url not configured"}, status_code=400)

    started = connection_manager.connect(slug=slug, api_key=api_key, relay_url=relay_url)
    if not started:
        return {"status": "already_connected", "slug": slug}

    return {"status": "connecting", "slug": slug}


@app.post("/slugs/{slug}/disconnect")
async def disconnect_slug(slug: str):
    """
    Disconnect a bot slug from the relay.

    Called by gatekeeper on profile stop.
    """
    stopped = connection_manager.disconnect(slug)
    if not stopped:
        return {"status": "not_connected", "slug": slug}

    return {"status": "disconnected", "slug": slug}


@app.get("/slugs")
async def list_slugs():
    """List all connected bot slugs and their status."""
    return {"slugs": connection_manager.list_connections()}


# =============================================================================
# HEALTH & STATUS
# =============================================================================

@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "ok"}


@app.get("/status")
async def status():
    """Get RegisterABot connection status (all slugs)."""
    connections = connection_manager.list_connections()
    return {
        "relay_url": RELAY_URL,
        "connections": connections,
        "total_connected": sum(1 for c in connections if c['connected']),
    }


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 8091))
    uvicorn.run(app, host="0.0.0.0", port=port)
