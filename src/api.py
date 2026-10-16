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
import re
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import JSONResponse
import uvicorn
import httpx
import websockets
import structlog

# Relative: this runs as `python -m src.api` from /app, so `src` is the package and a
# bare `from turntaking...` resolves against /app, not against this file's directory.
from .turntaking.config import ConfigCache as TurnConfigCache
from .turntaking.detector import TurnDetector
from .turntaking.session import SessionRegistry as TurnSessionRegistry, TurnSession
from .cues import CueLibrary, class_for_tool


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
LLM_URL = os.environ.get('LLM_URL', 'http://llm:11434')
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

# SPOKEN TOOL NARRATION — the bot says what it is doing WHILE it does it.
# Status frames already carry the bot's own stated intent (the `why` it passes to every tool
# call), so the text is genuinely descriptive rather than a generic "working…". This voices
# it. OFF by default: one TTS call per distinct step, and a busy turn has ten — great for a
# demo, expensive as a default. Silent unless the profile has a voice group.
NARRATE_TOOLS = os.environ.get('REGISTERABOT_NARRATE_TOOLS', '').lower() in ('1', 'true', 'yes')
# Skip trivially short phrases — narrating "thinking" is noise, not information.
NARRATE_MIN_CHARS = int(os.environ.get('REGISTERABOT_NARRATE_MIN_CHARS', '12'))

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


def extract_assertion_from_envelope(envelope: dict) -> str:
    """Pull the signed AI action assertion, if the sender attached one.

    Deliberately SEPARATE from user_context. user_context is forwarded into the bot's
    prompt, and this is a live credential: anything in the prompt can be quoted back,
    summarised, or talked out of the model. It is handed to the gatekeeper as its own
    field, bound to the session, and re-attached by the tool executor at call time — the
    bot acts under it without ever holding it, exactly as it never holds an API key.
    """
    payload_raw = envelope.get('payload')
    if isinstance(payload_raw, str):
        try:
            payload = json.loads(payload_raw)
        except json.JSONDecodeError:
            payload = None
        if isinstance(payload, dict) and isinstance(payload.get('assertion'), str):
            return payload['assertion']
    elif isinstance(payload_raw, dict) and isinstance(payload_raw.get('assertion'), str):
        return payload_raw['assertion']
    a = envelope.get('assertion')
    return a if isinstance(a, str) else ''


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


def build_reply_delta_envelope(original_envelope: dict, bot_slug: str, text: str) -> dict:
    """Her reply so far, while she is still writing it.

    Carries the WHOLE text each time rather than the increment. Deltas are small and a
    dropped frame would otherwise desynchronise the page permanently; sending the
    accumulated string makes every frame self-correcting and lets the page strip a
    marker that is still half-written.

    Optional/additive — a client that does not know 'reply_delta' ignores it and sees
    the final response exactly as before."""
    return {
        'v': 1,
        'type': 'reply_delta',
        'session_id': original_envelope.get('session_id', ''),
        'timestamp': int(time.time() * 1000),
        'from': {'kind': 'bot', 'slug': bot_slug},
        'to': original_envelope.get('from', {}),
        'encrypted': False,
        'payload': json.dumps({'text': text}),
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


def build_status_audio_envelope(original_envelope: dict, bot_slug: str,
                                text: str, audio_base64: str) -> dict:
    """Spoken narration for ONE status step.

    Deliberately its own frame type rather than audio bolted onto the status frame: the text
    must reach the UI immediately (so the progress line updates the moment the tool fires),
    while the audio arrives whenever TTS finishes, a second or two later. One frame carrying
    both would make the visible line wait on speech synthesis.

    Clients that don't know this type ignore it — but note any client whose terminal branch
    matches "has a payload" MUST skip it explicitly, or it will end the turn the way the
    decoupled audio_response frame used to.
    """
    return {
        'v': 1,
        'type': 'status_audio',
        'session_id': original_envelope.get('session_id', ''),
        'timestamp': int(time.time() * 1000),
        'from': {'kind': 'bot', 'slug': bot_slug},
        'to': original_envelope.get('from', {}),
        'encrypted': False,
        'payload': json.dumps({
            'status': text,
            'format': 'mp3',
            'mime_type': 'audio/mpeg',
            'base64': audio_base64,
        }),
    }


# slug -> voice_group_id ('' = none). Resolved once per slug; the profile's voice does not
# change mid-conversation, and this is on the hot path of every turn.
_voice_group_cache: dict[str, str] = {}


async def _get_voice_group(bot_slug: str) -> str:
    """The profile's effective voice group, for narrating tool calls as they happen.
    The /process response also carries this, but only AFTER the turn — too late to narrate."""
    if bot_slug in _voice_group_cache:
        return _voice_group_cache[bot_slug]
    admin = os.environ.get('ADMIN_API_URL', 'http://gatekeeper:8090')
    vg = ''
    try:
        async with httpx.AsyncClient() as client:
            r = await client.get(f"{admin}/api/profiles/{bot_slug}/voice", timeout=8.0)
            if r.status_code == 200:
                vg = str(r.json().get('voice_group_id') or '')
    except Exception as e:
        log.debug("voice_group_lookup_failed", slug=bot_slug, error=str(e))
    _voice_group_cache[bot_slug] = vg
    return vg


def build_video_envelope(original_envelope: dict, bot_slug: str, video_base64: str,
                         turn_id: str = '', seq: int = 0, final: bool = True) -> dict:
    """A talking-head video frame (MP4) routed back to the sender. Optional/additive —
    clients without a <VideoAvatar> ignore it. Arrives well after text (render latency).

    turn_id/seq/final carry the ordering a chunked turn needs (see the frame contract in
    project-instar docs/STREAMING_CHAT_BROKER.md). Unchunked turns send a single seq=0,
    final=true frame, which is what every client saw before chunking existed — a client
    that ignores these fields behaves exactly as it did.
    """
    return {
        'v': 1,
        'type': 'video',
        'session_id': original_envelope.get('session_id', ''),
        'timestamp': int(time.time() * 1000),
        'turn_id': turn_id,
        'seq': seq,
        'from': {'kind': 'bot', 'slug': bot_slug},
        'to': original_envelope.get('from', {}),
        'encrypted': False,
        'payload': json.dumps({'format': 'mp4', 'base64': video_base64,
                               'turn_id': turn_id, 'seq': seq, 'final': final}),
    }


_MEDIA_STATUS_TEXT = {
    'audio': 'finding her voice',
    'video': 'rendering her face',
}


def build_media_status_envelope(original_envelope: dict, bot_slug: str,
                                stage: str, state: str) -> dict:
    """Progress for the media that arrives AFTER the text — TTS, then the avatar render.

    The existing 'status' frame can't carry this: it narrates tool use during the turn and
    the portal only shows it while the turn is in flight, but media is produced after the
    final response, when that indicator is already gone. So it gets its own type, and the
    client shows it on the avatar rather than in the turn's status line.

    Optional/additive — a client that doesn't understand 'media_status' ignores it, which
    is exactly the behaviour before this existed.
    """
    return {
        'v': 1,
        'type': 'media_status',
        'session_id': original_envelope.get('session_id', ''),
        'timestamp': int(time.time() * 1000),
        'from': {'kind': 'bot', 'slug': bot_slug},
        'to': original_envelope.get('from', {}),
        'encrypted': False,
        'payload': json.dumps({
            'stage': stage,                 # 'audio' | 'video'
            'state': state,                 # 'start' | 'done'
            'text': _MEDIA_STATUS_TEXT.get(stage, stage),
        }),
    }


async def _send_media_status(ws, envelope: dict, bot_slug: str, stage: str, state: str) -> None:
    """Fire-and-forget progress ping. Never let telling someone we're working break the work."""
    try:
        await ws.send(json.dumps(build_media_status_envelope(envelope, bot_slug, stage, state)))
    except Exception as e:
        log.debug("media_status_send_failed", error=str(e), stage=stage, state=state, bot=bot_slug)


def build_idle_envelope(original_envelope: dict, bot_slug: str, video_base64: str,
                        variant: int = None, count: int = None) -> dict:
    """A looping mouth-closed idle clip for the avatar circle (browser loops it between
    turns so she's always present). Optional/additive; clients ignore unknown types.

    `variant` is what turns this from one clip into a playlist: the renderer builds several
    rest clips that all start and end on the same frame, and a stamped frame ACCUMULATES on
    the page instead of replacing what is there. Unstamped keeps the old behaviour, so a
    renderer that has only one loop still works.
    """
    payload = {'format': 'mp4', 'base64': video_base64}
    if variant is not None:
        payload['variant'] = variant
        payload['count'] = count
    return {
        'v': 1,
        'type': 'idle',
        'session_id': original_envelope.get('session_id', ''),
        'timestamp': int(time.time() * 1000),
        'from': {'kind': 'bot', 'slug': bot_slug},
        'to': original_envelope.get('from', {}),
        'encrypted': False,
        'payload': json.dumps(payload),
    }


def build_cue_clip_envelope(original_envelope: dict, bot_slug: str, clip: dict) -> dict:
    """One pre-rendered filler clip, for the page to hold until it wants it.

    Sent once per conversation, before the first cue can fire. A clip fetched at cue time
    would arrive after the moment it exists to cover, so the only useful delivery is early
    and in advance.
    """
    return {
        'v': 1,
        'type': 'cue_clip',
        'session_id': original_envelope.get('session_id', ''),
        'timestamp': int(time.time() * 1000),
        'from': {'kind': 'bot', 'slug': bot_slug},
        'to': original_envelope.get('from', {}),
        'encrypted': False,
        'payload': json.dumps(clip),
    }


def build_cue_envelope(original_envelope: dict, bot_slug: str, cue_class: str) -> dict:
    """A SUGGESTION that now might be a good moment for a filler clip of this class.

    Carries the class and nothing else — no phrase, no clip, no instruction to play. The
    page owns whether anything happens: it knows what is on screen, whether she is already
    speaking, and how long ago the last one went, and the server knows none of that. A
    turn with six tool calls sends six cues and should usually produce one clip.
    """
    return {
        'v': 1,
        'type': 'cue',
        'session_id': original_envelope.get('session_id', ''),
        'timestamp': int(time.time() * 1000),
        'from': {'kind': 'bot', 'slug': bot_slug},
        'to': original_envelope.get('from', {}),
        'encrypted': False,
        'payload': json.dumps({'cue_class': cue_class}),
    }


# The clip library, and which conversations have already been sent it. Keyed by session so
# a reconnecting page gets it again — it holds the clips in memory, not on disk.
CUES = CueLibrary()
_cues_sent: dict = {}
CUES_RESEND_AFTER = 1800.0


def cues_forget_session(session_id: str) -> None:
    """Forget that this conversation has the clips, so the next turn resends them.

    Called on [visitor_arrived], which is the page saying it just loaded. Clips live in
    browser memory as object URLs, but the session id is persisted in localStorage — so a
    reload keeps the conversation and loses the library, and "already sent" would have
    been true and wrong for the next half hour. The arrival marker is the only honest
    signal that the far end is empty.
    """
    if not session_id:
        return
    for key in [k for k in _cues_sent if k[0] == session_id]:
        _cues_sent.pop(key, None)


async def _send_cue_clips(ws, envelope: dict, bot_slug: str,
                          voice_group_id: str, avatar_id: str) -> None:
    """Build the library if needed, then push whatever exists to this page, once.

    Build runs off the turn path and takes ~25s the first time for a voice. The first turn
    of a brand-new voice therefore gets no cues, which is correct: better nothing than a
    clip that arrives late and plays over her answer.
    """
    try:
        session = envelope.get('session_id', '')
        last = _cues_sent.get((session, voice_group_id, avatar_id))
        if last and (time.time() - last) < CUES_RESEND_AFTER:
            return
        await CUES.ensure(voice_group_id, avatar_id)
        clips = CUES.manifest(voice_group_id, avatar_id)
        if not clips:
            return
        for clip in clips:
            await ws.send(json.dumps(build_cue_clip_envelope(envelope, bot_slug, clip)))
        _cues_sent[(session, voice_group_id, avatar_id)] = time.time()
        log.info("cue_clips_sent", bot=bot_slug, count=len(clips), avatar=avatar_id)
    except Exception as e:                                            # noqa: BLE001
        log.warning("cue_clips_error", error=str(e), bot=bot_slug)


# avatar_id -> (base64 idle mp4, fetched_at). TTL'd, NOT permanent: the idle is retuned
# and rebaked from the admin UI, and a process-lifetime cache meant a running container
# served the first clip it ever saw until someone restarted it.
_idle_b64_cache: dict = {}
IDLE_CACHE_TTL = float(os.environ.get('IDLE_CACHE_TTL', '60'))


async def _send_head_kind(ws, envelope: dict, bot_slug: str, kind: str) -> None:
    """Tell the page WHICH KIND OF HEAD this bot has: 'video' or 'rig'.

    The page cannot work this out for itself — `face_enabled` lives in the gatekeeper's
    database and the page is served by the relay — so it used to guess with a timer: wait
    four seconds, and if no rest clip had arrived, download the 3D rig. That guess costs
    three.js, three-vrm and an 11MB character on every video-head conversation where the
    first turn happened to be slow, which is exactly the cost the video head exists to
    avoid.

    Saying so explicitly removes the guess. Nothing speculative is ever fetched: a bot with
    a rendered head sends rest clips and the rig is never touched, and a bot without one
    says 'rig' and the page loads it at once instead of after a four-second wait.
    """
    try:
        await ws.send(json.dumps({
            'v': 1,
            'type': 'head',
            'session_id': envelope.get('session_id', ''),
            'timestamp': int(time.time() * 1000),
            'from': {'kind': 'bot', 'slug': bot_slug},
            'to': envelope.get('from', {}),
            'encrypted': False,
            'payload': json.dumps({'kind': kind}),
        }))
    except Exception as e:                                       # noqa: BLE001
        log.warning("head_kind_send_failed", error=str(e), kind=kind, bot=bot_slug)


async def _idle_variants(client, face_url: str, avatar_id: str) -> list:
    """Which rest clips this avatar has. [0] when the renderer has no playlist endpoint.

    Capped, because every clip crosses the WebSocket as base64 inside the first turn and
    the point of the playlist is variety, not completeness — six clips of twelve seconds is
    over a minute before anything repeats.
    """
    try:
        r = await client.get(f"{face_url}/avatars/{avatar_id}/idle-playlist")
        if r.status_code == 200:
            vs = [int(v) for v in (r.json() or {}).get('variants') or []]
            if vs:
                return vs[:int(os.environ.get('IDLE_MAX_CLIPS', '6'))]
    except Exception:
        pass
    return [0]


async def _send_idle_frame(ws, envelope: dict, bot_slug: str, avatar_id: str) -> None:
    """Fetch (and cache) the avatar's rest clips and send them so the browser can cycle
    them between turns. First fetch triggers a one-time render on the face-renderer."""
    import base64
    face_url = os.environ.get('FACE_RENDERER_URL', 'http://instar-face-renderer:8088')
    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            variants = await _idle_variants(client, face_url, avatar_id)
            single = len(variants) < 2
            for v in variants:
                key = (avatar_id, v)
                cached = _idle_b64_cache.get(key)
                b64 = cached[0] if cached and (time.time() - cached[1]) < IDLE_CACHE_TTL else None
                if not b64:
                    r = await client.get(f"{face_url}/avatars/{avatar_id}/idle",
                                         params={'variant': v})
                    if r.status_code != 200:
                        # Was a silent return — a missing idle then looked identical to a
                        # working one, and the browser just showed nothing.
                        log.warning("idle_fetch_failed", status=r.status_code,
                                    avatar=avatar_id, variant=v, bot=bot_slug)
                        if not cached:
                            continue
                        b64 = cached[0]      # serve the stale copy rather than nothing
                    else:
                        b64 = base64.b64encode(r.content).decode()
                        _idle_b64_cache[key] = (b64, time.time())
                await ws.send(json.dumps(build_idle_envelope(
                    envelope, bot_slug, b64,
                    variant=None if single else v,
                    count=None if single else len(variants))))
    except Exception as e:
        log.warning("idle_frame_error", error=str(e), bot=bot_slug)


# Last known face config per BOT SLUG, so the rest loop can go out at the start of a turn
# instead of at the end of one.
#
# The config is keyed by voice group, and the voice group only arrives in the gatekeeper's
# RESPONSE - which is after the bot has finished thinking. Sending the idle frame there
# meant the page had nothing to show for the whole turn: she sat frozen on her last frame
# through every pause, which is the single most dead-looking thing a talking head can do.
#
# A bot's avatar does not change between turns in practice, so remembering it is enough.
# The first turn after a restart still pays the old behaviour; every turn after is instant.
_face_cfg_by_slug: dict = {}

# ...and remembered on DISK, because an in-memory cache is empty exactly when it matters.
#
# The cache is populated from a completed turn. So after any restart the first visitor
# gets no rest loop at all - the page sits on whatever frame it had, which is the very
# thing the loop exists to prevent. Restarts are not rare here: every deploy is one.
_FACE_CFG_PATH = os.environ.get('FACE_CFG_CACHE', '/tmp/face_cfg_by_slug.json')


def _load_face_cache() -> None:
    try:
        with open(_FACE_CFG_PATH) as fh:
            _face_cfg_by_slug.update(json.load(fh))
        log.info("face_cache_loaded", bots=len(_face_cfg_by_slug))
    except Exception:
        pass


def _save_face_cache() -> None:
    try:
        with open(_FACE_CFG_PATH, 'w') as fh:
            json.dump(_face_cfg_by_slug, fh)
    except Exception as e:
        log.debug("face_cache_save_failed", error=str(e))


_load_face_cache()


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


# Chunked rendering is OFF by default. It only helps if the browser's <VideoAvatar>
# QUEUES incoming video frames and plays them back to back — if it instead swaps its
# source on each frame, a chunked turn plays only the last clip. That component lives in
# the hosted page, not this repo, so the safe default is the single-clip behaviour that
# has always worked. Flip FACE_CHUNKED=true once the client is known to queue by seq.
FACE_CHUNKED = os.environ.get('FACE_CHUNKED', '').lower() in ('1', 'true', 'yes')


async def _render_and_send_video(ws, envelope: dict, bot_slug: str, audio_b64: str, avatar_id: str) -> None:
    """Decoupled talking-head avatar: render the turn's audio through the avatar and send
    'video' frame(s) carrying the lip-synced clip (with audio baked in).
    Fire-and-forget; any failure is swallowed so it never affects the chat."""
    import base64
    face_url = os.environ.get('FACE_RENDERER_URL', 'http://instar-face-renderer:8088')
    turn_id = f"{envelope.get('session_id', '')}:{int(time.time() * 1000)}"
    wav = base64.b64decode(audio_b64)
    await _send_media_status(ws, envelope, bot_slug, 'video', 'start')
    try:
        async with httpx.AsyncClient(timeout=600.0) as client:
            if not FACE_CHUNKED:
                resp = await client.post(
                    f"{face_url}/render",
                    files={'audio': ('speech.wav', wav, 'audio/mpeg')},
                    data={'avatar_id': avatar_id},
                )
                if resp.status_code != 200:
                    log.debug("video_render_failed", status=resp.status_code, bot=bot_slug)
                    return
                await ws.send(json.dumps(build_video_envelope(
                    envelope, bot_slug, base64.b64encode(resp.content).decode(),
                    turn_id=turn_id, seq=0, final=True)))
                log.info("video_frame_sent", bytes=len(resp.content), bot=bot_slug)
                return

            # Chunked: forward each clip the moment it lands rather than waiting for the
            # whole turn. The renderer streams NDJSON, one object per clip.
            sent = 0
            async with client.stream(
                'POST', f"{face_url}/render/chunked",
                files={'audio': ('speech.wav', wav, 'audio/mpeg')},
                data={'avatar_id': avatar_id},
            ) as resp:
                if resp.status_code != 200:
                    log.debug("video_render_failed", status=resp.status_code, bot=bot_slug)
                    return
                async for line in resp.aiter_lines():
                    if not line.strip():
                        continue
                    try:
                        chunk = json.loads(line)
                    except Exception:
                        continue
                    if chunk.get('error') or not chunk.get('base64'):
                        log.debug("video_chunk_error", error=chunk.get('error'), bot=bot_slug)
                        continue
                    await ws.send(json.dumps(build_video_envelope(
                        envelope, bot_slug, chunk['base64'],
                        turn_id=turn_id, seq=chunk.get('seq', sent),
                        final=bool(chunk.get('final')))))
                    sent += 1
                    log.info("video_frame_sent", bot=bot_slug, seq=chunk.get('seq'),
                             final=bool(chunk.get('final')), frames=chunk.get('frames'),
                             render_secs=chunk.get('elapsed'))
            if not sent:
                log.debug("video_render_empty", bot=bot_slug)
    except Exception as e:
        log.debug("video_render_error", error=str(e), bot=bot_slug)
    finally:
        # Clear on every exit path, including failure — a stuck "rendering her face" reads
        # as a hang, which is worse than never having shown it.
        await _send_media_status(ws, envelope, bot_slug, 'video', 'done')


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


# Which content blocks are TEXT. The stream carries her thinking as content_block_delta
# too, so "forward every delta" would pipe her private reasoning into a visitor's
# browser. Proven, not guessed - a one-line prompt produced:
#
#   block_start idx 0 -> thinking     thinking_delta, signature_delta
#   block_start idx 1 -> text         text_delta x4
#
# So this tracks which block index is text and forwards only deltas on that index whose
# delta.type is text_delta. DEFAULT-DENY: an unrecognised block or delta type is
# dropped, because this stream is a debug channel carrying the inside of a turn and the
# page is shown to a visitor.
def _reply_text_deltas(event, text_blocks: set) -> list[str]:
    """Text she has written, from one stream-json event. Mutates text_blocks."""
    if not isinstance(event, dict) or event.get('type') != 'stream_event':
        return []
    ev = event.get('event') or {}
    kind = ev.get('type')

    if kind == 'content_block_start':
        blk = ev.get('content_block') or {}
        idx = ev.get('index')
        if blk.get('type') == 'text':
            text_blocks.add(idx)
        else:
            text_blocks.discard(idx)      # thinking, tool_use, anything else
        return []

    if kind == 'content_block_delta':
        idx = ev.get('index')
        if idx not in text_blocks:
            return []                      # not a text block - say nothing
        d = ev.get('delta') or {}
        if d.get('type') != 'text_delta':
            return []                      # belt and braces: only text_delta carries speech
        t = d.get('text')
        return [t] if isinstance(t, str) and t else []

    return []


# She marks the spoken summary inline. Mid-stream those brackets are just characters,
# so without this the visitor watches "[SPOKEN]" get typed out. The marker can also be
# SPLIT across deltas, which is why this works on the accumulated text and truncates at
# a dangling opener rather than trying to strip each fragment.
def _strip_spoken(buf: str) -> str:
    out = buf
    for open_tag, close_tag in (('[SPOKEN]', '[/SPOKEN]'), ('<spoken>', '</spoken>')):
        while True:
            i = out.find(open_tag)
            if i < 0:
                break
            j = out.find(close_tag, i)
            if j < 0:
                out = out[:i]              # still being written - hide the rest for now
                break
            out = out[:i] + out[j + len(close_tag):]
    # A partially-arrived opener, e.g. "[SPO", must not flash on screen either.
    for tag in ('[SPOKEN]', '<spoken>'):
        for n in range(len(tag) - 1, 0, -1):
            if out.endswith(tag[:n]):
                return out[:-n]
    return out


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
    # Streaming-reply state. _text_blocks tracks which content-block indices are TEXT,
    # so a thinking block can never be mistaken for speech.
    _text_blocks: set = set()
    _reply_buf: list = []
    _last_shown = ['']
    # Narration state. TTS runs OFF the read loop so a slow synth never delays the next
    # status line, but clips are serialised behind one lock so they arrive — and therefore
    # play — in the order the steps actually happened.
    voice_group = await _get_voice_group(bot_slug) if NARRATE_TOOLS else ''
    narrate_lock = asyncio.Lock()
    narrate_tasks: list[asyncio.Task] = []

    async def _narrate(phrase: str) -> None:
        async with narrate_lock:
            audio = await _generate_tts(phrase, voice_group)
            if not audio:
                return
            try:
                await relay_ws.send(json.dumps(
                    build_status_audio_envelope(envelope, bot_slug, phrase, audio)))
            except Exception:
                pass

    try:
        async with websockets.connect(ws_url, open_timeout=5) as tws:
            async for raw in tws:
                try:
                    msg = json.loads(raw)
                except Exception:
                    continue
                if msg.get('type') != 'output' or msg.get('conversation_id') != conversation_id:
                    continue

                # Her words, as she writes them. This loop was already open and already
                # filtering this stream for tool narration; the text was flowing past it
                # unexamined.
                for piece in _reply_text_deltas(msg.get('event', {}), _text_blocks):
                    _reply_buf.append(piece)
                    shown = _strip_spoken(''.join(_reply_buf))
                    if shown and shown != _last_shown[0]:
                        _last_shown[0] = shown
                        try:
                            await relay_ws.send(json.dumps(
                                build_reply_delta_envelope(envelope, bot_slug, shown)))
                        except Exception:
                            return

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

                    # The phrase is for READING. What gets SPOKEN is chosen from a closed
                    # set of pre-rendered clips by action class, because her `why` is
                    # written for a log and sounds wrong out loud — "Alex asked to search
                    # captain's logs for..." narrated in third person at the person who
                    # asked. Send the class; the page decides if anything plays.
                    #
                    # Deliberately NOT gated on `phrase == last_phrase` like the status
                    # line above: a repeated phrase is still a step that happened, and
                    # suppressing its cue would hide real progress. The pacing that stops
                    # her chattering is a timer in the browser, where it can see whether
                    # she is already talking.
                    try:
                        await relay_ws.send(json.dumps(
                            build_cue_envelope(envelope, bot_slug, class_for_tool(name))))
                    except Exception:
                        return

                    # Live TTS of the phrase. OFF by default and kept only as an escape
                    # hatch: measured on a real turn it fired four times in six seconds,
                    # spoke her log-voice verbatim, and repeated itself. Cues replaced it.
                    if voice_group and len(phrase) >= NARRATE_MIN_CHARS:
                        t = asyncio.create_task(_narrate(phrase))
                        t.add_done_callback(_log_task_exception)
                        narrate_tasks.append(t)
    except asyncio.CancelledError:
        # The caller cancels us once the final response is in hand. Drop any narration still
        # queued — speech for a step the operator has already seen the answer to is just
        # confusing, and it would talk over the final reply's own audio.
        for t in narrate_tasks:
            if not t.done():
                t.cancel()
        raise
    except Exception as e:
        log.debug("status_stream_ended", error=str(e), bot=bot_slug)


# =============================================================================
# MESSAGE HANDLER
# =============================================================================

# ─── Turn taking ──────────────────────────────────────────────────────────────────
#
# The browser streams what it is hearing as it hears it, instead of sending one block
# of text when the button comes up. Something then has to decide when a person has
# finished a thought, or the bot gets handed a turn per syllable.
#
# That decision lives in src/turntaking/ and is pure and tested. This is only the glue:
# it owns the clock, the sockets and the envelopes, and hands everything else over.
#
# See docs/VOICE_TURN_PIPELINE.md in project-instar.

_turn_config = TurnConfigCache(GATEKEEPER_URL)
_turn_detector = TurnDetector(LLM_URL)
_turn_sessions = TurnSessionRegistry()


def build_turn_state_envelope(original: dict, bot_slug: str,
                              state: str, text: str,
                              max_silence_ms: int = 0,
                              asides: list = None, dropping: bool = False) -> dict:
    """Tell the page what is held and what has been committed.

    This is the visible half of the whole feature. Without it the buffering is
    invisible: a person speaks, nothing appears, and they cannot tell whether they
    are being heard or ignored. 'buffering' is provisional and should be shown
    greyed; 'flushed' is the moment it became a turn.
    """
    return {
        'v': 1,
        'type': 'turn_state',
        'session_id': original.get('session_id', ''),
        'timestamp': int(time.time() * 1000),
        'from': {'kind': 'bot', 'slug': bot_slug},
        'to': original.get('from', {}),
        'encrypted': False,
        # The page draws a sweep for the pause that is about to end the turn. Send the
        # threshold rather than letting the browser assume one: a sweep that finishes
        # at a different moment than the flush is worse than no sweep, because it
        # teaches the visitor a timing that is not real.
        # `asides` are the segments judged not to have been addressed to us. They are
        # sent so the page can STRIKE THEM THROUGH - never so it can delete them. A
        # wrong call has to stay visible, or the assistant simply appears to ignore
        # things that were on screen a second ago. `dropping` says whether the verdict
        # actually withheld them from the bot or was only displayed.
        'payload': json.dumps({'state': state, 'text': text,
                               'max_silence_ms': max_silence_ms,
                               'asides': asides or [], 'dropping': bool(dropping)}),
    }


# Characters an action id may contain. It arrives from the BROWSER, is written into a
# turn, and is the thing the bot keys its behaviour on - so it is bounded here rather
# than trusted. A visitor with devtools can emit any id; that is inherent, and the rule
# is the same as a chat link's: it says what happened, it grants nothing.
_ACTION_ID_OK = re.compile(r"^[A-Za-z0-9._:-]{1,64}$")


async def handle_ui_event(envelope: dict, ws, bot_slug: str) -> None:
    """Someone pressed something on the page.

    Becomes an ordinary turn, so history, hints, identity and timeouts all stay on the
    one road into the gatekeeper rather than this inventing a second one.
    """
    try:
        payload = envelope.get('payload')
        if isinstance(payload, str):
            payload = json.loads(payload)
        payload = payload if isinstance(payload, dict) else {}
    except Exception:
        payload = {}

    action_id = str(payload.get('action_id') or '').strip()
    label = str(payload.get('label') or '').strip()[:120]
    value = str(payload.get('value') or '').strip()[:500]

    if not _ACTION_ID_OK.match(action_id):
        log.warning("ui_event_bad_action_id", raw=action_id[:40], bot=bot_slug)
        return

    # What the bot reads. The marker is APPENDED, never prepended: with it first,
    # whitespace collapse slides the visitor's own words inside the brackets and the
    # bot correctly reads its user's text as an injected directive and refuses it.
    said = value or label or action_id
    text = f"{said} [control:{action_id}]"

    turn = dict(envelope)
    turn['type'] = 'chat_request'
    turn['messages'] = [{'role': 'user', 'content': text}]
    turn.pop('payload', None)
    log.info("ui_event", bot=bot_slug, action=action_id, has_value=bool(value))
    await handle_message(turn, ws, bot_slug)


async def handle_stt_frame(envelope: dict, ws, bot_slug: str) -> None:
    """A partial, a final, or a release from the browser's recogniser."""
    session_id = envelope.get('session_id', '')
    if not session_id:
        return
    kind = envelope.get('type', '')
    text = ''
    try:
        payload = envelope.get('payload')
        if isinstance(payload, str):
            payload = json.loads(payload)
        text = ((payload or {}).get('text') or '') if isinstance(payload, dict) else ''
    except Exception:
        text = ''

    async def on_flush(flushed_text: str, reason: str) -> None:
        # A flush IS a turn. Rebuild the envelope the rest of this file already knows
        # how to handle rather than inventing a second path into the gatekeeper —
        # history, hints, identity and timeouts all live on that one road.
        turn = dict(envelope)
        turn['type'] = 'chat_request'
        turn['messages'] = [{'role': 'user', 'content': flushed_text}]
        turn.pop('payload', None)
        await handle_message(turn, ws, bot_slug)

    async def on_state(state: str, state_text: str, verdicts: dict = None) -> None:
        try:
            cfg = await _turn_config.get()
            ms = int(cfg.thresholds.max_silence * 1000)
            # Only the ASIDES travel. The page keeps everything on screen and strikes
            # these through, so sending the segments judged relevant would be the whole
            # transcript a second time for no added meaning.
            asides = [s for s, keep in (verdicts or {}).items() if not keep]
            await ws.send(json.dumps(build_turn_state_envelope(
                envelope, bot_slug, state, state_text, ms,
                asides=asides, dropping=cfg.drops_asides)))
        except Exception as e:
            log.debug("turn_state_send_failed", error=str(e))

    session = _turn_sessions.get(session_id, lambda: TurnSession(
        session_id, on_flush, on_state, detector=_turn_detector))
    session.start(_turn_config.get)

    cfg = await _turn_config.get()
    if kind == 'stt_partial':
        await session.partial(text, cfg)
    elif kind == 'stt_final':
        await session.final(text, cfg)
    elif kind == 'stt_release':
        await session.release(cfg)


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
    # REST LOOP FIRST, before the bot has said anything.
    #
    # Rendering a reply takes seconds. Without this the page holds whatever frame she
    # ended on - or an empty circle on the very first turn - for that entire time. The
    # loop is already built and cached by the renderer, so sending it here costs one
    # frame and buys a head that is alive while she thinks.
    _cached_face = _face_cfg_by_slug.get(bot_slug)
    if _cached_face:
        if _cached_face.get('face_avatar_id'):
            asyncio.create_task(
                _send_idle_frame(ws, envelope, bot_slug, _cached_face['face_avatar_id'])
            )
        else:
            asyncio.create_task(_send_head_kind(ws, envelope, bot_slug, 'rig'))
    session_id = envelope.get('session_id', '')
    messages, source_info = extract_messages_from_envelope(envelope)
    user_context = extract_user_context_from_envelope(envelope)
    assertion = extract_assertion_from_envelope(envelope)
    # Diagnostic: which top-level fields actually arrived, and did the assertion survive
    # the portal -> relay -> here hop. Logs shape only, never the token itself.
    log.info("envelope_fields", keys=sorted(envelope.keys()),
             has_assertion=bool(assertion), assertion_len=len(assertion or ""))

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

    # A fresh page holds no filler clips, whatever this conversation was sent before.
    if any('[visitor_arrived]' in str(m.get('content', '')) for m in messages):
        cues_forget_session(session_id)

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
            if assertion:
                # Session-bound credential, never merged into user_context (see
                # extract_assertion_from_envelope) so it stays out of the bot's prompt.
                gk_body['assertion'] = assertion
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
                if face_on and _face_cfg_by_slug.get(bot_slug) != face_cfg:
                    _face_cfg_by_slug[bot_slug] = face_cfg
                    _save_face_cache()
                elif not face_on:
                    # Remember the ABSENCE too, so the next turn can tell the page to load
                    # the rig at the start rather than after the gatekeeper has answered.
                    if _face_cfg_by_slug.get(bot_slug) is not None and \
                            _face_cfg_by_slug[bot_slug].get('face_avatar_id'):
                        _face_cfg_by_slug[bot_slug] = {}
                        _save_face_cache()
                    elif bot_slug not in _face_cfg_by_slug:
                        _face_cfg_by_slug[bot_slug] = {}
                        _save_face_cache()
                    # No rendered head: this is the one case where the rig IS the avatar.
                    asyncio.create_task(_send_head_kind(ws, envelope, bot_slug, 'rig'))

                # Option A: generate TTS inline and embed it in the response envelope so
                # audio reaches synchronous /ask clients (the platform assistant). Also emit
                # the legacy separate frame below for persistent WS clients (the website).
                #
                # With the avatar ON the audio is NOT embedded (the video carries it), so
                # nothing in the text response depends on TTS — synthesising first just made
                # the reader wait for it. Measured 5.7s of dead time in front of the text on
                # 2026-08-17. Avatar on: send text, then synthesise. Avatar off: the envelope
                # genuinely needs the audio, so the original order stands.
                audio_b64 = ''
                if face_on:
                    await send_response(
                        ws, envelope, bot_slug, gk_data['response'], spoken,
                        attachments=text_attachments or None, audio_base64='',
                    )
                    if spoken and voice_group_id:
                        await _send_media_status(ws, envelope, bot_slug, 'audio', 'start')
                        audio_b64 = await _generate_tts(spoken, voice_group_id)
                        await _send_media_status(ws, envelope, bot_slug, 'audio', 'done')
                else:
                    if spoken and voice_group_id:
                        audio_b64 = await _generate_tts(spoken, voice_group_id)
                    embed_audio = audio_b64
                    await send_response(
                        ws, envelope, bot_slug, gk_data['response'], spoken,
                        attachments=text_attachments or None, audio_base64=embed_audio,
                    )
                    if embed_audio:
                        await _send_audio_frame(ws, envelope, bot_slug, embed_audio)

                # Talking-head avatar (face_enabled). Send the idle loop first (browser
                # loops it while she "thinks"), then the talking clip when it renders — both
                # fire-and-forget so neither blocks the text.
                if face_on:
                    avatar_id = face_cfg['face_avatar_id']
                    asyncio.create_task(_send_idle_frame(ws, envelope, bot_slug, avatar_id))
                    # Filler clips for the NEXT turn. Sending them now is the whole point:
                    # a cue is only useful if the clip is already in the browser.
                    if voice_group_id:
                        asyncio.create_task(
                            _send_cue_clips(ws, envelope, bot_slug, voice_group_id, avatar_id)
                        )
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


def _live_host_ws():
    """The CURRENT host socket, or None. Resolved at call time, never cached."""
    try:
        host = getattr(connection_manager, "_host", None)
        return getattr(host, "ws", None) if host else None
    except Exception:
        return None


async def send_response(ws, original_envelope: dict, bot_slug: str, text: str, spoken: str = '', attachments: list[dict] = None, audio_base64: str = '') -> None:
    """Wrap text in an envelope and send it back through the relay WebSocket.

    Sends on the LIVE socket, not the one the message arrived on.

    A turn takes seconds; the host connection can reconnect in that window. The socket
    captured at receive time is then dead, the reply was written to it, and the failure
    surfaced only as a 'send_on_closed_ws' warning — so the bot had answered, the
    gatekeeper had recorded the answer, and the user saw nothing. Observed exactly that:
    message in at :36, answered at :42, dropped at :45.

    So: try the captured socket, then the current one, and wait out a reconnect in
    progress rather than discarding a completed turn. Losing an answer is far more
    expensive than a few seconds of retry.
    """
    # Context for the relevance judge: a short reply like "yes, do that" is only
    # recognisable as a reply if something remembers what it is replying to.
    try:
        _turn_sessions.note_reply(original_envelope.get('session_id', ''), spoken or text)
    except Exception:                                            # noqa: BLE001
        pass

    response_envelope = build_response_envelope(
        original_envelope, bot_slug, text, spoken,
        attachments=attachments, audio_base64=audio_base64,
    )
    payload = json.dumps(response_envelope)

    deadline = time.time() + 15
    tried_captured = False
    while True:
        sock = ws if not tried_captured else _live_host_ws()
        tried_captured = True
        if sock is not None:
            try:
                await sock.send(payload)
                log.info("response_sent", bot=bot_slug, length=len(text),
                         on_captured_socket=(sock is ws))
                return
            except websockets.ConnectionClosed:
                pass
            except Exception as e:
                log.warning("response_send_error", bot=bot_slug, error=str(e))
        if time.time() >= deadline:
            break
        await asyncio.sleep(1)

    # Never lose an answer in silence.
    log.error("response_dropped", bot=bot_slug, length=len(text),
              note="no live host socket after retry; the turn completed but the user got nothing")


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


def _slugset(items) -> set[str]:
    """Normalize a relay slug list to a set of slug strings.

    The relay sends slugs in two shapes depending on the field: bare strings
    ("hal") and objects carrying a reason ({"slug": "hal", "reason": "..."}).
    Feeding the object form straight into a set raises
    `TypeError: unhashable type: 'dict'`, which — raised inside the message
    loop — killed the whole host socket and put every bot into a permanent
    reconnect cycle with no messages delivered. Normalize, never assume.
    """
    out: set[str] = set()
    for item in items or []:
        if isinstance(item, str):
            out.add(item)
        elif isinstance(item, dict):
            slug = item.get('slug')
            if isinstance(slug, str):
                out.add(slug)
        else:
            log.warning("host_unknown_slug_shape", shape=type(item).__name__)
    return out


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
        # When the relay last told us what it is ACTUALLY serving, as opposed to when
        # it last answered a ping. Kept apart on purpose - see routing_fresh().
        self.last_slugs: float | None = None
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

    async def _handle_frame(self, data: dict, ws) -> None:
        """Dispatch one inbound control/chat frame.

        Lives in its own method so the read loop can wrap the whole dispatch in a
        single guard: any failure here drops one frame instead of the socket.
        """
        mtype = data.get('type', '')
        if mtype == 'chat_request':
            slug = (data.get('to') or {}).get('slug')
            if not slug:
                log.warning("host_msg_no_slug")
                return
            task = asyncio.create_task(handle_message(data, ws, slug))
            task.add_done_callback(_log_task_exception)
        elif mtype == 'ui_event':
            ev_slug = (data.get('to') or {}).get('slug') or slug
            if ev_slug:
                task = asyncio.create_task(handle_ui_event(data, ws, ev_slug))
                task.add_done_callback(_log_task_exception)
        elif mtype in ('stt_partial', 'stt_final', 'stt_release'):
            # Speech as it arrives, on the HOST connection — which is the path every
            # bot actually uses. There are two dispatch sites in this file and only
            # this one is live; wiring the other is a change that deploys cleanly,
            # logs nothing and does nothing.
            stt_slug = (data.get('to') or {}).get('slug') or slug
            if not stt_slug:
                log.warning("host_stt_no_slug")
                return
            task = asyncio.create_task(handle_stt_frame(data, ws, stt_slug))
            task.add_done_callback(_log_task_exception)
        elif mtype == 'attached':
            # Normalize BOTH — `denied` arrives as {slug, reason} objects while `ok` is
            # bare strings, and mixing shapes into a set is what took the socket down.
            ok = _slugset(data.get('ok'))
            denied = _slugset(data.get('denied'))
            # ACT on the reply, don't just log it. A denied slug must leave `confirmed`
            # so /status stops claiming it's reachable, and must stay in `attached` so
            # the keepalive keeps retrying it.
            self.confirmed.update(ok)
            self.confirmed.difference_update(denied)
            if denied:
                # Log the relay's own reason verbatim, not just the slug names. A denial
                # is the difference between "attached" and "silently black-holing every
                # message for this bot", so WHY it was refused is the whole diagnostic.
                log.warning("host_attach_denied", denied=sorted(denied),
                            raw=data.get('denied'),
                            note="relay refused these slugs; keepalive will retry")
            log.info("host_attached", ok=sorted(ok), denied=sorted(denied),
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
            self.last_slugs = time.time()
            await self._reconcile(data.get('attached') or [])
        elif 'error' in data:
            log.warning("host_relay_error", error=data.get('error'))

    async def _reconcile(self, relay_slugs: list) -> None:
        """Relay's view is authoritative. Re-attach anything we want that it isn't serving."""
        relay = _slugset(relay_slugs)
        self.confirmed = self.attached & relay
        missing = self.attached - relay
        if missing:
            log.warning("host_routing_drift", missing=sorted(missing),
                        note="relay is not serving slugs we believe we serve; re-attaching")
            await self._send({'type': 'attach', 'slugs': sorted(missing)})

    def pong_age(self) -> float | None:
        return None if self.last_pong is None else time.time() - self.last_pong

    def routing_age(self) -> float | None:
        """How long since the relay last told us what it is actually serving."""
        return None if self.last_slugs is None else time.time() - self.last_slugs

    def routing_fresh(self) -> bool:
        """Do we have a RECENT answer about routing, or only an old one?

        This is the distinction that cost hours. A pong proves the control channel is
        alive; it proves nothing about whether the relay is still serving our slugs.
        Both used to refresh the same clock, so a relay answering pings while having
        lost a binding looked perfectly healthy and `confirmed` went on reporting the
        last good answer - which is exactly the state found on 2026-10-15: the adapter
        said seven was confirmed and healthy while the relay returned 'Bot not
        connected' to every visitor.
        """
        age = self.routing_age()
        return age is not None and age <= HOST_PING_SECONDS * 3

    def healthy(self) -> bool:
        """Connected, answering, AND recently confirmed to be routing.

        Unknown is not healthy. A status that cannot currently prove a slug is served
        must not claim it is - reporting green on stale evidence is worse than
        reporting unknown, because nothing ever investigates green.
        """
        age = self.pong_age()
        alive = bool(self.connected) and (age is None or age <= HOST_PING_SECONDS * 3)
        return alive and self.routing_fresh()

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
                        # One malformed or unexpected frame must never take the socket
                        # down. An unhandled TypeError in the dispatch previously killed
                        # this loop and put every bot on the account into a permanent
                        # reconnect cycle — connected, attached, delivering nothing.
                        # Contain per-frame failures and keep serving.
                        try:
                            await self._handle_frame(data, ws)
                        except Exception as e:
                            log.error("host_frame_error", mtype=data.get('type', ''),
                                      error=str(e), exc_info=True,
                                      note="frame dropped; socket kept open")
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
                'connected': bool(h.connected and h.routing_fresh() and s in h.confirmed),
                'attach_requested': True,
                # confirmed ONLY counts when the evidence is current. Stale
                # evidence reported as confirmation is what makes an outage invisible.
                'relay_confirmed': bool(h.routing_fresh() and s in h.confirmed),
                'routing_known': h.routing_fresh(),
                'routing_age_seconds': (None if h.routing_age() is None
                                        else int(h.routing_age())),
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
                        elif msg_type in ('stt_partial', 'stt_final', 'stt_release'):
                            # Speech as it arrives. These are NOT turns — the
                            # turn-taking session decides if and when they become one.
                            task = asyncio.create_task(
                                handle_stt_frame(data, ws, conn.slug))
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
