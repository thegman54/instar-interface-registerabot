"""
RegisterABot Interface Tool

FastAPI server providing:
- WebSocket connection to RegisterABot relay (as a bot)
- Message routing through gatekeeper and bot service
- Envelope stripping on inbound, envelope wrapping on outbound

Follows the same pattern as Slack and Zoom interface tools:
  message in → gatekeeper /process → bot /chat → gatekeeper /record-response → message out
"""

import os
import json
import asyncio
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI
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
BOT_SLUG = os.environ.get('REGISTERABOT_BOT_SLUG', '')
API_KEY = os.environ.get('REGISTERABOT_API_KEY', '')

RECONNECT_DELAY = 5  # seconds

# Conversations marked persistent by gatekeeper (conversation_id -> True)
# Once a persistence-granting passphrase is matched, the conversation stays persistent
_persistent_conversations: dict[str, bool] = {}


def _log_task_exception(task: asyncio.Task) -> None:
    """Log exceptions from fire-and-forget tasks."""
    if not task.cancelled() and task.exception():
        log.error("task_exception", error=str(task.exception()))
ws_connection: websockets.WebSocketClientProtocol | None = None
ws_task: asyncio.Task | None = None
connected: bool = False
connected_since: float | None = None


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
    session_id = envelope.get('session_id', '')
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


def build_response_envelope(
    original_envelope: dict,
    bot_slug: str,
    response_text: str,
) -> dict:
    """
    Build an outbound envelope to send back through the relay.

    Mirrors the original envelope's routing: from=bot, to=original sender.
    """
    return {
        'v': 1,
        'type': 'final_response',
        'session_id': original_envelope.get('session_id', ''),
        'timestamp': int(time.time() * 1000),
        'from': {'kind': 'bot', 'slug': bot_slug},
        'to': original_envelope.get('from', {}),
        'encrypted': False,
        'payload': json.dumps({'reply': response_text}),
    }


# =============================================================================
# MESSAGE HANDLER
# =============================================================================

async def handle_message(envelope: dict, ws) -> None:
    """
    Handle incoming RegisterABot message — route through gatekeeper and bot.

    Flow:
    1. Strip envelope, extract messages
    2. Flatten messages into text for gatekeeper (hints included as-is)
    3. Call gatekeeper /process to sanitize passphrases and get tools
    4. Call bot service /chat with cleaned message
    5. Wrap response in envelope, send back through relay
    6. Record response with gatekeeper
    """
    session_id = envelope.get('session_id', '')
    messages, source_info = extract_messages_from_envelope(envelope)

    if not messages:
        log.warning("empty_message", session=session_id, source=source_info)
        return

    log.info(
        "message_received",
        session=session_id,
        source=source_info,
        message_count=len(messages),
    )

    # Flatten messages into a single string for the gatekeeper.
    # System messages (including [HINTS: ...] from chat link passphrases)
    # are included — gatekeeper's sanitizer will strip the passphrases.
    full_message = '\n'.join(m.get('content', '') for m in messages)

    # Use session_id as conversation_id (groups multi-turn conversations)
    conversation_id = f"registerabot:{session_id}"
    user_id = source_info

    try:
        async with httpx.AsyncClient() as client:
            # 1. Process through gatekeeper
            gk_response = await client.post(
                f"{GATEKEEPER_URL}/process",
                json={
                    'conversation_id': conversation_id,
                    'source': 'registerabot',
                    'user_id': user_id,
                    'message': full_message,
                },
                timeout=30.0,
            )

            if gk_response.status_code != 200:
                log.error("gatekeeper_error", status=gk_response.status_code, body=gk_response.text)
                await send_response(ws, envelope, "I had trouble processing that. Please try again.")
                return

            gk_data = gk_response.json()
            log.debug("gatekeeper_response", data=gk_data)

            # Track persistence — once granted, stays granted for this conversation
            if gk_data.get('persistent'):
                _persistent_conversations[conversation_id] = True
                log.info("persistence_granted", conversation=conversation_id)
            is_persistent = _persistent_conversations.get(conversation_id, False)

            # If message was only passphrases (no actual content)
            if gk_data.get('is_empty'):
                if gk_data.get('newly_unlocked'):
                    tools = ', '.join(gk_data['newly_unlocked'])
                    await send_response(ws, envelope, f"Tools now available: {tools}. What would you like to do?")
                return

            # 2. Call bot service with cleaned message
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
                timeout=120.0,
            )

            if bot_response.status_code != 200:
                log.error("bot_error", status=bot_response.status_code, body=bot_response.text)
                await send_response(ws, envelope, "I encountered an error processing that.")
                return

            bot_data = bot_response.json()
            response_text = bot_data.get('response', '')

            # 3. Send response back through relay
            if response_text:
                await send_response(ws, envelope, response_text)
                log.debug("response_sent", session=session_id, length=len(response_text))

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
        log.error("connection_error", error=str(e))
        try:
            await send_response(ws, envelope, "I'm having trouble connecting to my internal services. Please try again.")
        except Exception:
            log.warning("failed_to_send_error_response")
    except Exception as e:
        log.error("message_handler_error", error=str(e))
        import traceback
        traceback.print_exc()
        try:
            await send_response(ws, envelope, "I encountered an error. Please try again.")
        except Exception:
            log.warning("failed_to_send_error_response")


async def send_response(ws, original_envelope: dict, text: str) -> None:
    """Wrap text in an envelope and send it back through the relay WebSocket."""
    response_envelope = build_response_envelope(original_envelope, BOT_SLUG, text)
    await ws.send(json.dumps(response_envelope))


# =============================================================================
# WEBSOCKET CONNECTION
# =============================================================================

async def relay_loop() -> None:
    """
    Persistent WebSocket connection to RegisterABot relay.

    Connects as the bot, listens for messages, handles them,
    and auto-reconnects on disconnect.
    """
    global ws_connection, connected, connected_since

    while True:
        try:
            ws_url = f"{RELAY_URL}/ws/bot/{BOT_SLUG}?key={API_KEY}"
            log.info("relay_connecting", url=RELAY_URL, slug=BOT_SLUG)

            async with websockets.connect(ws_url) as ws:
                ws_connection = ws
                connected = True
                connected_since = time.time()
                log.info("relay_connected", slug=BOT_SLUG)

                async for raw in ws:
                    try:
                        data = json.loads(raw)
                    except json.JSONDecodeError:
                        log.warning("invalid_json", raw=raw[:200])
                        continue

                    # Only handle chat_request envelopes
                    msg_type = data.get('type', '')
                    if msg_type == 'chat_request':
                        task = asyncio.create_task(handle_message(data, ws))
                        task.add_done_callback(_log_task_exception)
                    elif 'error' in data:
                        log.warning("relay_error", error=data.get('error'))
                    else:
                        log.debug("unhandled_message_type", type=msg_type)

        except websockets.ConnectionClosed as e:
            log.warning("relay_disconnected", code=e.code, reason=e.reason)
        except (OSError, websockets.InvalidURI, websockets.InvalidHandshake) as e:
            log.error("relay_connection_failed", error=str(e))
        except Exception as e:
            log.error("relay_unexpected_error", error=str(e))
            import traceback
            traceback.print_exc()
        finally:
            ws_connection = None
            connected = False
            connected_since = None

        log.info("relay_reconnecting", delay=RECONNECT_DELAY)
        await asyncio.sleep(RECONNECT_DELAY)


# =============================================================================
# LIFESPAN
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown logic."""
    global ws_task

    log.info("registerabot_api_starting")

    if not RELAY_URL or not BOT_SLUG or not API_KEY:
        log.error(
            "registerabot_not_configured",
            message="REGISTERABOT_RELAY_URL, REGISTERABOT_BOT_SLUG, and REGISTERABOT_API_KEY required",
        )
        yield
        return

    # Start relay WebSocket loop in background
    ws_task = asyncio.create_task(relay_loop())
    log.info("relay_loop_started", slug=BOT_SLUG)

    yield

    # Shutdown
    if ws_task:
        ws_task.cancel()
        try:
            await ws_task
        except asyncio.CancelledError:
            pass
    if ws_connection:
        await ws_connection.close()
    log.info("registerabot_api_stopped")


# =============================================================================
# APP
# =============================================================================

app = FastAPI(title="RegisterABot Interface Tool", lifespan=lifespan)


# =============================================================================
# HEALTH & STATUS
# =============================================================================

@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "ok"}


@app.get("/status")
async def status():
    """Get RegisterABot relay connection status."""
    uptime = None
    if connected_since:
        uptime = int(time.time() - connected_since)

    return {
        "connected": connected,
        "bot_slug": BOT_SLUG,
        "relay_url": RELAY_URL,
        "connected_since": connected_since,
        "uptime_seconds": uptime,
    }


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 8091))
    uvicorn.run(app, host="0.0.0.0", port=port)
