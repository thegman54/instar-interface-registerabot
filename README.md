# Instar Interface: RegisterABot

A [RegisterABot](https://registerabot.com) interface for [Project Instar](https://github.com/thegman54/project-instar) that connects your bot to the RegisterABot relay via persistent WebSocket. This is how your Instar bot becomes shareable — clients, teammates, and visitors connect through RegisterABot's approval-based access system, and this interface is the bridge.

## How It Works

```
Visitor/Service on RegisterABot
  └─► RegisterABot Relay (WebSocket)
        └─► This interface (registerabot-api container)
              ├─► Gatekeeper /process  (sanitize passphrases, get tools)
              ├─► Bot /chat            (LLM processes message with tools)
              └─► Gatekeeper /record   (store conversation history)
                    └─► Relay WebSocket (response envelope sent back)
```

The interface maintains a persistent WebSocket connection to the RegisterABot relay as a bot. When visitors or services send messages (including chat link hints/passphrases), the interface:

1. Strips the relay envelope, extracts the message payload
2. Flattens all messages (including system hints) into text for the Gatekeeper
3. Routes through Gatekeeper for passphrase extraction and tool resolution
4. Sends the cleaned message to the bot service
5. Wraps the response in a relay envelope and sends it back through the WebSocket

Auto-reconnects on disconnect. Tracks persistent conversations across turns.

## Prerequisites

- A running Project Instar deployment (Gatekeeper + Bot + MCP Server)
- Docker and Docker Compose
- A RegisterABot account with a registered bot

## RegisterABot Setup

### 1. Create an Account

1. Go to [registerabot.com](https://registerabot.com)
2. Sign up with GitHub OAuth
3. On first login, choose **"I have a bot"** during onboarding

### 2. Register Your Bot

1. In the dashboard, go to **Bots**
2. Click **Add Bot**
3. Fill in your bot's name and description
4. After creation, you'll see your bot card with:
   - **Bot Slug** → this is your `REGISTERABOT_BOT_SLUG`
   - **API Key** (generate one) → this is your `REGISTERABOT_API_KEY`

### 3. Get the Relay URL

The relay URL is the WebSocket endpoint your bot connects to:
```
wss://relay.registerabot.com
```
This is your `REGISTERABOT_RELAY_URL`.

### 4. Create Chat Links (Optional)

Chat links let you share your bot with specific access levels:

1. On your bot card, go to **Chat Links**
2. Create a link with optional **hints** (these become Gatekeeper passphrases)
3. Share the link — visitors who use it will have those hints injected into every message
4. Different links can carry different hints, giving different tool access to different people

## Required Secrets

Store these in Infisical (or provide as environment variables):

| Secret Name | Description | Where to Find |
|---|---|---|
| `REGISTERABOT_RELAY_URL` | WebSocket relay URL (`wss://relay.registerabot.com`) | Fixed — this is the relay endpoint |
| `REGISTERABOT_BOT_SLUG` | Your bot's unique identifier | RegisterABot Dashboard > Bot card |
| `REGISTERABOT_API_KEY` | API key for WebSocket authentication | RegisterABot Dashboard > Bot card > Keys |

## Installation

### Via Admin UI (Recommended)

1. Download this repo as a zip file
2. In the Instar Admin UI, go to **Interfaces**
3. Click **Upload Interface** and select the zip
4. The interface will be auto-discovered via `manifest.yaml`
5. Configure the required secrets in Infisical
6. Start the interface from the Admin UI

### Manual

```bash
git clone https://github.com/thegman54/instar-interface-registerabot.git tools/registerabot
```

The Instar stack manager will detect the `manifest.yaml` and make the interface available.

## Environment Variables

| Variable | Required | Default | Description |
|---|---|---|---|
| `REGISTERABOT_RELAY_URL` | Yes | — | WebSocket relay URL |
| `REGISTERABOT_BOT_SLUG` | Yes | — | Bot identifier on the relay |
| `REGISTERABOT_API_KEY` | Yes | — | API key for authentication |
| `GATEKEEPER_URL` | No | `http://gatekeeper:8080` | Instar Gatekeeper API URL |
| `BOT_SERVICE_URL` | No | `http://bot:8080` | Instar Bot Service API URL |
| `PORT` | No | `8091` | API server port |

### Infisical Integration (Optional)

If running within a full Instar deployment, secrets can be fetched from Infisical at startup instead of being passed as environment variables:

| Variable | Description |
|---|---|
| `INFISICAL_URL` | Infisical API URL (e.g., `http://infisical:8080`) |
| `INFISICAL_TOKEN` | Service token for Infisical |
| `INFISICAL_PROJECT_ID` | Project ID in Infisical |
| `INFISICAL_ENVIRONMENT` | Environment name (default: `prod`) |

The `entrypoint.sh` script fetches `REGISTERABOT_RELAY_URL`, `REGISTERABOT_BOT_SLUG`, and `REGISTERABOT_API_KEY` from Infisical and exports them before starting the app.

## Endpoints

| Endpoint | Description |
|---|---|
| `GET /health` | Health check |
| `GET /status` | Connection status — slug, relay URL, connected state, uptime |

## How Chat Link Hints Work With Gatekeeper

This is the key integration between RegisterABot and Instar:

1. Bot owner creates a chat link on RegisterABot with **hints** (e.g., passphrases)
2. Visitor clicks the link and sends a message
3. RegisterABot injects hints as system messages in the envelope: `[HINTS: passphrase1 passphrase2]`
4. This interface flattens all messages (including hints) into text
5. Gatekeeper's sanitizer extracts the passphrases, looks up matching grants, and unlocks tools
6. The bot receives the cleaned message (passphrases removed) plus the list of available tools
7. Different chat links → different passphrases → different tools → same bot, different capabilities

## Envelope Format

Inbound envelopes from the relay:
```json
{
  "v": 1,
  "type": "chat_request",
  "session_id": "unique-session-id",
  "from": { "kind": "service", "slug": "visitor-slug" },
  "to": { "kind": "bot", "slug": "your-bot-slug" },
  "payload": "{\"messages\": [{\"role\": \"user\", \"content\": \"hello\"}]}"
}
```

Outbound response envelopes:
```json
{
  "v": 1,
  "type": "final_response",
  "session_id": "unique-session-id",
  "from": { "kind": "bot", "slug": "your-bot-slug" },
  "to": { "kind": "service", "slug": "visitor-slug" },
  "payload": "{\"reply\": \"Hello! How can I help?\"}"
}
```

## License

MIT
