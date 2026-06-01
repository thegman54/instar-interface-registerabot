#!/bin/sh
# Fetch secrets from Infisical at startup, export as env vars, then exec the app.

if [ -n "$INFISICAL_URL" ] && [ -n "$INFISICAL_TOKEN" ] && [ -n "$INFISICAL_PROJECT_ID" ]; then
    echo "Fetching secrets from Infisical..."
    python3 /app/fetch_secrets.py REGISTERABOT_RELAY_URL REGISTERABOT_BOT_SLUG REGISTERABOT_API_KEY > /tmp/.secrets 2>/tmp/.secrets_err

    if [ -s /tmp/.secrets ]; then
        while IFS= read -r line; do
            export "$line"
            echo "  Loaded ${line%%=*}"
        done < /tmp/.secrets
        rm -f /tmp/.secrets
    else
        echo "WARNING: No secrets returned from Infisical"
        cat /tmp/.secrets_err 2>/dev/null
    fi
    rm -f /tmp/.secrets_err
else
    echo "No Infisical config — using environment variables as-is"
fi

exec python -m src.api
