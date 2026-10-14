"""
Where the thresholds come from.

The gatekeeper owns this config because that is where admin-editable state lives and
where the screen is. The adapter only reads it — and it has to keep working when the
read fails, because a turn-taking pipeline that stops because a settings endpoint
hiccuped is worse than one running on slightly stale numbers.

So: cached, refreshed on a timer, and falling back to defaults that are themselves a
working configuration.
"""

import time
from dataclasses import dataclass
from typing import Optional

import httpx
import structlog

from .rules import Thresholds

log = structlog.get_logger()

REFRESH_S = 60.0


@dataclass
class TurnConfig:
    enabled: bool = False
    model: str = ""
    thresholds: Thresholds = None
    # Judge whether what was heard was addressed to us at all. OFF by default: it can
    # only ever remove words a person said, so it is opted into deliberately.
    relevance: bool = False
    # What a verdict DOES. 'mark' shows the judgement on the page and still sends
    # everything; 'drop' also withholds the marked segments. 'mark' is the default so
    # the behaviour can be watched for a while before it is allowed to cost anything.
    relevance_action: str = "mark"

    def __post_init__(self):
        if self.thresholds is None:
            self.thresholds = Thresholds()

    @property
    def drops_asides(self) -> bool:
        return bool(self.relevance and self.model and self.relevance_action == "drop")

    @property
    def judges_asides(self) -> bool:
        return bool(self.relevance and self.model)

    @property
    def uses_model(self) -> bool:
        """Enabled AND actually pointed at something. Enabled-with-no-model is a
        half-configured state the UI warns about; here it simply means timing only."""
        return bool(self.enabled and self.model)


class ConfigCache:
    """Reads turn-detection config from the gatekeeper, and keeps working without it."""

    def __init__(self, gatekeeper_url: str, refresh_s: float = REFRESH_S):
        self._url = gatekeeper_url.rstrip("/")
        self._refresh_s = refresh_s
        self._cfg = TurnConfig()
        self._fetched_at = 0.0
        self._warned = False

    async def get(self) -> TurnConfig:
        if time.monotonic() - self._fetched_at < self._refresh_s:
            return self._cfg
        try:
            async with httpx.AsyncClient(timeout=5.0) as c:
                r = await c.get(f"{self._url}/api/turn-detection")
                r.raise_for_status()
                d = (r.json() or {}).get("config") or {}
            self._cfg = TurnConfig(
                enabled=bool(d.get("enabled")),
                model=str(d.get("model") or ""),
                thresholds=Thresholds(
                    min_silence=float(d.get("min_silence_ms", 350)) / 1000.0,
                    ask_silence=float(d.get("ask_silence_ms", 500)) / 1000.0,
                    punct_silence=float(d.get("punct_silence_ms", 500)) / 1000.0,
                    max_silence=float(d.get("max_silence_ms", 1200)) / 1000.0,
                ),
                relevance=bool(d.get("relevance_enabled")),
                relevance_action=str(d.get("relevance_action") or "mark"),
            )
            self._warned = False
        except Exception as e:
            # Log once per outage, not once per refresh. A settings endpoint that is
            # down for an hour should not produce sixty identical warnings.
            if not self._warned:
                log.warning("turn_config_unavailable", error=str(e),
                            note="using last known config")
                self._warned = True
        self._fetched_at = time.monotonic()
        return self._cfg
