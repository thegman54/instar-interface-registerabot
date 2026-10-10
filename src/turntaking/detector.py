"""
"Has this person finished a thought?" — asked of a small local model.

This sits between someone speaking and the bot replying, so its whole budget is tens
of milliseconds. Three consequences shape everything here:

  * It is asked a YES/NO question and nothing else. No conversation, no history, no
    reasoning - just the tail of what was said.
  * It is given a hard timeout, and a timeout means "do not know", not "wait".
  * "Do not know" always resolves to the safe side, which is to KEEP LISTENING -
    never to send a fragment. Past max_silence the rules flush anyway, so an
    unreachable detector costs a slightly slower turn and nothing else.

Never on the critical path for correctness. Only for speed.
"""

import asyncio
import time

import httpx
import structlog

log = structlog.get_logger()

TIMEOUT_S = 1.5
TAIL_CHARS = 200

PROMPT = (
    "You judge whether a speaker has finished a thought.\n"
    "Reply with exactly one word: COMPLETE or INCOMPLETE.\n\n"
    "COMPLETE means the sentence could stand on its own and be acted on.\n"
    "INCOMPLETE means they are clearly mid-sentence and more is coming.\n\n"
    "Transcript: {text}\n\nAnswer:"
)


class TurnDetector:
    """Asks an Ollama model whether an utterance is finished."""

    def __init__(self, llm_url: str, timeout_s: float = TIMEOUT_S):
        self._url = llm_url.rstrip("/")
        self._timeout = timeout_s
        self._fail_streak = 0

    async def is_complete(self, text: str, model: str) -> bool | None:
        """True / False, or None for 'do not know' — which the caller must treat as
        'keep listening', never as 'send'."""
        if not model or not text.strip():
            return None
        tail = text.strip()[-TAIL_CHARS:]
        t0 = time.monotonic()
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as c:
                r = await c.post(f"{self._url}/api/generate", json={
                    "model": model,
                    "prompt": PROMPT.format(text=tail),
                    "stream": False,
                    # Two tokens is enough for one word, and caps the damage when a
                    # model decides to explain itself instead of answering.
                    "options": {"num_predict": 3, "temperature": 0.0},
                })
                r.raise_for_status()
                out = ((r.json() or {}).get("response") or "").strip().upper()
        except (httpx.TimeoutException, asyncio.TimeoutError):
            self._fail_streak += 1
            log.debug("turn_detect_timeout", ms=int((time.monotonic() - t0) * 1000),
                      streak=self._fail_streak)
            return None
        except Exception as e:
            self._fail_streak += 1
            if self._fail_streak in (1, 10, 100):      # once, then rarely
                log.warning("turn_detect_failed", error=str(e), streak=self._fail_streak)
            return None

        self._fail_streak = 0
        ms = int((time.monotonic() - t0) * 1000)
        if ms > 400:
            # Worth saying out loud: at this latency the detector is costing more than
            # the pause it is adjudicating, and a smaller model would be strictly better.
            log.info("turn_detect_slow", ms=ms, model=model,
                     note="a 0.5-1B model answers this in tens of ms")
        if out.startswith("COMPLETE"):
            return True
        if out.startswith("INCOMPLETE"):
            return False
        log.debug("turn_detect_unparsed", got=out[:40])
        return None
