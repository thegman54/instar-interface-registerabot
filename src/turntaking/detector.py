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


# "Was that said to us, or did the microphone just overhear it?"
#
# A different question from completeness and a much more dangerous one to get wrong,
# so the prompt is written to make KEEPING the easy answer: it is told to choose
# RELEVANT when unsure, and the caller treats anything it cannot parse as RELEVANT too.
# Measured on gemma3:12b over twelve hand-written lines: 11/12 correct, and - the
# number that actually matters - ZERO false asides, the direction that costs someone
# their sentence. ~250ms per call once the model is warm.
#
# An earlier wording ended "when in doubt answer RELEVANT" and scored 5/12: it was so
# timid it let plain third-party chatter through, which is a feature that does nothing.
# Defining the two classes by what they ASK OF THE ASSISTANT, and putting the tie-break
# in one short line at the end, is what moved it. The safety does not live in the
# prompt's timidity - it lives in the caller: unparsed and unreachable mean keep, and a
# turn judged entirely aside is kept whole.
RELEVANCE_PROMPT = (
    "You are filtering a live microphone feed for an assistant.\n"
    "The microphone picks up the whole room, so some of what it hears is the person "
    "talking TO the assistant and some is not.\n\n"
    "Answer with exactly one word.\n"
    "ASIDE - the line is addressed to another person, or is talk about the world that "
    "asks the assistant for nothing: speaking to someone else in the room, one side of "
    "a phone call, a television, chatter or storytelling.\n"
    "RELEVANT - the line asks the assistant something, instructs it, or answers it.\n"
    "If the line could plausibly be either, answer RELEVANT.\n\n"
    "{context}"
    "Line: {text}\nAnswer:"
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
        # INCOMPLETE first: "COMPLETE" is a prefix-match away from being found inside
        # it, and a reversed verdict here sends fragments.
        if out.startswith("INCOMPLETE"):
            return False
        if out.startswith("COMPLETE"):
            return True
        log.debug("turn_detect_unparsed", got=out[:40])
        return None

    async def is_relevant(self, text: str, context: str, model: str) -> bool | None:
        """Was this line addressed to us? True / False, or None for 'do not know'.

        None is the ONLY safe default and the caller must treat it as 'keep'. Dropping
        something a person actually said leaves them watching an assistant ignore
        words that were on screen a second ago, with no way to tell why.
        """
        if not model or not text.strip():
            return None
        ctx = ""
        if context:
            # The last thing the assistant said, so "yes, do that" is recognisably a
            # reply rather than a stray fragment. Bounded - this is a latency budget
            # measured in tens of milliseconds, not a conversation.
            ctx = "The assistant last said: " + context.strip()[-240:] + "\n"
        t0 = time.monotonic()
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as c:
                r = await c.post(f"{self._url}/api/generate", json={
                    "model": model,
                    "prompt": RELEVANCE_PROMPT.format(context=ctx,
                                                      text=text.strip()[-TAIL_CHARS:]),
                    "stream": False,
                    "options": {"num_predict": 3, "temperature": 0.0},
                })
                r.raise_for_status()
                out = ((r.json() or {}).get("response") or "").strip().upper()
        except Exception as e:                                   # noqa: BLE001
            log.debug("relevance_detect_failed", error=str(e),
                      ms=int((time.monotonic() - t0) * 1000))
            return None
        if out.startswith("ASIDE"):
            return False
        if out.startswith("RELEVANT"):
            return True
        log.debug("relevance_unparsed", got=out[:40])
        return None
