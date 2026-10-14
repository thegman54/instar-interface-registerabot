"""
One speaker, one buffer, one decision loop.

The rules are pure and the buffer is pure; this is the part that has a clock and side
effects, kept deliberately thin so the thinking stays in the testable half.

What it does:

  * accumulates recogniser output for one conversation
  * re-decides on every update AND on a timer, because "they stopped talking" is an
    event that nothing sends - silence has to be noticed, not received
  * flushes a turn through a callback, so this module knows nothing about relays,
    envelopes or the gatekeeper
  * reports what it is holding, so the page can show words going from provisional to
    committed instead of the visitor guessing

Everything external arrives as a callback or a config object. That is what lets the
whole loop be tested with a fake clock and no network.
"""

import asyncio
import time
from typing import Awaitable, Callable, Optional

import structlog

from .aside import AsideJudge, surviving
from .buffer import TranscriptBuffer
from .config import TurnConfig
from .rules import decide

log = structlog.get_logger()

TICK_S = 0.15
# A session with no speech for this long is over; its buffer is dropped rather than
# left to be flushed into some later, unrelated turn.
IDLE_S = 300.0


class TurnSession:
    """Turn-taking for one conversation."""

    def __init__(
        self,
        session_id: str,
        on_flush: Callable[[str, str], Awaitable[None]],
        on_state: Optional[Callable[..., Awaitable[None]]] = None,
        detector=None,
        clock: Callable[[], float] = time.monotonic,
    ):
        self.session_id = session_id
        self._on_flush = on_flush          # (text, reason) -> send it as a turn
        self._on_state = on_state          # (state, text) -> tell the page
        self._detector = detector
        self._clock = clock
        self.buf = TranscriptBuffer(_clock=clock)
        self.buf.last_voice_at = clock()
        self._lock = asyncio.Lock()
        self._task: Optional[asyncio.Task] = None
        self._model_answer: Optional[bool] = None
        self._asked_for: str = ""          # the text the model's answer applies to
        self._judge = AsideJudge(detector)
        self._judging: Optional[asyncio.Task] = None
        # The last thing the assistant said. Context for "was that meant for us" -
        # without it, "yes do that" is indistinguishable from a stray fragment.
        self.last_reply: str = ""
        self._last_reported: str = ""
        self.last_activity = clock()

    # --- input ------------------------------------------------------------------

    async def partial(self, text: str, cfg: TurnConfig) -> None:
        async with self._lock:
            self.buf.interim(text)
            self._invalidate_answer()
            self.last_activity = self._clock()
        await self._report_buffering()
        await self._evaluate(cfg)

    async def final(self, text: str, cfg: TurnConfig) -> None:
        async with self._lock:
            self.buf.final(text)
            self._invalidate_answer()
            self.last_activity = self._clock()
        await self._report_buffering()
        await self._evaluate(cfg)

    async def release(self, cfg: TurnConfig) -> None:
        """The push-to-talk button came up. They have declared they are finished."""
        async with self._lock:
            self.last_activity = self._clock()
        await self._evaluate(cfg, released=True)

    def _invalidate_answer(self) -> None:
        """New words make a previous verdict stale.

        Without this the model's opinion of 'what time is' would still be applied
        after it became 'what time is the flight', which is how a detector ends up
        confidently answering a question nobody asked.
        """
        self._model_answer = None
        self._asked_for = ""

    # --- the decision ------------------------------------------------------------

    async def _evaluate(self, cfg: TurnConfig, released: bool = False) -> None:
        async with self._lock:
            text = self.buf.text
            silence = self.buf.silence_s()
            answer = self._model_answer if self._asked_for == text else None

        d = decide(text, silence_s=silence, released=released,
                   thresholds=cfg.thresholds, model_says_complete=answer)

        # Fire-and-forget, so the sentences are already judged by the time the pause
        # ends rather than the flush waiting on a second model.
        await self._judge_asides(cfg, released=released)

        if d.ask_model and cfg.uses_model and self._detector is not None:
            verdict = await self._detector.is_complete(text, cfg.model)
            async with self._lock:
                # Only keep it if nothing was said while we were asking.
                if self.buf.text == text:
                    self._model_answer = verdict
                    self._asked_for = text
                silence = self.buf.silence_s()
            if verdict is None:
                return        # do not know -> keep listening; max_silence will catch it
            d = decide(text, silence_s=silence, released=released,
                       thresholds=cfg.thresholds, model_says_complete=verdict)

        if not d.send:
            return

        async with self._lock:
            out = self.buf.take_all() if released else self.buf.take()
            # take() hands over only COMMITTED text, so a flush can legitimately be
            # empty when everything heard is still provisional. Sending an empty turn
            # would make the bot answer nothing at all.
            if not out.strip():
                return
            verdicts = self._judge.verdicts
            self._invalidate_answer()
            self._judge.reset()
            self._last_reported = ""

        # WHAT IS SHOWN IS ALWAYS THE WHOLE TURN; what is SENT may be less. The page
        # strikes the asides through, so a wrong call is visible and arguable rather
        # than a sentence that silently never happened.
        send_text = surviving(out, verdicts) if cfg.drops_asides else out
        dropped = len(out) - len(send_text)
        log.info("turn_flush", session=self.session_id[:8], reason=d.reason,
                 chars=len(send_text), silence_ms=int(silence * 1000),
                 aside_chars=dropped or None)
        if self._on_state:
            await self._on_state("flushed", out, verdicts)
        if not send_text.strip():
            return
        await self._on_flush(send_text, d.reason)

    async def _report_buffering(self) -> None:
        """Tell the page what is held but not yet sent, so a person can watch their
        own words go from provisional to committed rather than wondering."""
        if not self._on_state:
            return
        text = self.buf.text
        v = self._judge.verdicts
        # Re-report when a VERDICT lands even though the text has not changed - the
        # judgement arriving is itself the thing worth showing.
        stamp = text + "\x00" + repr(sorted(v.items()))
        if stamp == self._last_reported:
            return
        self._last_reported = stamp
        await self._on_state("buffering", text, v)

    # --- is any of this for us? --------------------------------------------------

    async def _judge_asides(self, cfg: TurnConfig, released: bool = False) -> None:
        """Judge completed sentences in the background, never in the decision path.

        OFF the critical path on purpose. This is a second model call on a pipeline
        whose entire budget is the length of a human pause, so it must never be
        something the flush waits on - a verdict that has not arrived simply means
        the segment is unjudged, which means kept.
        """
        if not cfg.judges_asides:
            return
        if self._judging and not self._judging.done():
            return
        pending = self._judge.unjudged(self.buf.text, include_last=released)
        if not pending:
            return

        async def run():
            for seg in pending:
                await self._judge.judge(seg, self.last_reply, cfg.model)
            await self._report_buffering()

        self._judging = asyncio.create_task(run())

    # --- the clock ---------------------------------------------------------------

    async def _tick(self, get_cfg: Callable[[], Awaitable[TurnConfig]]) -> None:
        """Silence is not an event anyone sends. It has to be looked for."""
        try:
            while True:
                await asyncio.sleep(TICK_S)
                if self.buf.is_empty():
                    continue
                try:
                    await self._evaluate(await get_cfg())
                except Exception as e:
                    log.warning("turn_tick_failed", session=self.session_id[:8],
                                error=str(e))
        except asyncio.CancelledError:
            pass

    def start(self, get_cfg) -> None:
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self._tick(get_cfg))

    async def close(self) -> None:
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except (asyncio.CancelledError, Exception):
                pass
            self._task = None

    def is_idle(self) -> bool:
        return (self._clock() - self.last_activity) > IDLE_S


class SessionRegistry:
    """One TurnSession per conversation, reaped when they go quiet."""

    def __init__(self):
        self._sessions: dict[str, TurnSession] = {}

    def note_reply(self, session_id: str, text: str) -> None:
        """What the assistant just said, kept for the relevance judge.

        Without it a short answer is unjudgeable: "yes, do that" is a reply to the
        last thing she said, and with no record of what that was it looks exactly like
        a fragment of someone else's conversation.
        """
        s = self._sessions.get(session_id)
        if s is not None and text:
            s.last_reply = str(text)[-400:]

    def get(self, session_id: str, factory) -> TurnSession:
        s = self._sessions.get(session_id)
        if s is None:
            s = factory()
            self._sessions[session_id] = s
        return s

    async def sweep(self) -> int:
        dead = [k for k, s in self._sessions.items() if s.is_idle()]
        for k in dead:
            await self._sessions.pop(k).close()
        return len(dead)

    async def close_all(self) -> None:
        for s in list(self._sessions.values()):
            await s.close()
        self._sessions.clear()

    def __len__(self) -> int:
        return len(self._sessions)
