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
        on_state: Optional[Callable[[str, str], Awaitable[None]]] = None,
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
            self._invalidate_answer()
            self._last_reported = ""

        log.info("turn_flush", session=self.session_id[:8], reason=d.reason,
                 chars=len(out), silence_ms=int(silence * 1000))
        if self._on_state:
            await self._on_state("flushed", out)
        await self._on_flush(out, d.reason)

    async def _report_buffering(self) -> None:
        """Tell the page what is held but not yet sent, so a person can watch their
        own words go from provisional to committed rather than wondering."""
        if not self._on_state:
            return
        text = self.buf.text
        if text == self._last_reported:
            return
        self._last_reported = text
        await self._on_state("buffering", text)

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
