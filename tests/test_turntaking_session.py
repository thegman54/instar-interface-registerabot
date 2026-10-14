"""
The orchestrator, with a fake clock and a fake detector.

The decision logic is tested in test_turntaking_rules; this tests the WIRING — that
answers go stale when new words arrive, that an empty flush is not sent, that the
detector saying "do not know" keeps us listening rather than firing.

    python3 tests/test_turntaking_session.py
"""

import asyncio
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from turntaking.config import TurnConfig          # noqa: E402
from turntaking.rules import Thresholds           # noqa: E402
from turntaking.session import TurnSession        # noqa: E402


class Clock:
    def __init__(self): self.t = 1000.0
    def __call__(self): return self.t
    def advance(self, s): self.t += s


class FakeDetector:
    """Returns whatever it is told to, and records what it was asked."""
    def __init__(self, answer=None):
        self.answer = answer
        self.calls = []

    async def is_complete(self, text, model):
        self.calls.append(text)
        return self.answer


def harness(detector=None, enabled=False, model=""):
    clock = Clock()
    flushes, states = [], []

    async def on_flush(text, reason): flushes.append((text, reason))
    # Third argument is the relevance verdicts; the harness ignores them so these
    # tests keep asserting on turn-taking alone.
    async def on_state(state, text, verdicts=None): states.append((state, text))

    s = TurnSession("sess-test", on_flush, on_state, detector=detector, clock=clock)
    cfg = TurnConfig(enabled=enabled, model=model, thresholds=Thresholds())
    return s, cfg, clock, flushes, states


def run(coro): return asyncio.get_event_loop().run_until_complete(coro)


# --- the basics ---------------------------------------------------------------------

def test_release_flushes_everything_including_the_guess():
    s, cfg, _, flushes, _ = harness()

    async def go():
        await s.final("book the flight", cfg)
        await s.partial("to Dallas", cfg)
        await s.release(cfg)
    run(go())
    assert flushes == [("book the flight to Dallas", "released")], flushes


def test_silence_flushes_a_finished_sentence():
    s, cfg, clock, flushes, _ = harness()

    async def go():
        await s.final("Book the flight.", cfg)
        clock.advance(0.7)
        await s._evaluate(cfg)
    run(go())
    assert flushes and flushes[0][1] == "sentence_end", flushes


def test_filler_is_never_flushed_by_silence():
    s, cfg, clock, flushes, _ = harness()

    async def go():
        await s.final("so ah", cfg)
        clock.advance(5.0)
        await s._evaluate(cfg)
    run(go())
    assert flushes == [], f"hesitation must not become a turn: {flushes}"


def test_provisional_text_alone_never_flushes():
    """take() hands over committed text only. If everything heard is still a guess,
    a 'send' decision must produce nothing rather than an empty turn."""
    s, cfg, clock, flushes, _ = harness()

    async def go():
        await s.partial("what time is the flight", cfg)
        clock.advance(3.0)
        await s._evaluate(cfg)
    run(go())
    assert flushes == [], f"an empty turn must never be sent: {flushes}"


# --- the detector -------------------------------------------------------------------

def test_detector_not_consulted_when_disabled():
    det = FakeDetector(answer=True)
    s, cfg, clock, flushes, _ = harness(detector=det, enabled=False)

    async def go():
        await s.final("what time is", cfg)
        clock.advance(0.6)
        await s._evaluate(cfg)
    run(go())
    assert det.calls == [], "a disabled detector must not be called"


def test_detector_complete_sends():
    det = FakeDetector(answer=True)
    s, cfg, clock, flushes, _ = harness(detector=det, enabled=True, model="tiny")

    async def go():
        await s.final("what time is it", cfg)
        clock.advance(0.6)
        await s._evaluate(cfg)
    run(go())
    assert det.calls, "the detector should have been asked"
    assert flushes and flushes[0][1] == "model_complete", flushes


def test_detector_dont_know_keeps_listening():
    """None means 'do not know'. It must resolve to keep listening, never to send."""
    det = FakeDetector(answer=None)
    s, cfg, clock, flushes, _ = harness(detector=det, enabled=True, model="tiny")

    async def go():
        await s.final("what time is", cfg)
        clock.advance(0.6)
        await s._evaluate(cfg)
    run(go())
    assert det.calls, "it should still have been asked"
    assert flushes == [], "an unknown answer must not produce a turn"


def test_safety_valve_beats_a_detector_saying_no():
    det = FakeDetector(answer=False)
    s, cfg, clock, flushes, _ = harness(detector=det, enabled=True, model="tiny")

    async def go():
        await s.final("and then the thing about the", cfg)
        clock.advance(2.0)                 # past max_silence
        await s._evaluate(cfg)
    run(go())
    assert flushes and flushes[0][1] == "max_silence", \
        f"no detector may hold a sentence past the valve: {flushes}"


def test_new_words_invalidate_a_previous_verdict():
    """The failure this prevents: the model judged 'what time is', then the speaker
    continued, and the stale verdict was applied to different words."""
    det = FakeDetector(answer=True)
    s, cfg, clock, flushes, _ = harness(detector=det, enabled=True, model="tiny")

    async def go():
        await s.final("what time is it", cfg)
        clock.advance(0.6)
        await s._evaluate(cfg)             # asks, gets COMPLETE, flushes
        det.answer = False
        await s.final("and where from", cfg)   # new words -> verdict must be re-sought
        clock.advance(0.6)
        await s._evaluate(cfg)
    run(go())
    assert len(det.calls) >= 2, f"it must ask again after new words: {det.calls}"
    assert len(flushes) == 1, f"the stale COMPLETE must not flush the new text: {flushes}"


# --- what the page is told -----------------------------------------------------------

def test_page_sees_buffering_then_flushed():
    s, cfg, _, _, states = harness()

    async def go():
        await s.partial("book the", cfg)
        await s.final("book the flight", cfg)
        await s.release(cfg)
    run(go())
    kinds = [k for k, _ in states]
    assert "buffering" in kinds, states
    assert kinds[-1] == "flushed", states


def test_buffering_is_not_repeated_for_unchanged_text():
    s, cfg, _, _, states = harness()

    async def go():
        await s.partial("hello", cfg)
        await s.partial("hello", cfg)
        await s.partial("hello", cfg)
    run(go())
    assert len([k for k, _ in states if k == "buffering"]) == 1, states


if __name__ == "__main__":
    asyncio.set_event_loop(asyncio.new_event_loop())
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in fns:
        try:
            fn(); print(f"  pass  {fn.__name__}")
        except AssertionError as e:
            failed += 1; print(f"  FAIL  {fn.__name__}: {e}")
        except Exception as e:
            failed += 1; print(f"  ERROR {fn.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    sys.exit(1 if failed else 0)
