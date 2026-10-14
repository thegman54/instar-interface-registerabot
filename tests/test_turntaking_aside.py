"""
"Was that meant for us?" — the keep/drop arithmetic, tested without a model.

This is the only part of the pipeline that can REMOVE words a person actually said,
so the tests are weighted towards the expensive direction: every case where the
judge is wrong, absent, slow or unanimous must still end with the speaker heard.

    python3 -m pytest tests/test_turntaking_aside.py -q
    python3 tests/test_turntaking_aside.py          # no pytest needed
"""

import asyncio
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from turntaking.aside import (                     # noqa: E402
    AsideJudge, MIN_JUDGE_CHARS, split_sentences, surviving,
)


# --- splitting ------------------------------------------------------------------

def test_splits_on_sentence_ends():
    got = split_sentences("can you check the build? no not you mike. what broke")
    assert got == ["can you check the build?", "no not you mike.", "what broke"], got


def test_empty_text_is_no_segments():
    assert split_sentences("") == []
    assert split_sentences("   ") == []


def test_long_unpunctuated_run_is_still_cut():
    """Dictation often arrives with no punctuation at all. One endless segment would
    never be judged, which quietly disables the feature for exactly the speakers who
    talk the most."""
    long = " ".join(["word"] * 200)
    segs = split_sentences(long)
    assert len(segs) > 1
    assert all(len(s) <= 240 for s in segs)
    # And nothing is lost in the cutting.
    assert " ".join(segs).split() == long.split()


# --- what actually gets sent ------------------------------------------------------

def test_aside_is_removed_from_a_mixed_turn():
    text = "can you check the build? no not you mike. what broke"
    out = surviving(text, {"no not you mike.": False})
    assert out == "can you check the build? what broke", out


def test_unjudged_segments_survive():
    """No verdict means keep. A dead or slow model must cost nothing."""
    text = "can you check the build? what broke"
    assert surviving(text, {}) == text


def test_a_turn_judged_entirely_aside_is_kept_whole():
    """The most expensive failure available. If every segment is called an aside, the
    judge is far likelier to be wrong than the speaker is to have said nothing — so
    the verdicts are discarded rather than the turn."""
    text = "no not you mike."
    assert surviving(text, {text: False}) == text


def test_relevant_verdicts_change_nothing():
    text = "what broke in the build?"
    assert surviving(text, {text: True}) == text


def test_text_with_no_segments_passes_through():
    assert surviving("", {}) == ""


# --- what gets asked about --------------------------------------------------------

def test_unfinished_last_segment_is_not_judged_mid_sentence():
    """Judging the sentence someone is still speaking means judging half of it."""
    j = AsideJudge()
    pending = j.unjudged("tell me about the outage. and then can you")
    assert pending == ["tell me about the outage."], pending


def test_release_judges_the_last_segment_too():
    """The button came up; nothing more is coming, so the tail is fair game."""
    j = AsideJudge()
    pending = j.unjudged("tell me about the outage. and then can you check logs",
                         include_last=True)
    assert len(pending) == 2, pending


def test_short_fragments_are_never_judged():
    """'yes' and 'no' carry almost no signal for this question and are exactly the
    words a conversation turns on."""
    j = AsideJudge()
    assert j.unjudged("yes. no. ok.", include_last=True) == []
    assert MIN_JUDGE_CHARS > 3


def test_a_segment_is_only_asked_about_once():
    j = AsideJudge(_FakeDetector(False))
    text = "that is definitely not for the assistant."
    asyncio.get_event_loop_policy().new_event_loop().run_until_complete(
        _judge_all(j, text))
    assert j.unjudged(text, include_last=True) == []


# --- the judge's failure modes ----------------------------------------------------

class _FakeDetector:
    def __init__(self, verdict, raises=False):
        self.verdict, self.raises, self.calls = verdict, raises, 0

    async def is_relevant(self, text, context, model):
        self.calls += 1
        if self.raises:
            raise RuntimeError("ollama is down")
        return self.verdict


async def _judge_all(j, text, model="m"):
    for seg in j.unjudged(text, include_last=True):
        await j.judge(seg, "", model)


def _run(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


def test_a_raising_detector_leaves_everything_kept():
    j = AsideJudge(_FakeDetector(None, raises=True))
    text = "something that is plainly long enough to be judged."
    _run(_judge_all(j, text))
    assert j.verdicts == {}
    assert surviving(text, j.verdicts) == text


def test_dont_know_leaves_everything_kept():
    j = AsideJudge(_FakeDetector(None))
    text = "something that is plainly long enough to be judged."
    _run(_judge_all(j, text))
    assert j.verdicts == {}


def test_no_model_means_nothing_is_asked():
    d = _FakeDetector(False)
    j = AsideJudge(d)
    _run(_judge_all(j, "something long enough to be judged here.", model=""))
    assert d.calls == 0
    assert j.verdicts == {}


def test_reset_clears_the_turn():
    j = AsideJudge(_FakeDetector(False))
    text = "that was said to someone else entirely."
    _run(_judge_all(j, text))
    assert j.verdicts
    j.reset()
    assert j.verdicts == {}


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"  pass  {fn.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"  FAIL  {fn.__name__}: {e}")
        except Exception as e:                                   # noqa: BLE001
            failed += 1
            print(f"  ERROR {fn.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    sys.exit(1 if failed else 0)
