"""
The flush ladder, tested without a microphone, a GPU or a running stack.

Every case here is a sentence someone could actually say. The ones that matter most
are the two failure directions: handing the bot a syllable, and leaving a finished
speaker waiting.

    python3 -m pytest tests/test_turntaking_rules.py -q
    python3 tests/test_turntaking_rules.py          # no pytest needed
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from turntaking.rules import (                     # noqa: E402
    Thresholds, decide, is_filler_only, has_hard_marker, ends_sentence,
)

T = Thresholds()


def check(name, got, want):
    assert got == want, f"{name}: got {got!r}, wanted {want!r}"
    return True


# --- filler: the complaint that started this -----------------------------------------

def test_filler_is_never_a_turn():
    for text in ["so", "so ah", "so ah maybe", "um", "uh well", "like just"]:
        assert is_filler_only(text), text
        # Not even after a long pause. Answering hesitation is worse than silence.
        d = decide(text, silence_s=5.0, thresholds=T)
        check(f"filler {text!r}", (d.send, d.reason), (False, "filler_only"))


def test_nothing_said_is_distinct_from_filler():
    """Both hold, but for different reasons, and the reason is what gets logged."""
    for text in ["  ", "", "\n"]:
        d = decide(text, silence_s=5.0, thresholds=T)
        check(f"empty {text!r}", (d.send, d.reason), (False, "empty"))


def test_filler_plus_content_is_content():
    assert not is_filler_only("so what is the weather")
    d = decide("so what is the weather", silence_s=2.0, thresholds=T)
    assert d.send, "a real question behind a filler must still go"


# --- hard markers: must not wait ------------------------------------------------------

def test_stop_goes_immediately():
    d = decide("stop", silence_s=0.0, thresholds=T)
    check("stop", (d.send, d.reason), (True, "hard_marker"))


def test_marker_beats_min_silence():
    # 0.0s of silence would normally mean 'still speaking'.
    assert decide("wait", silence_s=0.0, thresholds=T).send


def test_stop_inside_a_sentence_is_not_a_marker():
    # 'stop by the shop' is not an instruction to stop.
    assert not has_hard_marker("stop by the shop on the way")
    d = decide("stop by the shop on the way", silence_s=0.1, thresholds=T)
    assert not d.send, "a sentence containing 'stop' must not be treated as a command"


def test_short_trailing_marker_counts():
    assert has_hard_marker("ok stop")
    assert has_hard_marker("no wait")


# --- release: a person saying they are done outranks every heuristic -----------------

def test_release_sends_even_mid_sentence():
    d = decide("and then I want to", silence_s=0.0, released=True, thresholds=T)
    check("released", (d.send, d.reason), (True, "released"))


def test_release_sends_even_a_one_word_answer():
    """The flaw this caught: 'maybe' and 'yeah' look like noise and are answers.

    Suppressing them leaves someone repeating themselves at a bot that heard fine.
    Release is a declaration of intent, so it outranks the filler rule.
    """
    for word in ["maybe", "yeah", "ok", "sure", "right"]:
        d = decide(word, silence_s=0.0, released=True, thresholds=T)
        assert d.send, f"a deliberate one-word answer {word!r} must not be swallowed"


def test_release_with_nothing_said_sends_nothing():
    d = decide("   ", silence_s=0.0, released=True, thresholds=T)
    assert not d.send, "releasing having said nothing is not a turn"


def test_short_answers_are_not_treated_as_filler():
    for word in ["yeah", "ok", "right", "sure"]:
        assert not is_filler_only(word), f"{word!r} is a complete answer, not hesitation"


# --- timing ---------------------------------------------------------------------------

def test_short_gap_holds():
    d = decide("I was thinking we could", silence_s=0.2, thresholds=T)
    check("short gap", (d.send, d.reason), (False, "still_speaking"))


def test_sentence_plus_breath_sends():
    d = decide("Book the flight.", silence_s=0.6, thresholds=T)
    check("sentence end", (d.send, d.reason), (True, "sentence_end"))


def test_sentence_without_the_breath_waits():
    d = decide("Book the flight.", silence_s=0.4, thresholds=T)
    assert not d.send, "punctuation alone is not an ending; people pause mid-thought"


def test_max_silence_is_a_safety_valve():
    # Incomplete, and the model never answered. It must STILL go.
    d = decide("and then the thing about the", silence_s=1.5,
               thresholds=T, model_says_complete=None)
    check("max silence", (d.send, d.reason), (True, "max_silence"))


def test_model_cannot_hold_words_hostage():
    # Model actively says 'not complete'. Past max_silence that does not matter.
    d = decide("I wanted to ask about the", silence_s=2.0,
               thresholds=T, model_says_complete=False)
    assert d.send, "no detector may keep a person waiting past the safety valve"


# --- the model is consulted only in the ambiguous band -------------------------------

def test_model_asked_only_when_ambiguous():
    d = decide("I wanted to ask about", silence_s=0.6, thresholds=T)
    check("ask", (d.send, d.ask_model, d.reason), (False, True, "needs_model"))


def test_model_not_asked_below_the_band():
    d = decide("I wanted to ask about", silence_s=0.2, thresholds=T)
    assert not d.ask_model, "no model call while they are plainly still speaking"


def test_model_not_asked_above_the_band():
    d = decide("I wanted to ask about", silence_s=2.0, thresholds=T)
    assert not d.ask_model, "past the valve the answer is irrelevant, so do not pay for it"


def test_model_answers_are_honoured_inside_the_band():
    yes = decide("what time is it", silence_s=0.6, thresholds=T, model_says_complete=True)
    no = decide("what time is", silence_s=0.6, thresholds=T, model_says_complete=False)
    check("model yes", (yes.send, yes.reason), (True, "model_complete"))
    check("model no", (no.send, no.reason), (False, "model_incomplete"))


# --- the two failure directions, stated as one test each -----------------------------

def test_never_hands_the_bot_a_syllable():
    """The original complaint: a turn per word."""
    for frag, gap in [("so", 0.6), ("so ah", 0.9), ("um", 2.0), ("well", 0.7)]:
        assert not decide(frag, silence_s=gap, thresholds=T).send, frag


def test_never_leaves_a_finished_speaker_waiting():
    """The opposite failure: they stopped and nothing happened."""
    for done, gap in [("Book the flight.", 0.6), ("what time is it", 1.3),
                      ("tell me about mars", 2.0)]:
        assert decide(done, silence_s=gap, thresholds=T).send, done


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
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    sys.exit(1 if failed else 0)
