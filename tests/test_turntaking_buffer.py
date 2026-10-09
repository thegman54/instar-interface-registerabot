"""
The buffer, with a fake clock so timing is tested rather than waited on.

    python3 tests/test_turntaking_buffer.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from turntaking.buffer import TranscriptBuffer      # noqa: E402


class Clock:
    """A clock that only moves when the test says so."""
    def __init__(self): self.t = 1000.0
    def __call__(self): return self.t
    def advance(self, s): self.t += s


def new():
    c = Clock()
    b = TranscriptBuffer(_clock=c)
    b.last_voice_at = c()
    return b, c


# --- the trap this class exists for -------------------------------------------------

def test_interim_replaces_and_never_appends():
    """Recognisers REVISE. Appending gives 'so I'm so I'm going so I'm gonna'."""
    b, _ = new()
    b.interim("so I'm")
    b.interim("so I'm going")
    b.interim("so I'm gonna")
    assert b.text == "so I'm gonna", b.text


def test_interim_is_never_sendable():
    b, _ = new()
    b.interim("probably wrong")
    assert b.sendable == "", "a guess must never be forwarded as a turn"
    assert b.text == "probably wrong", "but it must still be visible"


def test_final_accumulates():
    b, _ = new()
    b.final("book the flight")
    b.final("to Dallas")
    assert b.sendable == "book the flight to Dallas"


def test_final_clears_the_stale_guess():
    b, _ = new()
    b.interim("book the fli")
    b.final("book the flight")
    assert b.text == "book the flight", "the guess must not survive its own commit"


# --- flushing -----------------------------------------------------------------------

def test_take_leaves_the_unfinished_guess_behind():
    """A flush means a thought ended, not that the speaker vanished."""
    b, _ = new()
    b.final("book the flight.")
    b.interim("and also")
    assert b.take() == "book the flight."
    assert b.sendable == ""
    assert b.text == "and also", "mid-word speech belongs to the next turn"


def test_take_all_includes_the_guess():
    b, _ = new()
    b.final("book the flight")
    b.interim("to Dallas")
    assert b.take_all() == "book the flight to Dallas"
    assert b.is_empty()


# --- silence ------------------------------------------------------------------------

def test_silence_grows_and_speech_resets_it():
    b, c = new()
    c.advance(0.8)
    assert abs(b.silence_s() - 0.8) < 1e-6
    b.interim("hello")
    assert b.silence_s() < 1e-6, "a guess is still evidence of speaking"


def test_voice_without_words_counts_as_speech():
    """Audio energy with no transcript yet must not read as silence, or the rules
    flush mid-breath while the recogniser is still catching up."""
    b, c = new()
    c.advance(2.0)
    b.voice()
    assert b.silence_s() < 1e-6


def test_take_resets_the_silence_clock():
    b, c = new()
    b.final("done.")
    c.advance(3.0)
    b.take()
    assert b.silence_s() < 1e-6, "the next turn starts its own clock"


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in fns:
        try:
            fn(); print(f"  pass  {fn.__name__}")
        except AssertionError as e:
            failed += 1; print(f"  FAIL  {fn.__name__}: {e}")
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    sys.exit(1 if failed else 0)
