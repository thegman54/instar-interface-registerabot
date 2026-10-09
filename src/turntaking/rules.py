"""
When has someone finished a thought?

This is the whole problem. Everything else in the voice pipeline is plumbing around
this one decision: the speaker is mid-sentence and we have to choose, right now,
whether to hand what we have to the bot or keep listening.

Get it wrong one way and the bot is handed "so" as a turn and answers it. Get it wrong
the other way and the speaker finishes, waits, and nothing happens.

The decision is a LADDER, first match wins, and the ordering is the design:

    1  nothing said         never send
    2  button released      send now        a person saying they are done
    3  filler only          hold            "so", "um", "so ah" are not input
    4  hard marker          send now        "stop" must not wait on anything
    5  silence < MIN        hold            still mid-sentence
    6  punctuation + pause  send            a finished sentence and a breath
    7  silence > MAX        send            the safety valve
    8  silence > ASK        ask the model   genuinely ambiguous

Rules 1-6 cost nothing. Only rule 7 consults a model, which is why most turns never
touch one: a model call per fragment would add hundreds of milliseconds each and spend
real money deciding that "so ah" is not a sentence.

Rule 6 is the safety valve and it matters more than it looks. A detector that cannot
decide must never be able to hold a person's words hostage — past MAX_SILENCE the
buffer goes, complete or not, model or no model.

These functions are PURE. No clock, no network, no state: every input is an argument.
That is deliberate — this is the part that has to be right, so it has to be testable
without a GPU, a microphone or a running stack.
"""

from dataclasses import dataclass
from typing import Optional
import re


@dataclass(frozen=True)
class Thresholds:
    """Timing, in seconds. Defaults are a starting point, not a result."""
    min_silence: float = 0.35      # below this, they are still talking
    ask_silence: float = 0.50      # long enough to be worth asking about
    punct_silence: float = 0.50    # a full stop plus a breath is an ending
    max_silence: float = 1.20      # the safety valve; nothing waits longer


@dataclass(frozen=True)
class Decision:
    send: bool
    reason: str
    ask_model: bool = False        # caller should consult the detector, then re-decide


# Words that mean "act now" and must never wait on a pause or a model. Deliberately
# short: every entry here is a word that can no longer be used as the start of a longer
# sentence, so a wrong one costs a truncated turn. Grow it from real transcripts.
HARD_MARKERS = {"stop", "wait", "cancel", "no", "nevermind", "never mind", "shut up"}

# Sounds people make while deciding what to say, plus conjunctions and articles left
# dangling mid-thought. If the buffer is ONLY these, there is no thought to finish.
#
# "yeah", "ok", "right", "sure" are deliberately NOT here. They look like noise and
# they are complete answers to a yes/no question - suppressing them would swallow a
# reply and leave the speaker repeating themselves at a bot that heard them fine.
#
# "maybe" IS here, because alone it is hesitation; "maybe tomorrow" is not filler-only
# and still goes. That is the whole reason this tests every word rather than the first.
FILLERS = {
    "so", "um", "uh", "ah", "er", "erm", "hmm", "mm", "like", "well", "maybe",
    "and", "but", "i", "its", "it's", "the", "a", "just", "kinda", "sorta",
}

_WORD = re.compile(r"[a-z']+")
TERMINAL = (".", "?", "!")


def normalise(text: str) -> str:
    return " ".join((text or "").lower().split())


def words(text: str) -> list[str]:
    return _WORD.findall(normalise(text))


def is_filler_only(text: str) -> bool:
    """True if there is no content here, only hesitation.

    Empty counts as filler: there is nothing to send either way, and treating it
    separately just gives the caller a second empty case to forget about.
    """
    w = words(text)
    if not w:
        return True
    return all(x in FILLERS for x in w)


def has_hard_marker(text: str) -> bool:
    """A command that must interrupt, not queue.

    Matched against the WHOLE utterance, not as a substring: "stop" is an instruction,
    "stop by the shop" is not, and "don't stop" means the opposite of the word in it.
    """
    n = normalise(text).strip(" .,!?")
    if n in HARD_MARKERS:
        return True
    # Also catch a marker that ends a short utterance - "ok stop", "no wait".
    w = words(n)
    return len(w) <= 3 and bool(w) and w[-1] in HARD_MARKERS


def ends_sentence(text: str) -> bool:
    return normalise(text).endswith(TERMINAL)


def decide(text: str, silence_s: float, released: bool = False,
           thresholds: Optional[Thresholds] = None,
           model_says_complete: Optional[bool] = None) -> Decision:
    """Should this buffer be handed to the bot?

    `silence_s`   how long since speech was last detected
    `released`    the push-to-talk button came up: they have declared they are done
    `model_says_complete`
                  the detector's answer, if it has already been consulted this round.
                  None means "not asked yet" and is what produces ask_model.
    """
    t = thresholds or Thresholds()

    # 1. Nothing at all. Not a turn under any rule below.
    if not words(text):
        return Decision(False, "empty")

    # 2. They let go of the button. A person declaring they are finished outranks every
    #    heuristic here - the heuristics exist only to GUESS at this, so when the real
    #    signal arrives they stop applying. That includes the filler rule: someone who
    #    held a button to say "maybe" chose to say it, and swallowing a deliberate
    #    one-word answer is worse than forwarding a shrug.
    if released:
        return Decision(True, "released")

    # 3. Hesitation only. Never an automatic turn, however long the pause - answering
    #    "so ah" is worse than answering nothing. This suppresses the SILENCE-driven
    #    path, which is the one that fires repeatedly mid-sentence and produced the
    #    turn-per-word problem in the first place.
    if is_filler_only(text):
        return Decision(False, "filler_only")

    # 4. An instruction that is useless late.
    if has_hard_marker(text):
        return Decision(True, "hard_marker")

    # 4. Still inside the gaps that happen mid-sentence.
    if silence_s < t.min_silence:
        return Decision(False, "still_speaking")

    # 5. A complete sentence followed by a real breath.
    if ends_sentence(text) and silence_s >= t.punct_silence:
        return Decision(True, "sentence_end")

    # 6. THE SAFETY VALVE. Past here nothing is allowed to keep waiting - not an
    #    undecided model, not an unreachable one, not a missing one.
    if silence_s >= t.max_silence:
        return Decision(True, "max_silence")

    # 7. Ambiguous, and worth the cost of asking.
    if silence_s >= t.ask_silence:
        if model_says_complete is None:
            return Decision(False, "needs_model", ask_model=True)
        if model_says_complete:
            return Decision(True, "model_complete")
        return Decision(False, "model_incomplete")

    return Decision(False, "waiting")
