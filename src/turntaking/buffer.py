"""
What has been heard, and how much of it is safe to believe.

Speech recognisers do not emit a stream of words. They emit a guess that keeps
changing: "so I'm" becomes "so I'm going" and then rewrites itself to "so I'm gonna".
Forwarding fragments as they appear therefore sends text that is later revised, and
the bot answers something the speaker never said.

So the buffer has two halves:

    committed   finalised by the recogniser. Will not change. Safe to send.
    tail        the current guess. Replaced wholesale on every update.

Only `committed` is ever flushed. The tail exists to be SHOWN - greyed out, so the
speaker can watch their words arrive before they count - and to let the rules look
ahead at what is probably coming.

Keeping these apart is the difference between a page that shows you what it heard and
a page that argues with you about what you said.
"""

import time
from dataclasses import dataclass, field


@dataclass
class TranscriptBuffer:
    """Accumulates recogniser output for one speaking turn."""

    committed: str = ""
    tail: str = ""
    last_voice_at: float = field(default_factory=time.monotonic)
    # Monotonic, not wall clock: the whole pipeline reasons about elapsed silence, and
    # a clock that can step backwards across an NTP correction would flush mid-word.
    _clock: callable = time.monotonic

    # --- input -----------------------------------------------------------------

    def final(self, text: str) -> None:
        """The recogniser has committed to these words. They will not change again."""
        t = (text or "").strip()
        if not t:
            return
        self.committed = (self.committed + " " + t).strip() if self.committed else t
        self.tail = ""
        self.last_voice_at = self._clock()

    def interim(self, text: str) -> None:
        """The current guess. REPLACES the tail; it never appends.

        Appending here is the bug this class exists to prevent - interim results are
        revisions of each other, so concatenating them produces 'so I'm so I'm going
        so I'm gonna'.
        """
        t = (text or "").strip()
        if t:
            self.last_voice_at = self._clock()
        self.tail = t

    def voice(self) -> None:
        """Audio energy with no transcript yet. Still counts as not-silent."""
        self.last_voice_at = self._clock()

    # --- reading ---------------------------------------------------------------

    @property
    def text(self) -> str:
        """Everything heard so far, committed and provisional. For DISPLAY and for
        the rules to look ahead with - never the thing that gets sent."""
        return (self.committed + " " + self.tail).strip()

    @property
    def sendable(self) -> str:
        """Only what the recogniser will not take back."""
        return self.committed.strip()

    def silence_s(self) -> float:
        return max(0.0, self._clock() - self.last_voice_at)

    def is_empty(self) -> bool:
        return not self.text

    # --- flushing --------------------------------------------------------------

    def take(self) -> str:
        """Hand over the committed text and clear it.

        The tail SURVIVES. A flush happens because the speaker finished a thought, not
        because they stopped existing - whatever they are part-way through saying is
        still part-way through being said, and belongs to the next turn.
        """
        out = self.committed.strip()
        self.committed = ""
        self.last_voice_at = self._clock()
        return out

    def take_all(self) -> str:
        """Everything, including the unfinished guess.

        For release only: the button came up, so there will be no further revision and
        waiting for the recogniser to finalise would just drop the last word or two.
        """
        out = self.text
        self.committed = ""
        self.tail = ""
        self.last_voice_at = self._clock()
        return out

    def reset(self) -> None:
        self.committed = ""
        self.tail = ""
        self.last_voice_at = self._clock()
