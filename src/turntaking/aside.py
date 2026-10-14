"""
"Was that meant for us?" — telling a turn from the room it was spoken in.

An open microphone hears everything: a colleague in the doorway, a phone call, the
television, the speaker thinking out loud. The turn-taking detector next door asks
whether a thought is FINISHED. This asks whether it was ADDRESSED TO US, which is a
different question and fails in a much more expensive way.

That asymmetry drives every decision here:

  * A false ASIDE is the bad one. Dropping something a person actually said to you
    makes the assistant look deaf, and they have no way to tell what happened - the
    words were on screen a moment ago and then the bot ignored them. A false
    RELEVANT merely wastes a reply.
  * So "do not know" means KEEP, always. Timeouts keep, unparsed answers keep, an
    unreachable model keeps everything, and with the feature off nothing is judged.
  * NEVER DROP THE WHOLE TURN. If every sentence is judged an aside, the verdict is
    far more likely to be wrong than the speaker is to have said nothing to us.
    `surviving()` hands back the original text in that case.
  * Judgements are shown, not just applied. The page strikes an aside through rather
    than deleting it, so a wrong call is visible and arguable instead of silent.

Pure except for the detector call, so the splitting and the keep/drop arithmetic can
be tested without a model.
"""

import structlog

log = structlog.get_logger()

# Short fragments are not judged at all. "yes", "no", "wait" carry almost no signal
# for this question, and they are exactly the words that a conversation turns on.
MIN_JUDGE_CHARS = 14
# Nothing sentence-like has been seen past this; judge what is there rather than
# letting one long unpunctuated stretch go unexamined forever.
MAX_SEGMENT_CHARS = 240

_ENDS = ".?!"


def split_sentences(text: str) -> list:
    """Split into sentence-ish segments, keeping their punctuation.

    Deliberately crude: recogniser output has little punctuation and no capitals to
    speak of, so anything cleverer would be inventing structure that is not there. A
    long unpunctuated run is cut at MAX_SEGMENT_CHARS on a word boundary.
    """
    t = (text or "").strip()
    if not t:
        return []
    out, cur = [], ""
    for ch in t:
        cur += ch
        if ch in _ENDS:
            out.append(cur.strip())
            cur = ""
    if cur.strip():
        out.append(cur.strip())

    final = []
    for seg in out:
        while len(seg) > MAX_SEGMENT_CHARS:
            cut = seg.rfind(" ", 0, MAX_SEGMENT_CHARS)
            if cut <= 0:
                cut = MAX_SEGMENT_CHARS
            final.append(seg[:cut].strip())
            seg = seg[cut:].strip()
        if seg:
            final.append(seg)
    return [s for s in final if s]


def surviving(text: str, verdicts: dict) -> str:
    """What should actually be sent, given what has been judged.

    Anything unjudged survives - that is the "do not know means keep" rule, and it is
    why a dead model costs nothing. And if the verdicts would silence the turn
    entirely, they are discarded wholesale: a speaker who said something is far more
    likely than a judge that got every segment wrong.
    """
    segs = split_sentences(text)
    if not segs:
        return text
    kept = [s for s in segs if verdicts.get(s, True)]
    if not kept:
        log.info("aside_all_dropped", note="keeping the whole turn instead", segs=len(segs))
        return text
    return " ".join(kept).strip()


class AsideJudge:
    """Judges segments once each, and remembers the answers for this turn."""

    def __init__(self, detector=None):
        self._detector = detector
        self._verdicts: dict = {}      # segment -> True (relevant) / False (aside)
        self._pending: set = set()

    @property
    def verdicts(self) -> dict:
        return dict(self._verdicts)

    def reset(self) -> None:
        """A flush ends the turn, and with it the context these verdicts were made in."""
        self._verdicts.clear()
        self._pending.clear()

    def unjudged(self, text: str, include_last: bool = False) -> list:
        """Segments worth asking about that have not been asked about yet.

        The LAST segment is skipped while the speaker is still talking: it is the one
        they are part-way through, so judging it means judging half a sentence. On
        release there is no more coming and it is included.
        """
        segs = split_sentences(text)
        if not segs:
            return []
        if not include_last and not segs[-1].endswith(tuple(_ENDS)):
            segs = segs[:-1]
        return [s for s in segs
                if len(s) >= MIN_JUDGE_CHARS
                and s not in self._verdicts
                and s not in self._pending]

    async def judge(self, segment: str, context: str, model: str) -> None:
        """Ask about one segment. Never raises; an unanswerable question just leaves
        the segment unjudged, which means kept."""
        if not self._detector or not model or segment in self._verdicts:
            return
        self._pending.add(segment)
        try:
            verdict = await self._detector.is_relevant(segment, context, model)
        except Exception as e:                                   # noqa: BLE001
            log.debug("aside_judge_failed", error=str(e))
            verdict = None
        finally:
            self._pending.discard(segment)
        if verdict is None:
            return                                               # do not know -> keep
        self._verdicts[segment] = verdict
        if not verdict:
            log.info("aside_marked", text=segment[:60])
