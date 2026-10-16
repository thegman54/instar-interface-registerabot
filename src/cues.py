"""Pre-rendered filler clips, chosen by ACTION CLASS and paced by the browser.

Why this exists
---------------
Narrating tool calls with live TTS was tried and it sounds wrong: the phrase is her
`why`, written for a log ("Alex asked to search captain's logs for..."), spoken in third
person about the person listening; and because a turn makes several calls she talks over
herself, twice saying the same line. Measured on a live turn: four clips between 2.75s
and 8.48s.

So nothing here renders at turn time and nothing here speaks a phrase the model wrote.

  tool          declares nothing yet (see TODO)
  this module   maps tool name -> a CLOSED action class
  this module   renders a few fixed clips PER CLASS, once, and caches them
  the page      decides whether playing one right now is a good idea

That last line is the real design. The server cannot know whether a clip should play —
it does not know what is on screen, whether she is already speaking, or how long ago the
last one went. Only the browser knows, so the server sends a CUE (a class and nothing
else) and the browser owns the timer, the variant choice and the right to ignore it.
Same split as the rest of this system: the bot says what is happening, the surface
decides what that looks like.

Clips are generated, not recorded: TTS is 0.4-0.6s for a short phrase and FLOAT renders
at ~1.15x realtime, so the whole library is ~25s of one-time work. Keyed by voice group,
so changing her voice re-warms rather than requiring a new recording session.

TODO: `action_class` belongs on the tool, declared the way capabilities are. The keyword
map below is an interim that works for 135 tools without touching any of them, and it
will be wrong for some. Moving it to the tool is the fix, not a longer keyword list.
"""

import asyncio
import base64
import os
import time
from pathlib import Path

import httpx
import structlog

log = structlog.get_logger()

CUE_DIR = Path(os.environ.get('CUE_CACHE_DIR', '/state/cues'))

# A CLOSED set. An unmapped tool becomes `work` — never silence by accident, and never a
# class invented at runtime, because the clip library is built from exactly these keys.
#
# Lines are short and deliberately NON-FINAL: rising or continuing, so the real answer
# lands as the rest of the same thought instead of a second separate reply. Nothing here
# promises a result, because the tool may still fail.
PHRASES: dict[str, list[str]] = {
    'think':  ["Hmm.", "One second.", "Let me think."],
    'check':  ["Checking.", "Let me check.", "One sec."],
    'search': ["Let me look.", "Searching.", "Let me go find that."],
    'render': ["This takes a moment.", "Making that now.", "Give me a second."],
    'work':   ["Working on it.", "On it.", "Just a moment."],
}

CLASSES = tuple(PHRASES.keys())

# Longest match wins, so 'image_create' is `render` rather than `check` on 'create'.
# Ordered most-specific first within each class for readability only.
_KEYWORDS: list[tuple[str, str]] = [
    ('search', 'search'), ('find', 'search'), ('query', 'search'), ('grep', 'search'),
    ('lookup', 'search'), ('consult', 'search'), ('expert', 'search'), ('recall', 'search'),

    ('render', 'render'), ('image', 'render'), ('video', 'render'), ('speech', 'render'),
    ('draw', 'render'), ('map_render', 'render'), ('imagen', 'render'), ('generate', 'render'),

    ('status', 'check'), ('read', 'check'), ('list', 'check'), ('info', 'check'),
    ('show', 'check'), ('get', 'check'), ('snapshot', 'check'), ('verify', 'check'),

    ('toolsearch', 'think'),
]


def class_for_tool(name: str) -> str:
    """Which action class a tool call belongs to. Never raises, never returns unknown."""
    n = (name or '').lower()
    if not n:
        return 'work'
    best, best_len = 'work', 0
    for kw, cls in _KEYWORDS:
        if kw in n and len(kw) > best_len:
            best, best_len = cls, len(kw)
    return best


def _clip_path(voice_group_id: str, avatar_id: str, cls: str, variant: int) -> Path:
    safe = f"{voice_group_id}_{avatar_id}".replace('/', '_')
    return CUE_DIR / safe / f"{cls}-{variant}.mp4"


class CueLibrary:
    """Renders and caches the clip set for one (voice, avatar) pair.

    Generation is best-effort and happens off the turn path. A missing clip means the
    page gets no cue for that class, which is the behaviour before any of this existed —
    silence is a legitimate outcome, and a wrong clip is worse than none because it is
    wrong out loud.
    """

    def __init__(self) -> None:
        self._building: set[tuple[str, str]] = set()
        self._lock = asyncio.Lock()

    # ---- generation ------------------------------------------------------------

    async def _tts(self, client: httpx.AsyncClient, admin: str,
                   voice_group_id: str, text: str) -> bytes | None:
        r = await client.post(f"{admin}/api/tts/generate",
                              json={'voice_group_id': voice_group_id,
                                    'text': text, 'format': 'wav'})
        if r.status_code != 200:
            log.warning("cue_tts_failed", status=r.status_code, text=text)
            return None
        return r.content

    async def _render(self, client: httpx.AsyncClient, face_url: str,
                      avatar_id: str, wav: bytes) -> bytes | None:
        r = await client.post(f"{face_url}/render",
                              files={'audio': ('cue.wav', wav, 'audio/wav')},
                              data={'avatar_id': avatar_id})
        if r.status_code != 200:
            log.warning("cue_render_failed", status=r.status_code, avatar=avatar_id)
            return None
        return r.content

    async def ensure(self, voice_group_id: str, avatar_id: str) -> None:
        """Render whatever is missing. Safe to call repeatedly; one build per pair."""
        if not voice_group_id or not avatar_id:
            return
        key = (voice_group_id, avatar_id)
        async with self._lock:
            if key in self._building:
                return
            if self._complete(voice_group_id, avatar_id):
                return
            self._building.add(key)
        started = time.time()
        made = 0
        admin = os.environ.get('ADMIN_API_URL', 'http://gatekeeper:8090')
        face_url = os.environ.get('FACE_RENDERER_URL', 'http://instar-float-renderer:8098')
        try:
            async with httpx.AsyncClient(timeout=300.0) as client:
                for cls, lines in PHRASES.items():
                    for i, line in enumerate(lines):
                        p = _clip_path(voice_group_id, avatar_id, cls, i)
                        if p.exists() and p.stat().st_size > 0:
                            continue
                        wav = await self._tts(client, admin, voice_group_id, line)
                        if not wav:
                            continue
                        mp4 = await self._render(client, face_url, avatar_id, wav)
                        if not mp4:
                            continue
                        p.parent.mkdir(parents=True, exist_ok=True)
                        # Write-then-rename: a half-written clip that the next request
                        # finds "existing" would be served as a corrupt video forever.
                        tmp = p.with_suffix('.part')
                        tmp.write_bytes(mp4)
                        tmp.rename(p)
                        made += 1
            log.info("cue_library_built", voice=voice_group_id, avatar=avatar_id,
                     made=made, seconds=round(time.time() - started, 1))
        except Exception as e:                                        # noqa: BLE001
            log.warning("cue_library_build_error", error=str(e),
                        voice=voice_group_id, avatar=avatar_id)
        finally:
            self._building.discard(key)

    def _complete(self, voice_group_id: str, avatar_id: str) -> bool:
        for cls, lines in PHRASES.items():
            for i in range(len(lines)):
                p = _clip_path(voice_group_id, avatar_id, cls, i)
                if not (p.exists() and p.stat().st_size > 0):
                    return False
        return True

    # ---- delivery --------------------------------------------------------------

    def manifest(self, voice_group_id: str, avatar_id: str) -> list[dict]:
        """Every clip that exists, as base64, one entry per clip.

        Sent as separate frames: the relay drops a WebSocket frame over 1 MiB, and the
        whole library in one envelope would exceed that and take the conversation with it.
        """
        out: list[dict] = []
        for cls, lines in PHRASES.items():
            for i in range(len(lines)):
                p = _clip_path(voice_group_id, avatar_id, cls, i)
                try:
                    if not (p.exists() and p.stat().st_size > 0):
                        continue
                    raw = p.read_bytes()
                except OSError:
                    continue
                out.append({
                    'cue_class': cls,
                    'variant': i,
                    'format': 'mp4',
                    'base64': base64.b64encode(raw).decode(),
                })
        return out
