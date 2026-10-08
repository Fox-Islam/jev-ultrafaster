"""Observed actions through Browser Harness; one CDP session, no per-step subprocess."""

import atexit
import hashlib
import json
import os
import sys
import threading
import time
from pathlib import Path

from browser_harness.admin import ensure_daemon, restart_daemon
from browser_harness.helpers import NAME as DAEMON
from browser_harness.helpers import cdp, drain_events

# Atomically read visible content and controls, preserving actual DOM node identity.
READ_STATE = Path(__file__).with_name("snapshot.js").read_text()
MARKER = f"(() => {{ const state={READ_STATE}; return state?.marker ?? null; }})()"

# A frame narrower or shorter than this, or starting beyond the window, carries nothing a decision
# could act on. The window matches the size the page is emulated at.
USABLE_FRAME = 60

# How long a document must go unchanged before a decision is taken about it, as a duration and not
# a number of reads: three reads a third of a second apart is a second of stillness, and the same
# three polled quickly is a seventh of one. A duration is unaffected by how often it is checked.
SETTLE_STILL = float(os.environ.get("JEV_SETTLE_STILL", "0.3"))
# A hosted endpoint is reached over the network, where every look costs a round trip rather than
# a pipe. Waiting is bounded harder there because the wait is what the round trips are spent on.
REMOTE = bool(os.environ.get("BU_CDP_WS"))
SETTLE_TIMEOUT = float(os.environ.get("JEV_SETTLE_TIMEOUT", "1.5" if REMOTE else "6"))
SETTLE_POLL = float(os.environ.get("JEV_SETTLE_POLL", "0.05"))

# How long the screen must go unpainted to count as still. The browser sends a frame when it
# paints, so a gap is reported instead of sampled, and costs no round trip to find. That is why it
# is so much shorter than the window above, which has to catch a document twice looking the same.
SETTLE_PAINT = float(os.environ.get("JEV_SETTLE_PAINT", "0.05"))

# How long to watch the screen before handing the page to the document.
SETTLE_WATCH = float(os.environ.get("JEV_SETTLE_WATCH", "0.3"))

# How long a page may keep changing under a decision before the decision is taken anyway. A clock,
# a countdown or a progress figure never holds still, so a check that the page is unchanged can
# never pass on one, and waiting for it to pass is waiting forever. Past this, the page counts as
# restless instead of arriving, and is acted on as it is.
RESTLESS = float(os.environ.get("JEV_RESTLESS", "1.5"))
SCREEN = {"format": "jpeg", "quality": 30, "maxWidth": 160, "maxHeight": 120, "everyNthFrame": 1}

# How long an arriving page may keep fetching, or keep showing a loading state, before it is read
# anyway. A single-page app paints its shell, holds still and is operable well before the list it
# was opened for comes back, so stillness alone hands the decision a page without its content.
SETTLE_BUSY = float(os.environ.get("JEV_SETTLE_BUSY", "1.5" if REMOTE else "8"))
# The kinds of request a page draws from. Documents, scripts and images are covered by the load
# event and by paint; beacons and pings draw nothing.
FETCHES = {"XHR", "Fetch"}
# A fetch open longer than this is a stream or a long poll, not content on its way.
LONG_REQUEST = float(os.environ.get("JEV_LONG_REQUEST", "10"))
# How often a page with nothing operable of its own is checked for controls inside its frames.
# Reading the frames costs a listing and a read per frame, so it is not asked on every poll.
FRAME_CHECK = 0.5
VIEWPORT = (int(os.environ.get("JEV_VIEWPORT_WIDTH", "1120")), int(os.environ.get("JEV_VIEWPORT_HEIGHT", "780")))

# How many of each kind of fault to keep. A page that fails one request per image would otherwise
# report its whole gallery.
FAULTS_KEPT = 25

# How much of a query's answer to carry back. A caller asking a page a question wants what it
# said, not the page; anything larger is a document being returned a field at a time.
QUERY_LENGTH = int(os.environ.get("JEV_QUERY_LENGTH", "2048"))

# What the page is, independent of anything a run did to it. A bare image answers every goal with
# nothing on screen, which reads as a page that failed rather than a page that is an image.
IDENTITY = """(() => {
  const heading = document.querySelector('h1');
  return {title: document.title || null,
          h1: heading ? (heading.innerText || '').trim().slice(0, 300) : null};
})()"""

# The page keeps its own record of when it last changed, so how long it has been quiet is a
# question instead of a wait. Polling can only establish stillness it watched: a page quiet for two
# seconds before the first look would otherwise be watched for a whole further window.
QUIET = """(() => {
  const w = window;
  if (!w.__jevQuiet) {
    w.__jevQuiet = {last: performance.now()};
    const touched = () => { w.__jevQuiet.last = performance.now(); };
    new MutationObserver(touched).observe(document, {
      subtree: true, childList: true, attributes: true, characterData: true});
    addEventListener('scroll', touched, {passive: true, capture: true});
    return 0;
  }
  return performance.now() - w.__jevQuiet.last;
})()"""


# Waiting for the document to hold still, inside the page. Polling costs a round trip per look,
# which is nothing beside a local pipe and most of a remote wait: a page that never holds still
# spent hundreds of calls establishing it. The page can watch itself and answer once.
STILL = """((still, timeout) => new Promise(resolve => {
  const began = performance.now();
  const look = () => {
    const quiet = """ + QUIET + """;
    const waited = performance.now() - began;
    if (quiet >= still || waited >= timeout) {
      return resolve({still: quiet >= still, quiet: Math.round(quiet), waited: Math.round(waited)});
    }
    setTimeout(look, Math.max(10, Math.min(50, still - quiet)));
  };
  look();
}))(%s, %s)"""


class StalePage(ValueError):
    """A decision no longer refers to the observed page."""


class Covered(StalePage):
    """The chosen control is there but cannot be reached: something sits over it, or it never
    appeared under the pointer. Nothing was sent. Choosing it again finds it covered again, so the
    agent counts these and stops offering a control that keeps being refused."""

    def __init__(self, message, action=None):
        super().__init__(message)
        self.action = action or {}


# Find the observed node and bring it to where it can be used. A control below the fold of a
# scrolling panel, or of the page, is scrolled into view here rather than by a separate step, so
# offering it costs the model nothing. Visibility is checked without opacity, because a control
# revealed on hover is transparent until the pointer reaches it; that is settled after the hover.
LOCATE = """(async action => {
  const e=window.__jevFast?.nodes.get(action.node);
  const across=(n,s)=>{
    for(;n;n=n.parentElement||n.getRootNode()?.host) if(n.matches?.(s)) return true;
    return false;
  };
  if (!e?.isConnected || e.matches(':disabled') || across(e,'[aria-disabled="true"],[inert],[aria-hidden="true"]') ||
      !e.checkVisibility({checkVisibilityCSS:true})) return null;
  if (action.kind==='fill' && (e.readOnly || e.getAttribute('aria-readonly')==='true')) return null;
  let r=e.getBoundingClientRect();
  // Inside the window and inside every scrolling ancestor, which is where it is actually drawn.
  const inside=()=>{
    const x=r.x+r.width/2, y=r.y+r.height/2;
    if (!r.width || !r.height || x<0 || y<0 || x>=innerWidth || y>=innerHeight) return false;
    for (let n=e.parentElement; n && n!==document.body; n=n.parentElement) {
      const style=getComputedStyle(n);
      if (!/(auto|scroll|overlay|hidden)/.test(style.overflowY+style.overflowX)) continue;
      const p=n.getBoundingClientRect();
      if (x<p.left || x>p.right || y<p.top || y>p.bottom) return false;
    }
    return true;
  };
  if (!inside()) {
    e.scrollIntoView({block:'center',inline:'center'});
    await new Promise(done=>requestAnimationFrame(()=>requestAnimationFrame(done)));
    r=e.getBoundingClientRect();
    if (!inside()) return null;
  }
  return {x:r.x+r.width/2, y:r.y+r.height/2,
          transparent:!e.checkVisibility({checkOpacity:true,checkVisibilityCSS:true})};
})(%s)"""

# Whether the pointer at (x, y) lands on the observed node, looking through shadow roots: the
# document reports a shadow host for anything inside it, which would read every injected control
# as covered. A transparent control is waited for first, since it is fading in under the hover.
REACHES = """(async (action, x, y) => {
  const e=window.__jevFast?.nodes.get(action.node);
  if (!e?.isConnected) return 'gone';
  for (let t=0; t<12 && !e.checkVisibility({checkOpacity:true,checkVisibilityCSS:true}); t++)
    await new Promise(done=>setTimeout(done,50));
  if (!e.checkVisibility({checkOpacity:true,checkVisibilityCSS:true})) return 'transparent';
  let hit=document.elementFromPoint(x,y);
  while (hit?.shadowRoot) {
    const inner=hit.shadowRoot.elementFromPoint(x,y);
    if (!inner || inner===hit) break;
    hit=inner;
  }
  for (let n=hit; n; n=n.parentNode || n.host) if (n===e) return 'ok';
  return 'covered';
})(%s, %s, %s)"""

# How far the pointer travels in a drag, and in how many moves. A pointer-driven drag library
# starts dragging only past an activation distance, and a native drag starts only once the button
# has moved while held; the moves are what both of them are waiting for.
DRAG_STEPS = 12
DRAG_PAUSE = 0.03


class DaemonBusy(RuntimeError):
    """Another browser is already using this daemon."""


# One browser at a time per daemon. The daemon keeps a single event buffer for every caller and
# empties all of it on each drain, so a second browser reading events takes the first one's:
# console errors and failed requests would be attributed to whichever run drained last. Nothing
# in the protocol prevents that, so it is refused here.
IN_USE = threading.Lock()

# The lock above holds within one process and no further, and every process on the machine reaches
# the same daemon by name (BU_NAME, "default" unless set). Two runs started side by side shared one
# event buffer and each took the other's events without either knowing. A lock on a file named for
# the daemon holds across processes, and the operating system lets go of it when its holder dies,
# so a run that was killed never leaves the daemon claimed.
LOCKS = Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache") / "jev"

# Set when JEV_DAEMON_PER_RUN gave this process a daemon of its own (see __init__), which nothing
# else will use once this process is gone, so it is stopped on the way out.
PER_RUN = os.environ.get("JEV_DAEMON_PER_RUN") == "1"


def claim_daemon(name=None):
    """An exclusive hold on the daemon across processes, or None when another process has it."""
    LOCKS.mkdir(parents=True, exist_ok=True)
    held = open(LOCKS / f"{name or DAEMON}.lock", "a+")
    try:
        if sys.platform == "win32":
            import msvcrt

            held.seek(0)
            msvcrt.locking(held.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl

            fcntl.flock(held.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        held.close()
        return None
    return held


def let_go(held):
    if sys.platform == "win32":
        import msvcrt

        try:
            held.seek(0)
            msvcrt.locking(held.fileno(), msvcrt.LK_UNLCK, 1)
        except OSError:
            pass
    held.close()


def stop_own_daemon():
    """Stop the daemon this process was given. Best effort: the process is leaving either way."""
    try:
        restart_daemon(DAEMON)
    except Exception:
        pass


def ensure_own_daemon():
    ensure_daemon()
    if PER_RUN:
        atexit.unregister(stop_own_daemon)  # once, however many browsers this process opens
        atexit.register(stop_own_daemon)


class Browser:
    def __init__(self, url, context=None, target=None):
        if not IN_USE.acquire(blocking=False):
            raise DaemonBusy(
                "This daemon already has a browser. Its events are drained as one buffer, so a "
                "second browser would take the first one's; run one browser per daemon."
            )
        self.daemon_lock = claim_daemon()
        if self.daemon_lock is None:
            IN_USE.release()
            raise DaemonBusy(
                f"Another process is using the {DAEMON!r} daemon, and its events are drained as one "
                "buffer. Give each run its own daemon with JEV_DAEMON_PER_RUN=1 or a distinct BU_NAME."
            )
        self.holds_daemon = True
        try:
            if target:
                self.borrow(target)
            else:
                self.open(url, context, watching=os.environ.get("JEV_FOREGROUND") == "1")
        except BaseException:
            self.release()
            raise

    @classmethod
    def attach(cls, target):
        """A browser on a page that is already open, which stays the caller's.

        A caller that has already driven a tab to where a leg starts would otherwise pay for a new
        tab and a reload of the same page, which was most of a short leg's time. The page is used
        as it stands: its window, its size, its cookies and the address it is on.
        """
        return cls(None, target=target)

    def open(self, url, context, watching):
        ensure_own_daemon()
        # Background by default so a run does not steal the window. JEV_FOREGROUND=1 brings it to
        # the front instead, for watching a run live.
        # A run gets its own browser context, so the cookies and logins of the site before it do
        # not carry into this one. A remote endpoint serves one customer after another through the
        # same browser, where that would otherwise leak between them.
        self.context = context if context is not None else self.own_context()
        made = {"browserContextId": self.context} if self.context else {}
        self.target = cdp("Target.createTarget", url="about:blank", background=not watching, **made)["targetId"]
        self.session = cdp("Target.attachToTarget", targetId=self.target, flatten=True)["sessionId"]
        width, height = VIEWPORT
        self.call(
            "Emulation.setDeviceMetricsOverride", width=width, height=height, deviceScaleFactor=1, mobile=False
        )
        # Keep rAF/menus rendering in an owned background tab, without activating the user's Chrome tab.
        self.call("Emulation.setFocusEmulationEnabled", enabled=True)
        self.faults = {}
        self.watch_faults()
        if watching:
            cdp("Target.activateTarget", targetId=self.target)
        self.navigate(url)

    def borrow(self, target):
        """Attach to the caller's page without changing what it is.

        No target or context is made, so there is nothing of jev's to close afterwards, and no
        size is imposed: the page keeps the layout the caller sees, and frames are judged against
        the window it actually has. Focus emulation is still set, because a background tab stops
        drawing menus and animation frames otherwise; it belongs to this session and ends with it.
        """
        ensure_own_daemon()
        self.borrowed = True
        self.context = None
        self.target = target
        self.session = cdp("Target.attachToTarget", targetId=target, flatten=True)["sessionId"]
        self.call("Emulation.setFocusEmulationEnabled", enabled=True)
        try:
            width, height = self.evaluate("[innerWidth, innerHeight]")
            self.viewport = (int(width), int(height))
        except (TypeError, ValueError, TimeoutError, StalePage):
            pass
        self.faults = {}
        self.watch_faults()

    def release(self):
        """Give the daemon back. Idempotent, so closing twice is not an error."""
        if getattr(self, "holds_daemon", False):
            self.holds_daemon = False
            if getattr(self, "daemon_lock", None) is not None:
                let_go(self.daemon_lock)
                self.daemon_lock = None
            IN_USE.release()

    def watch_faults(self):
        """Ask the page to report its console, its exceptions and its network.

        These arrive as events, so nothing is spent per step to collect them: the wait already
        drains the buffer, and what is not a screencast frame is read here instead of dropped.
        """
        for domain in ("Log", "Runtime", "Network", "Page"):
            try:
                self.call(f"{domain}.enable")
            except (RuntimeError, KeyError):
                pass

    def note_fault(self, event):
        """Record one event if it reports something wrong with the page."""
        faults = getattr(self, "faults", None)
        if faults is None:
            faults = self.faults = {}
        method, params = event.get("method"), event.get("params") or {}
        if method == "Log.entryAdded":
            entry = params.get("entry") or {}
            if entry.get("level") in {"error", "warning"}:
                faults.setdefault("console", []).append({
                    "level": entry["level"], "text": (entry.get("text") or "")[:200], "url": entry.get("url"),
                })
        elif method == "Runtime.exceptionThrown":
            details = params.get("exceptionDetails") or {}
            thrown = (details.get("exception") or {}).get("description") or details.get("text") or ""
            faults.setdefault("exceptions", []).append({"text": thrown[:200], "url": details.get("url")})
        elif method == "Network.responseReceived":
            response = params.get("response") or {}
            if params.get("type") == "Document" and response.get("url"):
                faults.setdefault("document_type", (response.get("mimeType") or "").split(";")[0] or None)
            if (response.get("status") or 0) >= 400:
                faults.setdefault("requests", []).append({
                    "url": (response.get("url") or "")[:200], "status": response["status"],
                })
            if params.get("type") == "Document" and response.get("url"):
                # The first document response is the page's own status; later ones are its frames.
                faults.setdefault("document_status", response["status"])
        elif method == "Network.loadingFailed":
            self.requests_open().pop(params.get("requestId"), None)
            faults.setdefault("requests", []).append({
                "url": None, "status": None, "failed": (params.get("errorText") or "")[:80],
            })
        if method == "Network.requestWillBeSent" and params.get("type") in FETCHES:
            self.requests_open()[params.get("requestId")] = time.monotonic()
        elif method == "Network.loadingFinished":
            self.requests_open().pop(params.get("requestId"), None)

    def requests_open(self):
        """Fetches the page has started and not finished, by request id, with when each began."""
        found = getattr(self, "open_requests", None)
        if found is None:
            found = self.open_requests = {}
        return found

    def pending(self):
        """How many of the page's own fetches are still outstanding.

        A request that has been open longer than LONG_REQUEST is a stream, a long poll or one that
        hung, and the page is not waiting on it to draw anything, so it stops counting.
        """
        now = time.monotonic()
        return sum(1 for began in self.requests_open().values() if now - began < LONG_REQUEST)

    def own_context(self):
        """A fresh browser context, or None where the browser will not make one."""
        if os.environ.get("JEV_BROWSER_CONTEXT") == "0":
            return None
        try:
            return cdp("Target.createBrowserContext", disposeOnDetach=False)["browserContextId"]
        except (RuntimeError, KeyError):
            return None

    def call(self, method, **params):
        return cdp(method, session_id=self.session, **params)

    def navigate(self, url):
        """Go to `url` and wait for the document to finish loading."""
        self.call("Page.navigate", url=url)
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            try:
                if self.evaluate("document.readyState") == "complete":
                    return
            except (TimeoutError, StalePage):
                # An app booting its bundle can hold the main thread past the transport's patience,
                # and a document swapped mid-question answers nothing. Both mean not loaded yet;
                # raising here ended runs before their first look, with nothing reported.
                pass
            time.sleep(0.02)

    def frames(self):
        """One session per cross-origin iframe worth reading, with where each sits on the page.

        A cross-origin iframe runs out of process, so it is absent from the page's frame tree and
        unreachable from the page's JavaScript: the reader running in the page sees nothing of it.
        It is its own CDP target, and the same reader works inside it. The offset is needed because
        the Input domain exists only on the page target, so a click worked out inside a frame has to
        be dispatched in the page's coordinates.
        """
        found = []
        attached = getattr(self, "attached", None)
        if attached is None:
            attached = self.attached = {}
        try:
            targets = cdp("Target.getTargets")["targetInfos"]
        except (RuntimeError, KeyError):
            return found
        for info in targets:
            if info.get("type") != "iframe" or info.get("url", "").startswith("about:"):
                continue
            try:
                # Attaching is once per frame, not once per observation: a session outlives the
                # look that found it, and re-attaching would spend calls to learn what is known.
                session = attached.get(info["targetId"])
                if session is None:
                    session = cdp("Target.attachToTarget", targetId=info["targetId"], flatten=True)["sessionId"]
                    attached[info["targetId"]] = session
                owner = self.call("DOM.getFrameOwner", frameId=info["targetId"])
                box = self.call("DOM.getBoxModel", backendNodeId=owner["backendNodeId"])["model"]["content"]
            except (RuntimeError, KeyError):
                continue  # not ours, or gone between listing and attaching
            # A frame too small or too far off-screen to be used is not worth reading. Tracking
            # pixels and advertising slots are most of the frames on a page and none of its work,
            # and each one read costs a call whose answer can never be acted on.
            width, height = box[2] - box[0], box[7] - box[1]
            window = getattr(self, "viewport", VIEWPORT)
            if width < USABLE_FRAME or height < USABLE_FRAME or box[0] > window[0] or box[1] > window[1]:
                continue
            found.append({"session": session, "offset": (box[0], box[1])})
        return found

    def evaluate(self, expression, session=None):
        response = cdp(
            "Runtime.evaluate", session_id=session or self.session, expression=expression, returnByValue=True
        )
        if response.get("exceptionDetails"):
            raise StalePage("Document changed during evaluation")
        return response.get("result", {}).get("value")

    def collect(self):
        """Take what the page has reported since the last look.

        Faults arrive as events whether or not a wait needed the screen, and an event left in the
        buffer is one nobody reads. Draining on every observation keeps collection independent of
        which wait ran.
        """
        try:
            self.painted()
        except (RuntimeError, OSError):
            pass

    def observe(self, screenshot=True, frames=True):
        if getattr(self, "after_input", None):
            action, self.after_input = self.after_input, None
            # This is read-only and happens after execution was logged, even if navigation interrupts it.
            try:
                self.call(
                    "Runtime.evaluate",
                    expression="""(action => new Promise(resolve => {
                      const field=window.__jevFast?.nodes.get(action.node);
                      const autocomplete=action.kind==='fill' && field?.getAttribute('role')==='combobox';
                      let frames=0, stopped=false;
                      const finish=()=>{stopped=true;resolve()};
                      setTimeout(finish,autocomplete ? 200 : 50);
                      const ready=()=>{
                        if (stopped) return;
                        const ids=(field?.getAttribute('aria-controls')||field?.getAttribute('aria-owns')||'')
                          .split(/\\s+/).filter(Boolean);
                        const roots=ids.length ? ids.map(id=>document.getElementById(id)).filter(Boolean) : [document];
                        const options=roots.flatMap(root=>[...root.querySelectorAll('[role="option"]')]);
                        if (++frames>=2 && (!autocomplete || options.some(e=>{
                          const r=e.getBoundingClientRect();
                          return r.width && r.height && r.bottom>0 && r.top<innerHeight &&
                            e.checkVisibility({checkOpacity:true,checkVisibilityCSS:true});
                        }))) finish();
                        else requestAnimationFrame(ready);
                      };
                      requestAnimationFrame(ready);
                    }))(""" + json.dumps(action) + ")",
                    awaitPromise=True,
                    returnByValue=True,
                )
            except RuntimeError:
                pass
        self.collect()
        for attempt in range(10):
            try:
                return self.read(screenshot, frames)
            except StalePage:
                if attempt == 9:
                    raise
                time.sleep(0.02)
            except TimeoutError:
                # A page busy booting can hold a read past the transport's patience. Reading has no
                # effect on the page, so it is asked again, as a document caught mid-swap is.
                if attempt >= 3:
                    raise
                time.sleep(0.25)
        raise StalePage("Page did not settle")

    def worth_reading_frames(self, page):
        """Whether to spend calls looking inside this page's frames.

        Reading them costs a listing, two measurements and a read per frame on every observation -
        about a third more protocol calls on a page whose own controls were all readable anyway.
        JEV_FRAMES selects where to spend that: "1" for every page, or a comma-separated list of
        strings matched against the address, so a surface that proxies its content into a frame can
        be named without slowing everything else down.

        A page offering nothing is read whatever the setting says, since no list can anticipate it
        and the alternative is a step that can only answer BLOCKED.
        """
        setting = os.environ.get("JEV_FRAMES", "").strip()
        if setting == "1":
            return True
        wanted = [part.strip() for part in setting.split(",") if part.strip()]
        if wanted and any(part in (page.get("url") or "") for part in wanted):
            return True
        return not operable(page)

    def read(self, screenshot, frames=True):
        """One observation of the page, and of its frames when those are worth reading.

        Ids, guards and page keys are namespaced by frame whether or not any frame is read, because
        every frame numbers its own nodes from one and a decision has to say which frame it means.
        Numbering only when a frame happens to be present would make the same element answer to two
        different names depending on the page, and a guard looked up under the wrong one reads as a
        page that has moved.
        """
        merged = None
        for index, frame in enumerate(self.frames_to_read(screenshot, frames)):
            state = frame.pop("state", None) or browser_operation(
                {"operation": "observe", "session": frame["session"], "screenshot": False}
            )
            for action in state["actions"]:
                action["frame"] = index
                action["session"] = frame["session"]
                action["offset"] = frame["offset"]
                action["id"] = f"{index}:{action['id']}"
            guards = {f"{index}:{node}": guard for node, guard in state.get("guards", {}).items()}
            if merged is None:
                merged = state
                merged["guards"] = guards
                merged["page_keys"] = {str(index): state.get("page_key")}
                continue
            merged["actions"].extend(state["actions"])
            merged["text"] = (merged.get("text") or "") + "\n" + (state.get("text") or "")
            merged["guards"].update(guards)
            merged["page_keys"][str(index)] = state.get("page_key")
        merged["fingerprint"] = fingerprint(merged)
        return merged

    def frames_to_read(self, screenshot, frames=True):
        """The page, plus its frames when they are worth the calls. The page's own observation is
        carried along so it is never taken twice.

        Finding the frames costs a listing and two measurements each, and their positions move
        when the page scrolls, so the answer cannot be kept. `frames` is False for the readings
        taken while waiting, which nothing is decided from: a wait only needs to know whether the
        page has stopped, and the reading a decision is taken from is where frames are worth
        finding.
        """
        page = browser_operation({"operation": "observe", "session": self.session, "screenshot": screenshot})
        first = {"session": self.session, "offset": (0.0, 0.0), "state": page}
        if not frames or not self.worth_reading_frames(page):
            return [first]
        return [first] + self.frames()

    def fresh(self, page, action=None):
        if action is not None and action["kind"] in {"click", "select", "drag"}:
            node = action["node"]
            if type(node) is not int:
                return False
            frame = action.get("frame", 0)
            current = self.evaluate(
                "(() => { const c=window.__jevFast; "
                f"return c ? [c.pageKey(),c.guard(c.nodes.get({node}))] : null; }})()",
                session=action.get("session"),
            )
            keys = page.get("page_keys") or {"0": page.get("page_key")}
            return current == [keys.get(str(frame)), page["guards"].get(f"{frame}:{node}")]
        return self.evaluate(MARKER) == page["marker"]

    def settle(self, screenshot=False, timeout=None, previous=None, stabilise=False, glimpse=None):
        """Observe until the page can be acted on, has stopped moving and has stopped arriving.

        Stillness is `hold_still`. Arriving is a page still fetching, or still saying it is
        loading, after it has gone still: the shell of a single-page app is both operable and still
        while the list it was opened for is on its way. That second wait follows the same
        transitions the first one does, so typing into a field never waits on a fetch.
        """
        started = time.monotonic()
        page = self.hold_still(screenshot, timeout, previous, stabilise, glimpse)
        if stabilise or previous is None or unsettling(previous, page):
            page = self.arrive(page, screenshot, started)
        return page

    def arrive(self, page, screenshot, started):
        """The page once its fetches have come back and it has stopped saying it is loading.

        Bounded by SETTLE_BUSY from the start of the settle. A loading state that outlasts that is
        a permanent one - a progress bar, a spinner in a corner - and is remembered as this page's
        floor, so the next wait on the same page waits only for loading beyond it.
        """
        floors = getattr(self, "busy_floor", None)
        if floors is None:
            floors = self.busy_floor = {}
        url = page.get("url")
        floor = floors.get(url, 0)
        deadline = started + SETTLE_BUSY
        waited = False
        while (page.get("busy", 0) > floor or self.pending()) and time.monotonic() < deadline:
            waited = True
            time.sleep(SETTLE_POLL * 2)
            page = self.observe(screenshot=False, frames=False)
        if page.get("busy", 0) > floor:
            floors[url] = page["busy"]
        if self.pending():
            # Still open at the deadline, so not content on its way. Forgotten, so the next wait is
            # not spent on the same request.
            self.requests_open().clear()
        if not waited:
            return self.observe(screenshot=screenshot)
        # What arrived is drawn in more than one pass, so it is given a moment to hold still.
        quiet_by = min(deadline + SETTLE_STILL, time.monotonic() + 1.0)
        while self.quiet_ms() < SETTLE_STILL * 1000 and time.monotonic() < quiet_by:
            time.sleep(SETTLE_POLL)
        return self.observe(screenshot=screenshot)

    def usable(self, page):
        """Whether a reading taken without its frames can be acted on.

        The readings taken while waiting skip frames to stay cheap, so a page whose controls all
        live in a frame - a site proxied into one, with a toolbar injected into it - reads as having
        none, and every wait on it ran to its deadline. Its frames are looked into instead, at most
        every FRAME_CHECK, and a page found to work that way is remembered as one that does.
        """
        if operable(page):
            return True
        url = page.get("url")
        if getattr(self, "framed_url", None) == url:
            return True
        now = time.monotonic()
        if now - getattr(self, "frame_checked", 0.0) < FRAME_CHECK:
            return False
        self.frame_checked = now
        try:
            whole = self.read(False, frames=True)
        except StalePage:
            return False
        if operable(whole):
            self.framed_url = url
            return True
        return False

    def hold_still(self, screenshot=False, timeout=None, previous=None, stabilise=False, glimpse=None):
        """Observe until the page can be acted on, then wait for it to stop moving.

        Waiting for an operable page removes a wasted call: an empty action space can only be
        answered BLOCKED. The wait for stillness follows only a transition that tends to stream
        content in - a navigation, a menu or dialog opening, or a large change in what is on offer.
        A fingerprint change is not a trigger, because it moves when a field is typed into or the
        page is scrolled, neither of which means the page is arriving.

        Stillness is established from the screen where the browser will report it, and from the
        document where it will not; `still_screen` says which applies. `glimpse` is offered every
        reading taken along the way, so a caller can decide about a page before the wait ends.
        """
        deadline = time.monotonic() + (SETTLE_TIMEOUT if timeout is None else timeout)
        page = self.observe(screenshot=screenshot, frames=False)
        if glimpse:
            # A reading taken while waiting is enough to decide from. The wait governs when to act,
            # not when to start deciding.
            glimpse(page)
        while not self.usable(page) and time.monotonic() < deadline:
            # Whether anything can be acted on is a yes or no, not a window to wait out, so it is
            # asked as often as it is cheap to ask.
            time.sleep(SETTLE_POLL)
            page = self.observe(screenshot=screenshot, frames=False)
            if glimpse:
                glimpse(page)
        if not (stabilise or previous is None or unsettling(previous, page)):
            return self.observe(screenshot=screenshot)
        # If it has already been quiet for long enough, it is still, and watching would only
        # confirm what it has just said.
        if self.quiet_ms() >= SETTLE_STILL * 1000:
            return self.observe(screenshot=screenshot)
        watched = self.still_screen(screenshot, deadline, glimpse)
        if watched is not None:
            return self.observe(screenshot=screenshot)
        if REMOTE:
            # The page reports its own stillness, so a wait of any length is one round trip. The
            # deadline governs, as it does for polling.
            while time.monotonic() < deadline:
                held = self.held_still(deadline - time.monotonic())
                page = self.observe(screenshot=screenshot, frames=False)
                if glimpse:
                    glimpse(page)
                if self.usable(page) and (held is None or held["still"]):
                    return self.observe(screenshot=screenshot)
            return self.observe(screenshot=screenshot)
        still_since = None
        while time.monotonic() < deadline:
            time.sleep(SETTLE_POLL)
            following = self.observe(screenshot=screenshot, frames=False)
            if glimpse:
                glimpse(following)
            # Hydration has quiet moments, so a page counts as still once it has been unchanged
            # for a while, not once two reads happen to agree.
            still_since = (still_since or time.monotonic()) if following["fingerprint"] == page["fingerprint"] else None
            page = following
            if self.usable(page) and (
                (still_since and time.monotonic() - still_since >= SETTLE_STILL)
                or self.quiet_ms() >= SETTLE_STILL * 1000
            ):
                return self.observe(screenshot=screenshot)
        return self.observe(screenshot=screenshot)

    def watching_paint(self):
        """Ask the browser to report its painting, once per page. False when it will not.

        Never on a remote endpoint: a frame is acknowledged one round trip at a time, and a page
        that paints continuously spends more calls on acknowledging frames than on reading itself.
        """
        if REMOTE:
            return False
        watching = getattr(self, "screencast", None)
        if watching is None:
            try:
                self.call("Page.startScreencast", **SCREEN)
                watching = True
            except (RuntimeError, KeyError) as refusal:
                # A page left reporting by an earlier run is already doing what was asked for.
                watching = "already active" in str(refusal)
            self.screencast = watching
        return watching

    def painted(self):
        """When the screen last painted, or None if it has not yet.

        Draining empties the daemon's whole event buffer, which has no per-method filter. Anything
        else waiting in it is discarded, so a caller reading events of its own will see none of
        those produced while a wait is running.
        """
        try:
            events = drain_events()
        except (RuntimeError, OSError):
            events = []
        for event in events:
            if event.get("method") != "Page.screencastFrame":
                # The buffer holds one daemon's whole stream, and this browser owns it, so what is
                # not a frame is this page reporting a fault.
                self.note_fault(event)
                continue
            self.last_paint = time.monotonic()
            try:
                self.call("Page.screencastFrameAck", sessionId=event["params"]["sessionId"])
            except (RuntimeError, KeyError):
                pass
        return getattr(self, "last_paint", None)

    def still_screen(self, screenshot, deadline, glimpse=None):
        """The page once the screen has stopped painting, or None if the screen cannot say so.

        None means the caller falls back to its slower answer, which always works: a page that
        paints without pause, or one whose painting is never reported, is settled by the document.
        """
        if not self.watching_paint():
            return None
        self.painted()
        began = time.monotonic()
        give_up = min(deadline, began + SETTLE_WATCH)
        while time.monotonic() < give_up:
            painted = self.painted()
            if painted is None:
                # Nothing has been reported, so nothing will be. Recorded on the browser, so the
                # next wait does not spend its whole budget learning the same thing.
                if time.monotonic() - began >= SETTLE_PAINT * 2:
                    self.screencast = False
                    return None
            elif time.monotonic() - painted >= SETTLE_PAINT:
                settled = self.observe(screenshot=screenshot, frames=False)
                if glimpse:
                    glimpse(settled)
                return settled if self.usable(settled) else None
            time.sleep(0.01)
        return None

    def text_of(self, action):
        """The text inside an observed element, read from the node the reader kept.

        Read from the page rather than described by a model: the caller is checking what the page
        shows, and a description would be the thing under test writing its own evidence.
        """
        node = action.get("node")
        if type(node) is not int:
            return ""
        try:
            return self.evaluate(
                f"(() => {{ const e = window.__jevFast?.nodes.get({node});"
                " return e ? (e.innerText || e.value || '') : ''; })()",
                session=action.get("session"),
            ) or ""
        except (StalePage, RuntimeError, KeyError):
            return ""

    def held_still(self, timeout):
        """Wait in the page until it has been quiet for SETTLE_STILL, or until `timeout` passes.

        One round trip covers the whole wait. None means the page could not answer - it navigated
        out from under the question, or the connection refused it - and the caller looks again.
        """
        if timeout <= 0:
            return None
        try:
            response = cdp(
                "Runtime.evaluate",
                session_id=self.session,
                expression=STILL % (round(SETTLE_STILL * 1000), round(timeout * 1000)),
                awaitPromise=True,
                returnByValue=True,
                # The call is held open for the whole wait, which outlasts the transport's own
                # patience; without this a wait longer than five seconds fails as a dead daemon.
                _response_timeout=timeout + 2,
            )
        except (RuntimeError, KeyError, OSError):
            return None
        if response.get("exceptionDetails"):
            return None
        answer = response.get("result", {}).get("value")
        return answer if isinstance(answer, dict) else None

    def ask(self, expression, cap=None):
        """Run a caller's expression in the main frame and return what it answered.

        A promise is awaited, so a query may look at something the page has yet to finish. The
        answer is reported as a value or as the text of what it raised; either way the run carries
        on, because a question that fails is an answer about the page.
        """
        cap = QUERY_LENGTH if cap is None else cap
        try:
            response = cdp(
                "Runtime.evaluate",
                session_id=self.session,
                expression=expression,
                awaitPromise=True,
                returnByValue=True,
                _response_timeout=30,
            )
        except (RuntimeError, OSError, KeyError) as refusal:
            return {"exception": str(refusal)[:cap]}
        details = response.get("exceptionDetails")
        if details:
            thrown = (details.get("exception") or {}).get("description") or details.get("text") or "failed"
            if "Illegal return statement" in str(thrown) and not expression.startswith("(async () => {"):
                # Written as a function body, which is how a question with several statements comes
                # out naturally. Run as one rather than failed for its form.
                return self.ask("(async () => {\n" + expression + "\n})()", cap)
            return {"exception": str(thrown)[:cap]}
        value = response.get("result", {}).get("value")
        if isinstance(value, str):
            return {"value": value[:cap]}
        if value is None or isinstance(value, (int, float, bool)):
            return {"value": value}
        rendered = json.dumps(value, default=str)
        return {"value": value} if len(rendered) <= cap else {"value": rendered[:cap]}

    def identity(self):
        """What the page is: its title and first heading, whatever a run made of it."""
        try:
            found = self.evaluate(IDENTITY)
        except (StalePage, RuntimeError):
            return {}
        return found if isinstance(found, dict) else {}

    def quiet_ms(self):
        """How long the page reports going without changing. Zero when it cannot report."""
        try:
            answer = self.evaluate(QUIET)
        except (StalePage, RuntimeError):
            return 0.0
        return float(answer) if isinstance(answer, (int, float)) else 0.0

    def act(self, action, page, text=None, insist=False, drop=None):
        if not insist and not self.fresh(page, action):
            raise StalePage("Page changed since this decision. Observe again.")
        if action["kind"] == "wait":
            time.sleep(0.1)
        result = browser_operation({
            "operation": "act",
            "session": action.get("session", self.session),
            "dispatch_session": self.session,
            "offset": action.get("offset", (0.0, 0.0)),
            "action": action,
            "text": text,
            "drop": drop,
        })
        if action["kind"] == "drag":
            self.drag(*result.pop("drag"))
        self.after_input = action if action["kind"] not in {"wait", "drag"} else None
        return result

    def drag(self, start, end):
        """Press at `start`, travel to `end` and let go, as either kind of drag needs.

        A pointer-driven library (dnd-kit, react-beautiful-dnd) follows the moves. A native drag
        is begun by the browser once the held pointer moves, and with drags intercepted it is
        handed back here instead of being run by the operating system, which is the only way a
        native drag can be finished from the protocol: the drop is then dispatched to `end`.
        Logged as executed by the caller before anything is observed, like any other input.
        """
        (sx, sy), (ex, ey) = start, end
        mouse = lambda kind, x, y, **more: self.call("Input.dispatchMouseEvent", type=kind, x=x, y=y, **more)  # noqa: E731
        try:
            self.call("Input.setInterceptDrags", enabled=True)
        except (RuntimeError, KeyError):
            pass
        intercepted = None
        try:
            mouse("mousePressed", sx, sy, button="left", buttons=1, clickCount=1)
            for step in range(1, DRAG_STEPS + 1):
                x, y = sx + (ex - sx) * step / DRAG_STEPS, sy + (ey - sy) * step / DRAG_STEPS
                mouse("mouseMoved", x, y, button="left", buttons=1)
                time.sleep(DRAG_PAUSE)
                intercepted = intercepted or self.drag_started()
                if intercepted:
                    break
            if intercepted:
                for kind in ("dragEnter", "dragOver", "drop"):
                    self.call("Input.dispatchDragEvent", type=kind, x=ex, y=ey, data=intercepted)
                    time.sleep(DRAG_PAUSE)
            mouse("mouseReleased", ex, ey, button="left", buttons=0, clickCount=1)
        finally:
            try:
                self.call("Input.setInterceptDrags", enabled=False)
            except (RuntimeError, KeyError):
                pass

    def drag_started(self):
        """The data of a native drag the browser has handed back, if one has begun.

        Read from the same event buffer as everything else, so what else is in it is passed on
        rather than dropped.
        """
        try:
            events = drain_events()
        except (RuntimeError, OSError):
            return None
        found = None
        for event in events:
            method = event.get("method")
            if method == "Input.dragIntercepted":
                found = (event.get("params") or {}).get("data")
            elif method == "Page.screencastFrame":
                self.last_paint = time.monotonic()
                try:
                    self.call("Page.screencastFrameAck", sessionId=event["params"]["sessionId"])
                except (RuntimeError, KeyError):
                    pass
            else:
                self.note_fault(event)
        return found

    def close(self):
        if getattr(self, "screencast", False):
            # A page outlives the reader that started the report, and Chrome refuses a second
            # screencast on a target that already has one.
            try:
                self.call("Page.stopScreencast")
            except (RuntimeError, KeyError):
                pass
            self.screencast = None
        if getattr(self, "borrowed", False):
            # The page is the caller's. Detaching ends this session's emulation and leaves the tab
            # as it was found, wherever the run took it.
            if getattr(self, "session", None):
                try:
                    cdp("Target.detachFromTarget", sessionId=self.session)
                except (RuntimeError, KeyError):
                    pass
                self.session = None
            self.target = None
            self.release()
            return
        if os.environ.get("JEV_KEEP_OPEN") == "1":
            self.target = None  # leave the page up to be looked at
            self.release()
            return
        if self.target:
            cdp("Target.closeTarget", targetId=self.target)
            self.target = None
        if getattr(self, "context", None):
            # Disposing takes the context's cookies and storage with it.
            try:
                cdp("Target.disposeBrowserContext", browserContextId=self.context)
            except (RuntimeError, KeyError):
                pass
            self.context = None
        self.release()


def diagnosis(faults):
    """What went wrong on the page, deduplicated and capped."""
    seen, out = set(), {}
    for kind in ("console", "exceptions", "requests"):
        kept = []
        for fault in faults.get(kind) or []:
            key = (kind, fault.get("text"), fault.get("url"), fault.get("status"), fault.get("failed"))
            if key in seen:
                continue
            seen.add(key)
            kept.append(fault)
            if len(kept) == FAULTS_KEPT:
                break
        if kept:
            out[kind] = kept
    for named in ("document_status", "document_type"):
        if faults.get(named) is not None:
            out[named] = faults[named]
    return out


def expanded(page):
    return sum(1 for action in page["actions"] if str(action.get("expanded")).lower() == "true")


def unsettling(previous, current):
    """Whether the change between two observations is the kind that keeps arriving.

    Removing this and waiting only when a decision was already rejected was measured faster and
    less reliable: a run in three stopped part-way. Deciding on a page that is still arriving costs
    more than the wait it saves.
    """
    if previous["url"] != current["url"]:
        return True
    if expanded(current) > expanded(previous):
        return True
    before, after = len(previous["actions"]), len(current["actions"])
    return abs(after - before) >= max(3, before // 10)


def operable(page):
    """Whether anything on the page can be acted on. An unhydrated page still carries the WAIT
    control, so the presence of actions says nothing on its own."""
    return any(action.get("kind") in {"click", "fill", "select"} for action in page["actions"])


def fingerprint(state):
    content = {k: state[k] for k in ("url", "text", "actions", "scroll")}
    return hashlib.sha256(json.dumps(content, sort_keys=True).encode()).hexdigest()


def browser_operation(request):
    operation = request["operation"]
    session = request["session"]

    def call(method, **params):
        return cdp(method, session_id=session, **params)

    def evaluate(expression):
        result = call("Runtime.evaluate", expression=expression, returnByValue=True)
        if result.get("exceptionDetails"):
            if operation == "act" and request["action"]["kind"] == "select":
                raise RuntimeError("Dropdown execution was interrupted; inspect before retrying.")
            raise StalePage("Document changed during evaluation")
        return result.get("result", {}).get("value")

    if operation == "act":
        action = request["action"]
        kind = action["kind"]
        dispatch = request.get("dispatch_session", session)
        dx, dy = request.get("offset", (0.0, 0.0))

        def send(method, **params):
            # The Input domain lives on the page target only: a frame cannot dispatch its own
            # events, so the page does it, in page coordinates.
            return cdp(method, session_id=dispatch, **params)

        def awaited(expression):
            result = call("Runtime.evaluate", expression=expression, returnByValue=True, awaitPromise=True)
            if result.get("exceptionDetails"):
                raise StalePage("Document changed during evaluation")
            return result.get("result", {}).get("value")

        if kind == "scroll":
            send("Input.dispatchMouseEvent", type="mouseWheel", x=550, y=650, deltaX=0, deltaY=action["delta"])
        elif kind == "select":
            if type(action["node"]) is not int:
                raise ValueError("Invalid observed node")
            # Set in the page rather than through the pointer: a native dropdown opens a list the
            # page cannot see, and the change it makes is the same. Reachability is checked in the
            # same evaluation as the change, so an interruption can never be mistaken for one that
            # happened before anything was set.
            done = evaluate("""(action => {
              const e=window.__jevFast?.nodes.get(action.node);
              if (!e?.isConnected || e.tagName!=='SELECT' || e.disabled ||
                  !e.checkVisibility({checkOpacity:true,checkVisibilityCSS:true}) ||
                  ![...e.options].some(o=>o.value===action.value && !o.disabled && !o.closest('optgroup[disabled]')))
                return null;
              let r=e.getBoundingClientRect();
              const inside=()=>r.width && r.height && r.x+r.width/2>=0 && r.y+r.height/2>=0 &&
                r.x+r.width/2<innerWidth && r.y+r.height/2<innerHeight;
              if (!inside()) { e.scrollIntoView({block:'center'}); r=e.getBoundingClientRect(); }
              if (!inside()) return null;
              let hit=document.elementFromPoint(r.x+r.width/2,r.y+r.height/2);
              while (hit?.shadowRoot) {
                const inner=hit.shadowRoot.elementFromPoint(r.x+r.width/2,r.y+r.height/2);
                if (!inner || inner===hit) break;
                hit=inner;
              }
              let reached=false;
              for (let n=hit; n; n=n.parentNode || n.host) if (n===e) reached=true;
              if (!reached) return null;
              e.value=action.value;
              e.dispatchEvent(new Event('input',{bubbles:true}));
              e.dispatchEvent(new Event('change',{bubbles:true}));
              return true;
            })(""" + json.dumps(action) + ")")
            if done is None:
                raise RuntimeError("Dropdown execution was not confirmed; inspect before retrying.")
        elif kind != "wait":
            if type(action["node"]) is not int:
                raise ValueError("Invalid observed node")
            # Code-owned node IDs refer to actual observed elements, never model-generated selectors.
            target = awaited(LOCATE % json.dumps(action))
            if target is None:
                raise StalePage("Target changed or is out of reach. Observe again.")
            # The pointer arrives before it presses, as a person's does. A control revealed on hover
            # appears only then, and a page that tracks the pointer to place what a click makes
            # places it where the pointer last was.
            send("Input.dispatchMouseEvent", type="mouseMoved", x=target["x"] + dx, y=target["y"] + dy)
            reached = awaited(REACHES % (json.dumps(action), target["x"], target["y"]))
            if reached != "ok":
                raise Covered(f"Target is {reached}; nothing was sent. Observe again.", action)
            if kind == "drag":
                zone = request.get("drop")
                if not zone or zone.get("session", session) != action.get("session", session):
                    raise ValueError("A drag needs a drop zone in the same frame")
                end = awaited(LOCATE % json.dumps({**zone, "kind": "click"}))
                if end is None:
                    raise StalePage("Drop zone changed or is out of reach. Observe again.")
                return {"executed": action["id"], "drag": [
                    (target["x"] + dx, target["y"] + dy), (end["x"] + dx, end["y"] + dy),
                ]}
            if kind == "click" or kind == "fill":
                x, y = target["x"] + dx, target["y"] + dy
                for event in ("mousePressed", "mouseReleased"):
                    send("Input.dispatchMouseEvent", type=event, x=x, y=y, button="left", clickCount=1)
                if kind == "fill":
                    send(
                        "Input.dispatchKeyEvent",
                        type="keyDown",
                        key="a",
                        code="KeyA",
                        modifiers=4 if sys.platform == "darwin" else 2,
                        commands=["selectAll"],
                    )
                    send(
                        "Input.dispatchKeyEvent",
                        type="keyUp",
                        key="a",
                        code="KeyA",
                        modifiers=4 if sys.platform == "darwin" else 2,
                    )
                    send("Input.insertText", text=request["text"])
        return {"executed": action["id"]}

    info = evaluate(READ_STATE)
    if info is None:
        raise StalePage("Document is navigating")
    info["fingerprint"] = fingerprint(info)
    if request.get("screenshot", True):
        info["screenshot"] = call("Page.captureScreenshot", format="jpeg", quality=72)["data"]
    return info
