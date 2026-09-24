"""Observed actions through Browser Harness; one CDP session, no per-step subprocess."""

import hashlib
import json
import os
import sys
import time
from pathlib import Path

from browser_harness.admin import ensure_daemon
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
SETTLE_TIMEOUT = float(os.environ.get("JEV_SETTLE_TIMEOUT", "6"))
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
VIEWPORT = (1120, 780)

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


class StalePage(ValueError):
    """A decision no longer refers to the observed page."""


class Browser:
    def __init__(self, url):
        ensure_daemon()
        # Background by default so a run does not steal the window. JEV_FOREGROUND=1 brings it to
        # the front instead, for watching a run live.
        watching = os.environ.get("JEV_FOREGROUND") == "1"
        self.target = cdp("Target.createTarget", url="about:blank", background=not watching)["targetId"]
        self.session = cdp("Target.attachToTarget", targetId=self.target, flatten=True)["sessionId"]
        width, height = VIEWPORT
        self.call(
            "Emulation.setDeviceMetricsOverride", width=width, height=height, deviceScaleFactor=1, mobile=False
        )
        # Keep rAF/menus rendering in an owned background tab, without activating the user's Chrome tab.
        self.call("Emulation.setFocusEmulationEnabled", enabled=True)
        if watching:
            cdp("Target.activateTarget", targetId=self.target)
        self.call("Page.navigate", url=url)
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            if self.evaluate("document.readyState") == "complete":
                break
            time.sleep(0.02)

    def call(self, method, **params):
        return cdp(method, session_id=self.session, **params)

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
            if width < USABLE_FRAME or height < USABLE_FRAME or box[0] > VIEWPORT[0] or box[1] > VIEWPORT[1]:
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

    def observe(self, screenshot=True):
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
        for attempt in range(10):
            try:
                return self.read(screenshot)
            except StalePage:
                if attempt == 9:
                    raise
                time.sleep(0.02)
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

    def read(self, screenshot):
        """One observation of the page, and of its frames when those are worth reading.

        Ids, guards and page keys are namespaced by frame whether or not any frame is read, because
        every frame numbers its own nodes from one and a decision has to say which frame it means.
        Numbering only when a frame happens to be present would make the same element answer to two
        different names depending on the page, and a guard looked up under the wrong one reads as a
        page that has moved.
        """
        merged = None
        for index, frame in enumerate(self.frames_to_read(screenshot)):
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

    def frames_to_read(self, screenshot):
        """The page, plus its frames when they are worth the calls. The page's own observation is
        carried along so it is never taken twice."""
        page = browser_operation({"operation": "observe", "session": self.session, "screenshot": screenshot})
        first = {"session": self.session, "offset": (0.0, 0.0), "state": page}
        if not self.worth_reading_frames(page):
            return [first]
        return [first] + self.frames()

    def fresh(self, page, action=None):
        if action is not None and action["kind"] in {"click", "select"}:
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
        page = self.observe(screenshot=screenshot)
        if glimpse:
            # A reading taken while waiting is enough to decide from. The wait governs when to act,
            # not when to start deciding.
            glimpse(page)
        while not operable(page) and time.monotonic() < deadline:
            # Whether anything can be acted on is a yes or no, not a window to wait out, so it is
            # asked as often as it is cheap to ask.
            time.sleep(SETTLE_POLL)
            page = self.observe(screenshot=screenshot)
            if glimpse:
                glimpse(page)
        if not (stabilise or previous is None or unsettling(previous, page)):
            return page
        # If it has already been quiet for long enough, it is still, and watching would only
        # confirm what it has just said.
        if self.quiet_ms() >= SETTLE_STILL * 1000:
            return page
        watched = self.still_screen(screenshot, deadline, glimpse)
        if watched is not None:
            return watched
        still_since = None
        while time.monotonic() < deadline:
            time.sleep(SETTLE_POLL)
            following = self.observe(screenshot=screenshot)
            if glimpse:
                glimpse(following)
            # Hydration has quiet moments, so a page counts as still once it has been unchanged
            # for a while, not once two reads happen to agree.
            still_since = (still_since or time.monotonic()) if following["fingerprint"] == page["fingerprint"] else None
            page = following
            if operable(page) and (
                (still_since and time.monotonic() - still_since >= SETTLE_STILL)
                or self.quiet_ms() >= SETTLE_STILL * 1000
            ):
                return page
        return page

    def watching_paint(self):
        """Ask the browser to report its painting, once per page. False when it will not."""
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
                settled = self.observe(screenshot=screenshot)
                if glimpse:
                    glimpse(settled)
                return settled if operable(settled) else None
            time.sleep(0.01)
        return None

    def quiet_ms(self):
        """How long the page reports going without changing. Zero when it cannot report."""
        try:
            answer = self.evaluate(QUIET)
        except (StalePage, RuntimeError):
            return 0.0
        return float(answer) if isinstance(answer, (int, float)) else 0.0

    def act(self, action, page, text=None, insist=False):
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
        })
        self.after_input = action if action["kind"] != "wait" else None
        return result

    def close(self):
        if getattr(self, "screencast", False):
            # A page outlives the reader that started the report, and Chrome refuses a second
            # screencast on a target that already has one.
            try:
                self.call("Page.stopScreencast")
            except (RuntimeError, KeyError):
                pass
            self.screencast = None
        if os.environ.get("JEV_KEEP_OPEN") == "1":
            self.target = None  # leave the page up to be looked at
            return
        if self.target:
            cdp("Target.closeTarget", targetId=self.target)
            self.target = None


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

        if kind == "scroll":
            send("Input.dispatchMouseEvent", type="mouseWheel", x=550, y=650, deltaX=0, deltaY=action["delta"])
        elif kind != "wait":
            if type(action["node"]) is not int:
                raise ValueError("Invalid observed node")
            # Code-owned node IDs refer to actual observed elements, never model-generated selectors.
            target = evaluate("""(action => {
              const e=window.__jevFast?.nodes.get(action.node);
              if (!e?.isConnected || e.matches(':disabled') || e.closest('[aria-disabled="true"],[inert]') ||
                  !e.checkVisibility({checkOpacity:true,checkVisibilityCSS:true})) return null;
              if (action.kind==='fill' && (e.readOnly || e.getAttribute('aria-readonly')==='true')) return null;
              const r=e.getBoundingClientRect(), x=r.x+r.width/2, y=r.y+r.height/2;
              if (!r.width || !r.height || x<0 || y<0 || x>=innerWidth || y>=innerHeight) return null;
              if (!e.contains(document.elementFromPoint(x,y))) return null;
              if (action.kind==='select') {
                if (e.tagName!=='SELECT' || ![...e.options].some(o=>o.value===action.value &&
                    !o.disabled && !o.closest('optgroup[disabled]'))) return null;
                e.value=action.value;
                e.dispatchEvent(new Event('input',{bubbles:true}));
                e.dispatchEvent(new Event('change',{bubbles:true}));
              }
              return {x,y};
            })(""" + json.dumps(action) + ")")
            if target is None:
                if kind == "select":
                    raise RuntimeError("Dropdown execution was not confirmed; inspect before retrying.")
                raise StalePage("Target changed or is covered. Observe again.")
            if kind != "select":
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
