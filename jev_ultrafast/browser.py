"""Observed actions through Browser Harness; one CDP session, no per-step subprocess."""

import hashlib
import json
import os
import sys
import time
from pathlib import Path

from browser_harness.admin import ensure_daemon
from browser_harness.helpers import cdp

# Atomically read visible content and controls, preserving actual DOM node identity.
READ_STATE = Path(__file__).with_name("snapshot.js").read_text()
MARKER = f"(() => {{ const state={READ_STATE}; return state?.marker ?? null; }})()"

class StalePage(ValueError):
    """A decision no longer refers to the observed page."""


class Browser:
    def __init__(self, url):
        ensure_daemon()
        # Background by default so a run does not steal the window. JEV_FOREGROUND=1 brings it to
        # the front instead, for watching a run rather than reading it afterwards.
        watching = os.environ.get("JEV_FOREGROUND") == "1"
        self.target = cdp("Target.createTarget", url="about:blank", background=not watching)["targetId"]
        self.session = cdp("Target.attachToTarget", targetId=self.target, flatten=True)["sessionId"]
        self.call("Emulation.setDeviceMetricsOverride", width=1120, height=780, deviceScaleFactor=1, mobile=False)
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
        """The page's own session, plus one per cross-origin iframe, with each frame's offset.

        A cross-origin iframe runs out of process, so it is absent from the page's frame tree and
        unreachable from the page's JavaScript: the reader running in the page sees nothing of it.
        It is its own CDP target, though, and the same reader works inside it. The offset is where
        the frame sits on the page, because the Input domain exists only on the page target, so a
        click worked out inside a frame has to be dispatched in the page's coordinates.
        """
        found = [{"session": self.session, "offset": (0.0, 0.0)}]
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

    def read(self, screenshot):
        """One observation of the page and of every cross-origin frame in it, as a single state.

        Ids, guards and page keys are namespaced by frame, because each frame numbers its own nodes
        from one and a decision has to say which frame it is about. Each action carries the session
        that owns it and where its frame sits, so it can be validated and executed later.
        """
        merged = None
        for index, frame in enumerate(self.frames()):
            state = browser_operation(
                {"operation": "observe", "session": frame["session"], "screenshot": screenshot and index == 0}
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

    def settle(self, screenshot=False, timeout=6.0, previous=None, stabilise=False):
        """Observe until the page can be acted on, and wait for it to stop moving only when the
        last action was the kind that keeps a page moving.

        Waiting for an operable page is what removes a wasted call: an empty action space can only
        be answered BLOCKED. Waiting further costs protocol calls, which is what this reader exists
        to keep down, so it is spent only after a transition that tends to stream content in - a
        navigation, a menu or dialog opening, or a large change in what is on offer. A fingerprint
        change is deliberately not a trigger: it moves when a field is typed into or the page is
        scrolled, neither of which means the page is still arriving.
        """
        deadline = time.monotonic() + timeout
        page = self.observe(screenshot=screenshot)
        while not operable(page) and time.monotonic() < deadline:
            time.sleep(0.15)
            page = self.observe(screenshot=screenshot)
        if not (stabilise or previous is None or unsettling(previous, page)):
            return page
        held = 0
        while time.monotonic() < deadline:
            time.sleep(0.3)
            following = self.observe(screenshot=screenshot)
            held = held + 1 if following["fingerprint"] == page["fingerprint"] else 0
            page = following
            if held >= 3 and operable(page):
                return page
        return page

    def act(self, action, page, text=None):
        if not self.fresh(page, action):
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
        if os.environ.get("JEV_KEEP_OPEN") == "1":
            self.target = None  # leave the page up to be looked at
            return
        if self.target:
            cdp("Target.closeTarget", targetId=self.target)
            self.target = None


def expanded(page):
    return sum(1 for action in page["actions"] if str(action.get("expanded")).lower() == "true")


def unsettling(previous, current):
    """Whether the change between two observations is the kind that keeps arriving."""
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
