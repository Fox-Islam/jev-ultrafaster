"""A run's actions as a script, and a script run back without the model.

Controls are named, not numbered. A node id identifies nothing on a page loaded again, which is
why `Agent.reuse_held` re-resolves a held decision by label and kind rather than reusing the
element it named; a script is the same idea across runs. The page in front of the replay decides
which element a name refers to, so a script survives the page being laid out differently, and a
name that no longer matches exactly one control stops the replay instead of being guessed at.

A replay calls no model. It is the cheap, repeatable half of a run: the deciding happened once.
"""

import json
import time

from .browser import Browser, StalePage
from .model import submits

VERSION = 1

# How many times a step is re-resolved when the page moves under it. A page that settles late
# changes between the reading a step was matched against and the input being sent, which is what
# StalePage reports; looking again is the answer, and a step that keeps moving is a real failure.
STEP_ATTEMPTS = 3

# A dropped connection loses the browser and everything on it, and a replay cannot resume part
# way through a page that no longer exists. Taking the whole script again is only safe where the
# steps leave nothing behind: repeating a script that types, selects or submits would send its
# input twice, and the far side has no way to tell the second time from the first.
RECONNECTS = 1


def mutates(document):
    """Whether replaying this script twice would send anything twice."""
    return any(
        step["kind"] in {"fill", "select"} or (step["kind"] == "click" and submits(step))
        for step in document.get("steps") or []
    )


def script(state, name=None):
    """The actions a run executed, in a form `replay` can take.

    Everything a decision needed - probabilities, the element table it chose from, what each call
    cost - is left out. What remains is what was done.
    """
    steps = []
    for step in state.get("history") or []:
        recorded = {"kind": step["kind"], "label": step["action"]}
        if step.get("text") is not None:
            recorded["text"] = step["text"]
        steps.append(recorded)
    history = state.get("history") or []
    return {
        "version": VERSION,
        "name": name,
        "url": state.get("url") or (history[0].get("url") if history else None),
        "goal": state.get("goal"),
        "steps": steps,
    }


def write(state, path, name=None):
    """Write a run's script to `path`, and return it."""
    document = script(state, name=name)
    path.write_text(json.dumps(document, indent=2) + "\n")
    return document


def read(path):
    """A script from `path`, checked well enough to fail before opening a browser."""
    document = json.loads(path.read_text())
    if document.get("version") != VERSION:
        raise ValueError(f"Script version {document.get('version')!r} is not {VERSION}")
    if not document.get("url"):
        raise ValueError("Script has no url to start from")
    if not isinstance(document.get("steps"), list):
        raise ValueError("Script has no steps")
    return document


def control(page, step, number):
    """The one control on this page that the step names.

    Nothing matching, or more than one thing matching, stops the replay: choosing between them is
    the judgement the recording exists to avoid making twice.
    """
    wanted = (str(step.get("label", "")).strip(), step.get("kind"))
    found = [
        action
        for action in page["actions"]
        if (str(action.get("label", "")).strip(), action.get("kind")) == wanted
    ]
    if len(found) == 1:
        return found[0]
    offered = sorted({str(a.get("label", "")).strip() for a in page["actions"] if a.get("kind") == wanted[1]})
    raise ValueError(
        f"Step {number} ({wanted[1]} {wanted[0]!r}) matched {len(found)} controls. "
        f"This page offers: {', '.join(offered[:12]) or 'none of that kind'}"
    )


def replay(document, browser=None, screenshots=False, on_step=None):
    """Run a script back. Returns one record per step, in order.

    A browser may be supplied to replay into a page already open; otherwise one is opened at the
    script's url and closed afterwards.
    """
    if document.get("version") != VERSION:
        raise ValueError(f"Script version {document.get('version')!r} is not {VERSION}")
    borrowed, repeatable = browser is not None, not mutates(document)
    for attempt in range(RECONNECTS + 1):
        done = []
        try:
            run_once(document, browser, screenshots, on_step, done)
            return {"status": "done", "completed": len(done), "steps": done}
        except (RuntimeError, OSError) as dropped:
            # A borrowed browser belongs to the caller, so a lost connection is theirs to handle.
            if borrowed or not dropped_connection(dropped):
                raise
            if not repeatable or attempt == RECONNECTS:
                return {
                    "status": "connection_lost",
                    "completed": len(done),
                    "steps": done,
                    "error": str(dropped)[:200],
                    # Said plainly, because the caller has to decide whether repeating these steps
                    # is safe and cannot see from a status that the script types or submits.
                    "repeatable": repeatable,
                }
    raise RuntimeError("unreachable")


def dropped_connection(error):
    """Whether this error means the connection went rather than the page."""
    text = str(error).lower()
    return any(sign in text for sign in ("websocket", "connection closed", "not attached", "disconnected"))


def run_once(document, browser, screenshots, on_step, done):
    borrowed = browser is not None
    browser = browser or Browser(document["url"])
    started = time.perf_counter()
    try:
        page = browser.settle(screenshot=screenshots, stabilise=True)
        for number, step in enumerate(document["steps"], 1):
            for attempt in range(STEP_ATTEMPTS):
                try:
                    action = control(page, step, number)
                    browser.act(action, page, text=step.get("text"))
                    break
                except StalePage:
                    # The page moved between being read and being acted on. Read it again; the
                    # step names a control, so it can be found on whatever the page became.
                    if attempt == STEP_ATTEMPTS - 1:
                        raise
                    page = browser.settle(screenshot=screenshots, stabilise=True)
            # A scroll loads what was below the fold, and `previous` would call that unremarkable:
            # the url is the same, nothing expanded, and the count of controls creeps rather than
            # jumps. Waiting for stillness is what lets the next step find what the scroll brought.
            scrolled = step["kind"] == "scroll"
            previous, page = page, browser.settle(
                screenshot=screenshots, previous=None if scrolled else page, stabilise=scrolled
            )
            record = {
                "step": number,
                "kind": step["kind"],
                "label": step["label"],
                "text": step.get("text"),
                "url": page["url"],
                "page_changed": page["fingerprint"] != previous["fingerprint"],
                "elapsed_ms": round((time.perf_counter() - started) * 1000),
            }
            done.append(record)
            if on_step:
                on_step(record)
    finally:
        if not borrowed:
            browser.close()
    return done


CONTROLS_REPORTED = 25
LABEL_LENGTH = 32
CONTROL_LENGTH = 24
# A url is kept to show where a step went, not to be followed. Google Flights encodes the whole
# search into a query string of well over a hundred characters, which would otherwise be most of a
# report.
URL_LENGTH = 90


def brief(label, length=LABEL_LENGTH):
    label = " ".join(str(label or "").split())
    return label if len(label) <= length else label[: length - 1] + "…"


def short_url(url):
    return brief(url, URL_LENGTH) if url else url


def report(state, controls=CONTROLS_REPORTED):
    """What a run did, small enough to hand back to whatever sets its goals.

    A planner deciding what to try next needs to know what was attempted and where it got to, not
    what the page said: page text and markup are the things too large to pass around, and an
    element id means nothing outside the run that read it. So this carries labels, positions and
    the operation probabilities behind the last decision, and nothing that grows with the page.

    A ten-step run reports in well under a kilobyte, which `test_a_report_stays_small` holds to.
    """
    page = state.get("page") or {}
    decisions = state.get("decisions") or []
    last = decisions[-1] if decisions else {}
    scroll = page.get("scroll") or {}
    # A url and a page height repeat unchanged across most steps of a run, and a planner reading
    # ten identical copies learns what one tells it. Each is carried when it differs from the step
    # before, so a step that says nothing about them happened where the last one left off.
    steps, url, height = [], state.get("url"), None
    for step in state.get("history") or []:
        position = step.get("scroll") or {}
        recorded = {
            "kind": step["kind"],
            "label": brief(step["action"]),
            "changed": bool(step.get("page_changed")),
            "y": position.get("y"),
        }
        if step.get("text") is not None:
            recorded["text"] = brief(step["text"])
        if step.get("url") and step["url"] != url:
            url = step["url"]
            recorded["url"] = short_url(url)
        if position.get("height") is not None and position["height"] != height:
            recorded["height"] = height = position["height"]
        steps.append(recorded)
    return {
        "url": state.get("url"),
        "goals": [goal for goal in (state.get("plan") or []) if goal],
        "status": state.get("status"),
        "steps": steps,
        "final": {
            "url": short_url(page.get("url")),
            "title": brief(page.get("title")),
            "y": scroll.get("y"),
            "height": scroll.get("height"),
            "controls": [
                brief(action.get("label"), CONTROL_LENGTH)
                for action in (page.get("actions") or [])
                if action.get("kind") in {"click", "fill", "select"}
            ][:controls],
        },
        "operations": {
            name: round(value, 3)
            for name, value in (last.get("operation_probabilities") or {}).items()
        },
        "reason": state.get("reason"),
        "evidence": state.get("evidence"),
        "refused_url": state.get("refused_url"),
        "faults": state.get("faults") or {},
        "queries": state.get("queries") or [],
        "handle": state.get("handle"),
    }
