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

from .browser import Browser

VERSION = 1


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
    borrowed = browser is not None
    browser = browser or Browser(document["url"])
    started = time.perf_counter()
    done = []
    try:
        page = browser.settle(screenshot=screenshots, stabilise=True)
        for number, step in enumerate(document["steps"], 1):
            action = control(page, step, number)
            browser.act(action, page, text=step.get("text"))
            previous, page = page, browser.settle(screenshot=screenshots, previous=page)
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
