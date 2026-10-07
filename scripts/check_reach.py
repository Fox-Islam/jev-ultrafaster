"""Local-browser checks for controls that are there but not plainly on screen. No model calls.

Each case is a shape a real app was seen to use: an injected toolbar in a shadow root, a delete
button that fades in under the pointer, a submit button below the fold of a scrolling panel, an
icon-only button, a card holding its own buttons, a list that arrives after the page goes still,
and the two kinds of drag and drop.
"""

import time
from urllib.parse import quote

from jev_ultrafast.browser import Browser
from jev_ultrafast.model import action_space, drop_zones

HTML = """<!doctype html><title>Reach checks</title>
<style>
  body{margin:20px;font:14px sans-serif} button{min-width:60px;height:32px}
  .row{display:flex;gap:8px;align-items:center;width:320px;padding:6px;border:1px solid #ccc}
  .row .reveal{opacity:0;transition:opacity .15s} .row:hover .reveal{opacity:1}
  #panel{height:120px;overflow-y:auto;border:1px solid #999;width:320px}
  #panel .spacer{height:400px}
  .zone{width:200px;height:90px;border:2px dashed #888;display:inline-block;margin:4px}
  [draggable]{width:120px;height:30px;background:#ddf}
  .card{display:block;width:300px;padding:8px;border:1px solid #ccc}
</style>
<div id="host"></div>
<div class="row"><span>notes.txt</span>
  <span class="reveal">
    <button aria-label="Remove file" onclick="window.removed=(window.removed||0)+1">x</button>
  </span></div>
<div id="panel"><div class="spacer">Long list</div><button onclick="window.created=1">Create folder</button></div>
<button id="icon" onclick="window.trashed=1"><svg class="lucide lucide-trash-2" width="16" height="16"></svg></button>
<div class="card" role="button" tabindex="0" onclick="window.opened=1">Example project https://example.com/
  <button onclick="event.stopPropagation();window.shared=1">Share</button></div>
<section aria-label="Done column" class="zone" id="native-zone" ondragover="event.preventDefault()"
  ondrop="event.preventDefault();window.native=event.dataTransfer.getData('text/plain')"></section>
<div draggable="true" ondragstart="event.dataTransfer.setData('text/plain','task-7')">Task seven</div>
<section aria-label="Archive" class="zone" id="pointer-zone"></section>
<div id="canvas" style="width:300px;height:90px;border:1px solid #bbb">Start here</div>
<div id="pointer-card" aria-roledescription="sortable" tabindex="0"
  style="width:120px;height:30px;background:#dfd">Card eight</div>
<script>
  // A drop target with nothing in its markup to say so, as an app's canvas usually is.
  const canvas=document.getElementById('canvas');
  canvas.ondragover=e=>e.preventDefault();
  canvas.ondrop=e=>{ e.preventDefault(); window.canvas=e.dataTransfer.getData('text/plain'); };
  document.getElementById('host').attachShadow({mode:'open'}).innerHTML =
    '<button onclick="window.shadowed=1">Filter tasks</button>';
  // A pointer-driven drag, as dnd-kit does it: press, move past a threshold, release over a zone.
  const card=document.getElementById('pointer-card'); let held=null;
  card.addEventListener('pointerdown', e => { held={x:e.clientX,y:e.clientY,moved:false}; });
  addEventListener('pointermove', e => {
    if (held && Math.hypot(e.clientX-held.x, e.clientY-held.y) > 10) held.moved=true; });
  addEventListener('pointerup', e => {
    if (held?.moved) {
      const zone=document.elementFromPoint(e.clientX,e.clientY)?.closest('#pointer-zone');
      window.pointer = zone ? 'archived' : 'missed';
    }
    held=null;
  });
</script>"""

ARRIVING = """<!doctype html><title>Arriving</title><main><h1>Projects</h1>
<p id="state">Loading projects…</p><button>Settings</button></main>
<script>setTimeout(()=>{document.getElementById('state').outerHTML=
  '<ul><li><button>Open Example</button></li></ul>'}, 2500)</script>"""


def find(page, label, kind="click"):
    found = [a for a in page["actions"] if a["label"] == label and a["kind"] == kind]
    assert len(found) == 1, f"{label!r} ({kind}) matched {len(found)}: {sorted({a['label'] for a in page['actions']})}"
    return found[0]


def main():
    browser = Browser("data:text/html," + quote(HTML))
    passed = []
    try:
        page = browser.observe(screenshot=False)

        filter_tasks = find(page, "Filter tasks")
        browser.act(filter_tasks, page)
        assert browser.evaluate("window.shadowed") == 1
        passed.append("control inside an open shadow root is offered and clicked")

        page = browser.observe(screenshot=False)
        remove = find(page, "Remove file")
        assert remove.get("hover") is True, remove
        browser.act(remove, page)
        assert browser.evaluate("window.removed") == 1
        passed.append("control revealed on hover is offered, hovered, then clicked")

        page = browser.observe(screenshot=False)
        create = find(page, "Create folder")
        assert create.get("offscreen") == "panel", create
        browser.act(create, page)
        assert browser.evaluate("window.created") == 1
        passed.append("control below the fold of a scrolling panel is offered and scrolled to")

        page = browser.observe(screenshot=False)
        trash = find(page, "trash icon")
        browser.act(trash, page)
        assert browser.evaluate("window.trashed") == 1
        passed.append("icon-only button is named from its icon")

        card = find(page, "Example project https://example.com/")
        find(page, "Share")
        assert "Share" not in card["label"]
        passed.append("a card is named by its own content, not its buttons'")

        page = browser.observe(screenshot=False)
        elements, targets, _ = action_space(page["actions"])
        zones = drop_zones(page["actions"])
        assert "DRAG" in targets, sorted(targets)
        assert {"Done column", "Archive"} <= {z["label"] for z in zones.values()}, zones
        native = find(page, "Task seven", "drag")
        done_column = next(z for z in zones.values() if z["label"] == "Done column")
        browser.act(native, page, drop=done_column)
        assert browser.evaluate("window.native") == "task-7", browser.evaluate("window.native")
        passed.append("native drag and drop reaches its drop handler")

        page = browser.observe(screenshot=False)
        unlabelled = next(
            z for z in drop_zones(page["actions"]).values() if z["label"].startswith("drop area containing: Start here")
        )
        browser.act(find(page, "Task seven", "drag"), page, drop=unlabelled)
        assert browser.evaluate("window.canvas") == "task-7", browser.evaluate("window.canvas")
        passed.append("an unlabelled area that handles drops is offered as a zone")

        page = browser.observe(screenshot=False)
        pointer = find(page, "Card eight", "drag")
        archive = next(z for z in drop_zones(page["actions"]).values() if z["label"] == "Archive")
        browser.act(pointer, page, drop=archive)
        assert browser.evaluate("window.pointer") == "archived", browser.evaluate("window.pointer")
        passed.append("pointer-driven drag ends over its zone")

        browser.navigate("data:text/html," + quote(ARRIVING))
        started = time.monotonic()
        page = browser.settle(screenshot=False, stabilise=True)
        assert any(a["label"] == "Open Example" for a in page["actions"]), [a["label"] for a in page["actions"]]
        passed.append(f"settle waits for a loading list to arrive ({time.monotonic() - started:.1f}s)")
    finally:
        browser.close()
    print("\n".join(passed))
    print(f"PASS: {len(passed)} reach checks; no model calls")


if __name__ == "__main__":
    main()
