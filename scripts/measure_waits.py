"""Measure the waiting on pages that never hold still. Local fixture + paid model APIs.

The wait is the part of a run with no visible output, so a change to it is easy to make and hard
to notice. These two pages isolate the two ways a page refuses to settle, and the two signals the
wait can use:

  paint  the screen repaints forever while the document stays unchanged. Stillness read from the
         screen is never reached, so this measures the fallback to the document.
  dom    the document is rewritten every frame as well, so neither signal ever reads still and a
         whole-page freshness check can never pass. This measures the restlessness deadline.

Both pages offer the same two controls from the first paint, so a run that does not finish was
stopped by the waiting. A run that stops making progress is ended and reported rather than waited
out, because a regression here shows up as a hang.
"""

import argparse
import json
import os
import statistics
import sys
import threading
import time
from http.server import ThreadingHTTPServer

from jev_ultrafast import Agent
from jev_ultrafast.demo import PORT, Handler, load_environment

GOALS = ["Set the name field to Ada", "Click the Continue button"]
STALL_SECONDS = float(os.environ.get("JEV_STALL", "20"))


def serve():
    """The demo server, if nothing is already answering on its port."""
    try:
        server = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    except OSError:
        return None
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def watch(progress, done):
    """End a run that has stopped executing actions, naming where it was."""
    while not done.wait(0.5):
        idle = time.monotonic() - progress["at"]
        if idle > STALL_SECONDS:
            print(f"STALLED after {progress['actions']} actions, {idle:.0f}s without one", flush=True)
            os._exit(3)


def measure(mode):
    url = f"http://127.0.0.1:{PORT}/restless.html" + ("?dom" if mode == "dom" else "")
    progress, done = {"at": time.monotonic(), "actions": 0}, threading.Event()
    threading.Thread(target=watch, args=(progress, done), daemon=True).start()
    try:
        return drive(url, mode, progress)
    finally:
        done.set()


def drive(url, mode, progress):
    with Agent(url, GOALS) as agent:
        try:
            for state in agent.run():
                if len(state["history"]) > progress["actions"]:
                    progress["at"], progress["actions"] = time.monotonic(), len(state["history"])
        except Exception as exc:  # noqa: BLE001
            return {"mode": mode, "error": f"{type(exc).__name__}: {exc}"}
        state = agent.snapshot()
        finished = "your details were received" in agent.browser.evaluate("document.body.innerText")
        values = {a["label"].strip(): (a.get("value") or "") for a in state["page"]["actions"]}
    return {
        "mode": mode,
        "ms": state["elapsed_ms"],
        "actions": len(state["history"]),
        "decisions": len(state["decisions"]),
        "typed": any("ada" in value.lower() for value in values.values()),
        "clicked": finished,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--mode", choices=("paint", "dom", "both"), default="both")
    args = parser.parse_args()
    load_environment()
    serve()
    modes = ("paint", "dom") if args.mode == "both" else (args.mode,)
    results, failures = [], 0
    for run in range(args.runs):
        # Alternating, so a slow minute falls on both pages instead of whichever ran second.
        for mode in modes:
            result = measure(mode)
            result["run"] = run + 1
            results.append(result)
            failures += 0 if result.get("typed") and result.get("clicked") else 1
            print(json.dumps(result), flush=True)
    for mode in modes:
        passed = [r["ms"] for r in results if r["mode"] == mode and r.get("clicked") and r.get("typed")]
        if passed:
            print(f"{mode}: {len(passed)}/{args.runs} finished, median {statistics.median(passed):.0f}ms")
        else:
            print(f"{mode}: 0/{args.runs} finished")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
