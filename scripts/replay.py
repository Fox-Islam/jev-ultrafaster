"""Record a run's actions, or replay recorded ones. Replaying calls no model.

    uv run python scripts/replay.py record flights.json --url <url> --goal "..."
    uv run python scripts/replay.py run flights.json

Recording drives the task once and writes what it did. Running repeats those actions against the
live page, resolving each by name, which costs nothing and answers a different question: whether
the site still behaves the way the run assumed.
"""

import argparse
import json
import pathlib
import sys

from jev_ultrafast import Agent
from jev_ultrafast.demo import load_environment
from jev_ultrafast.replay import read, replay, write


def record(args):
    with Agent(args.url, args.goal) as agent:
        for state in agent.run():
            history = state["history"]
            print(state["elapsed_ms"], "ms", len(history), "actions",
                  history[-1]["action"] if history else "", flush=True)
        state = agent.snapshot()
    document = write(state, args.path, name=args.name)
    print(f"wrote {args.path}: {len(document['steps'])} steps from {document['url']}")
    return 0


def run(args):
    document = read(args.path)
    print(f"replaying {len(document['steps'])} steps from {document['url']}", flush=True)
    done = replay(document, on_step=lambda step: print(
        f"  {step['elapsed_ms']:>6}ms {step['kind']:<7} {step['label'][:48]!r}"
        f"{'' if step['page_changed'] else '  (page unchanged)'}", flush=True))
    print(json.dumps({"steps": len(done), "ms": done[-1]["elapsed_ms"] if done else 0}))
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    recorder = sub.add_parser("record", help="drive a task once and write what it did")
    recorder.add_argument("path", type=pathlib.Path)
    recorder.add_argument("--url", required=True)
    recorder.add_argument("--goal", required=True, action="append", dest="goal")
    recorder.add_argument("--name")
    recorder.set_defaults(handler=record)
    runner = sub.add_parser("run", help="replay a recorded script")
    runner.add_argument("path", type=pathlib.Path)
    runner.set_defaults(handler=run)
    args = parser.parse_args()
    load_environment()
    return args.handler(args)


if __name__ == "__main__":
    sys.exit(main())
