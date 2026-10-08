"""uv run --env-file .env python examples/run.py --url URL --goal 'A narrow goal'

--target TARGET_ID runs in a tab that is already open instead, as it stands: no new tab, no
reload, and the tab is left open afterwards.
"""

import argparse

from jev_ultrafast import Agent, Browser

parser = argparse.ArgumentParser()
start = parser.add_mutually_exclusive_group(required=True)
start.add_argument("--url")
start.add_argument("--target", help="CDP target id of an open tab to run in.")
parser.add_argument("--goal", action="append", required=True, help="Repeat for an ordered list of goals.")
args = parser.parse_args()

browser = Browser.attach(args.target) if args.target else None
try:
    with Agent(args.url, args.goal, browser=browser) as agent:
        for state in agent.run():
            print(f"{state['elapsed_ms']:>5} ms  {len(state['history'])} actions  {state['status']}")
        print(state["page"]["url"])
finally:
    if browser is not None:
        browser.close()
