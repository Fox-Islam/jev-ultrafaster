"""Live Google Flights search. Calls TypeSafe; never selects or books a flight.

The departure date rolls forward from today. A fixed date expires: this task named 20 September
2026, and from 21 September 2026 on, no runtime could pass it, because the site returns no flights
for a past day and the checker below looks for them. Pin a date with JEV_FLIGHTS_DATE to reproduce
one run; leave it unset to measure.
"""

import argparse
import base64
import json
import os
from datetime import date, timedelta
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from jev_ultrafast import Agent

URL = "https://www.google.com/travel/flights?hl=en"
DAYS_AHEAD = 30

# Google renders the page in English (hl=en) and the checks below match its strings, so the names
# are spelled out rather than taken from the process locale.
WEEKDAYS = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")
MONTHS = ("January", "February", "March", "April", "May", "June",
          "July", "August", "September", "October", "November", "December")


def departure_date():
    """The day searched for. Far enough ahead that the route is still being sold."""
    pinned = os.environ.get("JEV_FLIGHTS_DATE")
    return date.fromisoformat(pinned) if pinned else date.today() + timedelta(days=DAYS_AHEAD)


DEPARTURE = departure_date()
WEEKDAY, MONTH = WEEKDAYS[DEPARTURE.weekday()], MONTHS[DEPARTURE.month - 1]
GOALS = (
    f"Find one-way flights from Zurich to London on {MONTH} {DEPARTURE.day}, {DEPARTURE.year}, "
    "for one adult in economy. "
    "Stop when matching flight options are visible. Do not select or book a flight."
)


def verify(page, departure=None):
    """Independent checks on the resulting page, not the model's DONE answer."""
    departure = departure or DEPARTURE
    weekday, month = WEEKDAYS[departure.weekday()], MONTHS[departure.month - 1]
    parsed = urlparse(page["url"])
    encoded = parse_qs(parsed.query).get("tfs", [""])[0]
    try:
        date_in_url = departure.isoformat().encode() in base64.urlsafe_b64decode(
            encoded + "=" * (-len(encoded) % 4)
        )
    except ValueError:
        date_in_url = False
    actions = page["actions"]
    values = {a["label"].strip(): a.get("value") for a in actions}
    flights = [a["label"] for a in actions if "Select flight" in a["label"]]
    checks = {
        "search_page": parsed.hostname == "www.google.com" and parsed.path == "/travel/flights/search",
        "one_way": values.get("Change ticket type. One way") == "One way",
        "origin": values.get("Where from?") == "Zürich",
        "destination": values.get("Where to?") == "London",
        "date": values.get("Departure") == f"{weekday[:3]}, {month[:3]} {departure.day}",
        "year": date_in_url or f"departing {departure.isoformat()}" in page["text"],
        "results": bool(flights) and all(f"{weekday}, {month} {departure.day}" in f for f in flights),
    }
    return {"passed": all(checks.values()), "checks": checks, "visible_flights": flights}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="artifacts/flights/latest")
    parser.add_argument("--keep-open", action="store_true")
    args = parser.parse_args()
    folder = Path(args.output)
    folder.mkdir(parents=True, exist_ok=True)
    agent = Agent(URL, GOALS)
    try:
        for state in agent.run():
            last = state["history"][-1] if state["history"] else {}
            print(state["elapsed_ms"], state["status"], last.get("action", ""), flush=True)
    finally:
        state = agent.snapshot()
        state["verification"] = verify(state["page"])
        (folder / "state.json").write_text(json.dumps(state, indent=2))
        (folder / "session.json").write_text(
            json.dumps({"target": agent.browser.target, "session": agent.browser.session})
        )
        if not args.keep_open:
            agent.close()
    print(json.dumps(state["verification"], indent=2))
    if not state["verification"]["passed"]:
        raise SystemExit("Final page did not satisfy the route/date checks")


if __name__ == "__main__":
    main()
