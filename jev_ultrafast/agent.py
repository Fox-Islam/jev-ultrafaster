"""The complete agent loop. Typed choices, observable state, bounded execution."""

import base64
import os
import threading
import time
from pathlib import Path
from urllib.parse import urlparse

from . import session
from .browser import RESTLESS, Browser, StalePage, diagnosis, operable
from .model import action_space, choose, field_context, field_text, field_values, readable_only, unseen
from .questions import (
    EVIDENCE_TEXT,
    FIXATION_REPEATS,
    FIXATION_WINDOW,
    MAX_STEPS,
    PLAN_SATISFIED,
    TEXT_ATTEMPTS,
    UNSEEN_ENOUGH,
)


class Agent:
    def __init__(self, url, goals, *, browser=None, record_dir=None, screenshots=False, track_plan=False,
                 reuse_held=True, read_only=False, allowed_hosts=None, max_steps=MAX_STEPS,
                 queries=(), keep_open_s=0):
        steps = [goals] if isinstance(goals, str) else list(goals)
        # A goal list may carry questions for the page as well as things to do. A question is
        # asked once everything named before it is satisfied, so a caller can read the page part
        # way through rather than only at the end.
        self.asked, wanted = [], []
        for step in steps:
            if isinstance(step, dict) and step.get("query"):
                self.asked.append({"query": step["query"], "after": len(wanted)})
            elif isinstance(step, str) and step.strip():
                wanted.append(step.strip())
        self.asked += [{"query": query, "after": len(wanted)} for query in (queries or ())]
        steps = wanted
        task = "\n".join(steps)
        if not task:
            raise ValueError("Supply a task")
        # Tracking asks per sub-goal, so it needs them kept apart instead of joined into one string.
        self.track_plan = track_plan and len(steps) > 1
        plan = steps if self.track_plan else [task]
        # A sub-goal can be satisfied by an action aimed at another, so this is a set, not an index.
        self.plan_satisfied = set()
        # Decisions taken for later sub-goals, kept until they apply or are overwritten.
        self.held = {}
        self.ready_values = {}
        self.reuse_held_answers = reuse_held
        self.reused = 0
        self.pending_text = None
        self.text_failures = 0
        self.speculating = os.environ.get("JEV_SPECULATE", "1") != "0"
        self.guessing, self.guessed = None, None
        self.unfresh_since = None
        self.guesses, self.guesses_used = 0, 0
        self.read_only = read_only
        # Matched on suffix, so naming a site covers its subdomains without listing them.
        self.allowed_hosts = tuple(host.lower().lstrip(".") for host in (allowed_hosts or ()))
        self.max_steps = max_steps
        self.keep_open_s = keep_open_s
        self.answered = {}
        # A browser passed in is borrowed: it keeps whatever page it is on unless a url says
        # otherwise, and it outlives this agent, so a run can continue where another left off.
        self.borrowed = browser is not None
        self.browser = browser or Browser(url)
        if self.borrowed and url:
            self.browser.navigate(url)
        self.record_dir = Path(record_dir) if record_dir else None
        self.screenshots = screenshots or bool(record_dir)
        try:
            page = self.browser.settle(screenshot=self.screenshots)
        except Exception:
            if not self.borrowed:
                self.browser.close()
            raise
        self.state = dict(
            browser=self.browser,
            url=url,
            goal="\n".join(plan),
            page=page,
            decision=None,
            history=[],
            status="ready",
            evidence=None,
            reason=None,
            plan=plan,
            plan_index=0,
            decisions=[],
            text_calls=[],
            elapsed_ms=0,
            started_at=None,
            record=bool(self.record_dir),
        )
        if self.record_dir:
            self.record_dir.mkdir(parents=True, exist_ok=True)
            (self.record_dir / "000000.jpg").write_bytes(base64.b64decode(page["screenshot"]))

    STOPPED = {"done", "blocked", "budget", "off_site"}

    def snapshot(self):
        browser = self.state.get("browser")
        if browser is not None and getattr(self, "asked", None):
            self.ask_due(self.state)
        if browser is not None and self.state.get("status") in self.STOPPED and not self.state.get("evidence"):
            # Every stopped run says what the page was, not only the ones that found something.
            self.state["evidence"] = self.evidence(self.state)
        if (browser is not None and self.state.get("status") in self.STOPPED
                and getattr(self, "keep_open_s", 0) and not getattr(self, "handle", None)):
            # Made when the run stops, not when it is closed: a caller reads the result before
            # closing, and a handle that appears afterwards is a handle it never sees.
            self.handle = session.handle_for(browser)
        faults = getattr(self.state.get("browser"), "faults", None)
        return {
            **{k: v for k, v in self.state.items() if k != "browser"},
            # After the state, which holds no faults of its own and would otherwise blank these.
            "faults": diagnosis(faults) if isinstance(faults, dict) else {},
            "handle": getattr(self, "handle", None),
            "elements": action_space(self.state["page"]["actions"])[0],
        }

    def command(self, name, body=None):
        body = body or {}
        state = self.state
        if name == "tick":
            try:
                self.command("predict", {})
                return self.command("act", {"fingerprint": state["page"]["fingerprint"]})
            except StalePage:
                state["decision"] = None
                state["status"] = "ready"
                state["page"] = state["browser"].settle(screenshot=self.screenshots, stabilise=True)
                state["elapsed_ms"] = round((time.perf_counter() - state["started_at"]) * 1000)
                return self.snapshot()
        elif name == "predict":
            if not state["browser"]:
                raise ValueError("Start a demo first")
            if state["started_at"] is None:
                state["started_at"] = time.perf_counter()
            state["decision"] = None
            if state["status"] in {"done", "blocked"}:
                raise ValueError("This run has stopped. Start a fresh demo.")
            if len(state["decisions"]) >= MAX_STEPS * 2:
                raise ValueError("Reached the demo's model-call budget")
            outstanding = (
                [i for i in range(len(state["plan"])) if i not in self.plan_satisfied]
                if getattr(self, "track_plan", False)
                else []
            )
            reused = self.reuse_held(outstanding) if getattr(self, "reuse_held_answers", False) else None
            if reused:
                state["decision"] = reused
                self.reused += 1
                state["status"] = "predicted"
                return self.snapshot()
            if getattr(self, "read_only", False):
                state["page"] = dict(state["page"], actions=readable_only(state["page"]["actions"]))
            asked_about = state["page"]
            arguments = (
                state["goal"],
                state["history"],
                [state["plan"][i] for i in outstanding],
                self.fixated(),
            )
            ahead = self.take_guess(state["page"]["fingerprint"]) if self.guessing_ahead() else None
            if ahead is not None:
                state["decision"] = ahead
            if ahead is None and not self.settled_or_restless(asked_about):
                # Without `previous`, so this waits for stillness. Naming the page makes it a
                # readiness check instead, which is quicker and fails one run in three: the wait is
                # what keeps a decision from being taken on a moving page and then rejected.
                state["page"] = state["browser"].settle(screenshot=self.screenshots)
            if state["decision"] is None:
                state["decision"] = choose(state["page"], *arguments)
            for index, entry in zip(outstanding, state["decision"].get("plan") or []):
                satisfied = entry["satisfied"]
                if satisfied is not None and satisfied >= PLAN_SATISFIED:
                    self.plan_satisfied.add(index)
            state["plan_satisfied"] = sorted(self.plan_satisfied)
            self.held = {
                index: entry
                for index, entry in zip(outstanding, state["decision"].get("plan") or [])
                if entry.get("label") and index not in self.plan_satisfied
            }
            # After the holds are known, because what they name is what needs a value.
            self.fetch_values(state)
            state["decisions"].append(
                {
                    **state["decision"],
                    "fingerprint": state["page"]["fingerprint"],
                    "elapsed_ms": round((time.perf_counter() - state["started_at"]) * 1000),
                }
            )
            state["status"] = "predicted"
        elif name == "act":
            decision, page = state["decision"], state["page"]
            if not decision or body.get("fingerprint") != page["fingerprint"]:
                raise ValueError("Observe and choose before acting")
            # Consume once, before any mutation or model call. A retry cannot double-click.
            state["decision"] = None
            if getattr(self, "track_plan", False) and len(self.plan_satisfied) == len(state["plan"]):
                # Every sub-goal reads satisfied. Several narrow checks agreeing is firmer evidence
                # than one broad DONE, which answers weakly when the page arguably already fits.
                state["status"] = "done"
                state["elapsed_ms"] = round((time.perf_counter() - state["started_at"]) * 1000)
                return self.snapshot()
            selected = decision["choice"]
            if selected in {"DONE", "BLOCKED"}:
                if not self.settled_or_restless(page):
                    state["status"] = "ready"
                    raise StalePage("Page changed since the decision. Choose again.")
                if selected == "DONE":
                    state["status"] = "done"
                else:
                    state["status"] = "blocked"
                    # A page read to the bottom without the thing on it is a different answer from
                    # one abandoned part way down, and a planner can only act on the difference.
                    state["reason"] = "stuck" if unseen(page) >= UNSEEN_ENOUGH else "not_found"
                state["plan_index"] = int(selected == "DONE")
                state["elapsed_ms"] = round((time.perf_counter() - state["started_at"]) * 1000)
                return self.snapshot()
            action = next(a for a in page["actions"] if a["id"] == selected)
            if len(state["history"]) >= getattr(self, "max_steps", MAX_STEPS):
                # A budget reached is an outcome, not a fault: a caller cannot tell one from a
                # broken run when both arrive as an exception.
                state["status"] = "budget"
                state["elapsed_ms"] = round((time.perf_counter() - state["started_at"]) * 1000)
                return self.snapshot()
            text, helper = None, None
            if action["kind"] == "fill":
                if not self.settled_or_restless(page):
                    raise StalePage("Page changed before text generation. Choose again.")
                # A held decision was taken for one sub-goal, so that is what the field is for.
                # Handing over the whole task instead invites a value inferred from the wrong part
                # of it: asked to fill an email field under a five-line goal, with "Ada" and
                # "Lovelace" just typed, the helper answered "Ada Lovelace".
                held_for = decision.get("reused_for")
                wanted = state["plan"][held_for] if held_for is not None else state["goal"]
                context = field_context(wanted, action, page, state["history"])
                ready = getattr(self, "ready_values", {}).pop(action["label"], None)
                if ready is not None:
                    text, helper = ready, None
                elif self.pending_text and self.pending_text[0] == context:
                    _, text, helper = self.pending_text
                else:
                    try:
                        text, helper = field_text(context)
                    except ValueError:
                        # The helper answered with nothing usable. No input has been sent, so the
                        # decision can be taken again; this is not a mutation retry.
                        self.text_failures += 1
                        if self.text_failures > TEXT_ATTEMPTS:
                            raise
                        state["status"] = "ready"
                        raise StalePage("Text helper gave nothing usable. Choose again.") from None
                    self.text_failures = 0
                    self.pending_text = (context, text, helper)
                    state["text_calls"].append({**helper, "field": action["label"], "value": text})
            # Browser.act checks freshness immediately before input, including after text generation.
            state["browser"].act(action, page, text=text, insist=self.restless())
            self.pending_text = None
            state["elapsed_ms"] = round((time.perf_counter() - state["started_at"]) * 1000)
            # Record execution before observing. A stale post-action observation must not erase the action.
            state["history"].append(
                {
                    "step": len(state["history"]) + 1,
                    "action": action["label"],
                    "kind": action["kind"],
                    "choice": selected,
                    "probability": decision["probabilities"][selected],
                    "confidence": decision["confidence"],
                    "latency_ms": decision["latency_ms"],
                    "text": text,
                    "text_helper": helper["model"] if helper else None,
                    "text_latency_ms": helper["latency_ms"] if helper else 0,
                    "operation": decision["operation"],
                    "target": decision["target"],
                    "page_changed": None,
                    "url": page["url"],
                    "usage": decision["usage"],
                    "executed_ms": round((time.perf_counter() - state["started_at"]) * 1000),
                    "elapsed_ms": state["elapsed_ms"],
                }
            )
            state["page"] = state["browser"].settle(
                screenshot=self.screenshots, previous=page,
                glimpse=self.guess_ahead((state["goal"], state["history"], [], self.fixated()))
                if self.guessing_ahead() else None,
            )
            state["elapsed_ms"] = round((time.perf_counter() - state["started_at"]) * 1000)
            state["history"][-1].update(
                page_changed=state["page"]["fingerprint"] != page["fingerprint"],
                url=state["page"]["url"],
                # Where the action left the page. Only the states run() yields carried this, so a
                # finished run could not say whether a goal was missed below the fold.
                scroll=dict(state["page"].get("scroll") or {}),
                elapsed_ms=state["elapsed_ms"],
            )
            if state["record"]:
                (self.record_dir / f"{state['elapsed_ms']:06d}.jpg").write_bytes(
                    base64.b64decode(state["page"]["screenshot"])
                )
            if self.off_site(state["page"]):
                # Stopped where it went, not where it started: the caller needs the address that
                # was refused to know which link took the run off the site.
                state["status"] = "off_site"
                state["refused_url"] = state["page"].get("url")
                state["elapsed_ms"] = round((time.perf_counter() - state["started_at"]) * 1000)
                return self.snapshot()
            repeated = state["history"][-3:]
            state["status"] = (
                "blocked"
                if len(repeated) == 3 and all(h["page_changed"] is False and h["kind"] != "wait" for h in repeated)
                else "ready"
            )
        else:
            raise ValueError("Unknown command")
        return self.snapshot()

    def off_site(self, page):
        """Whether this page is outside the hosts the run was allowed."""
        allowed = getattr(self, "allowed_hosts", ())
        if not allowed:
            return False
        host = urlparse(page.get("url") or "").hostname or ""
        return not any(host.lower() == name or host.lower().endswith("." + name) for name in allowed)

    def ask_due(self, state):
        """Ask the questions whose goals are all satisfied, once each.

        Read-only governs what this agent will do to a page, not what its caller may ask it: a
        question is the caller's own code, and running it is the point of asking.
        """
        satisfied = len(state["plan"]) if state["status"] in {"done", "budget"} else len(self.plan_satisfied)
        for index, question in enumerate(getattr(self, "asked", [])):
            if index in self.answered or question["after"] > satisfied:
                continue
            self.answered[index] = state["browser"].ask(question["query"])
        state["queries"] = [self.answered.get(i) for i in range(len(getattr(self, "asked", [])))]

    def evidence(self, state):
        """What the run is pointing at, taken from the page rather than written by the model.

        A caller asking whether a page works needs the thing that answers it, and a model asked to
        describe that would be writing the answer it is being checked against.
        """
        page, last = state["page"], (state["decisions"] or [{}])[-1]
        named = state["browser"].identity()
        faults = getattr(state["browser"], "faults", {}) or {}
        # Always present, so a caller can tell a page that failed from one that is simply an image
        # or a file, which answers no goal and has nothing on screen.
        found = {
            "url": page.get("url"),
            "title": named.get("title") or page.get("title"),
            "h1": named.get("h1"),
            "document_status": faults.get("document_status"),
            "document_type": faults.get("document_type"),
        }
        # DONE names no element, and the action before it is usually a scroll, so falling back to
        # that reports the scrollbar as the evidence. Where nothing is named, the page speaks.
        chosen = next((a for a in page["actions"] if a["id"] == last.get("choice")), None)
        if chosen is not None:
            found["element"] = {
                "label": chosen.get("label"),
                "role": chosen.get("role"),
                "value": chosen.get("value"),
                "text": (self.state["browser"].text_of(chosen) or "")[:EVIDENCE_TEXT],
            }
        else:
            found["text"] = (page.get("text") or "")[:EVIDENCE_TEXT]
        return found

    def restless(self):
        """Whether the page has gone so long without holding still that waiting is futile.

        The wait has already run by this point, and run again. A page that will never satisfy a
        check that it stopped changing would otherwise never be acted on at all.
        """
        since = getattr(self, "unfresh_since", None)
        return since is not None and time.perf_counter() - since >= RESTLESS

    def settled_or_restless(self, page):
        """Report whether the page still matches, and keep the clock on how long it has not."""
        if self.state["browser"].fresh(page):
            self.unfresh_since = None
            return True
        if getattr(self, "unfresh_since", None) is None:
            self.unfresh_since = time.perf_counter()
        return self.restless()

    def guessing_ahead(self):
        """Whether decisions may start before the page they are about has settled."""
        return getattr(self, "speculating", False)

    def guess_ahead(self, arguments):
        """Start deciding about a page while the wait is watching it.

        A decision takes about as long as a wait, and the two need not be consecutive: the page a
        wait ends on is usually the page it was showing part way through. The answer is kept only
        when the settled page is the one it was asked about, so nothing is acted on early. A wrong
        guess costs one wasted call and no wall-clock, since it ran inside the wait.
        """
        def ask(page, arguments):
            try:
                self.guessed = (page["fingerprint"], choose(page, *arguments))
            except Exception:  # noqa: BLE001
                self.guessed = None

        def glimpse(page):
            if self.guessing is not None and self.guessing.is_alive():
                return  # one question in flight at a time; the newest page gets the next one
            asked = self.guessed[0] if self.guessed else None
            if page["fingerprint"] == asked or not operable(page):
                return
            self.guesses += 1
            self.guessing = threading.Thread(target=ask, args=(page, arguments), daemon=True)
            self.guessing.start()

        return glimpse

    def take_guess(self, fingerprint):
        """A decision already made about this exact page, if one was."""
        if getattr(self, "guessing", None) is not None:
            # Started before the page settled, so it is either already the answer or about to be.
            self.guessing.join()
            self.guessing = None
        guessed, self.guessed = getattr(self, "guessed", None), None
        if guessed and guessed[0] == fingerprint:
            self.guesses_used += 1
            return guessed[1]
        return None

    def fixated(self):
        """Controls chosen more than once in the recent past without moving the page.

        Costs nothing: `page_changed` is already recorded per action. It catches the shape that
        stopped a round-trip run dead - "Open Return" chosen until the no-progress guard fired,
        while the origin and destination were never attempted.
        """
        recent = self.state["history"][-FIXATION_WINDOW:]
        seen = {}
        for step in recent:
            if step.get("page_changed") is False and step.get("kind") != "wait":
                key = (step["action"], step["kind"])
                seen[key] = seen.get(key, 0) + 1
        return {key for key, count in seen.items() if count >= FIXATION_REPEATS}

    def fetch_values(self, state):
        """Ask for every field value this step will need, in one call instead of one each.

        The decisions just taken say which sub-goals are fills and which control each means, so the
        values can be asked for together. `field_values` carries what asking separately costs.
        """
        wanted, page = {}, state["page"]
        # A value is kept while its field is present and empty, and dropped once the field holds
        # something. Dropping on absence instead loses the batch whenever a decision is retried, or
        # whenever a field sits behind an open suggestion list, and the values are then fetched
        # again one at a time - twice the calls, for a saving.
        carries_value = {
            action["label"] for action in page["actions"] if action["kind"] == "fill" and action.get("value")
        }
        self.ready_values = {
            label: value for label, value in getattr(self, "ready_values", {}).items() if label not in carries_value
        }
        decision = state["decision"] or {}
        if decision.get("operation") == "TYPE_TEXT":
            acting = next((a for a in page["actions"] if a["id"] == decision.get("choice")), None)
            # Whatever it already holds: the decision is to type here, and a field arriving with
            # something in it - a city guessed from the connection, say - is one to replace.
            if acting is not None and acting["kind"] == "fill" and acting["label"] not in self.ready_values:
                wanted[acting["label"]] = (state["goal"], acting)
        for index, entry in self.held.items():
            if entry.get("operation") != "TYPE_TEXT" or not entry.get("label"):
                continue
            matches = [a for a in page["actions"] if a["label"] == entry["label"] and a["kind"] == "fill"]
            if len(matches) != 1 or matches[0].get("value") or matches[0]["label"] in wanted:
                continue
            if matches[0]["label"] in self.ready_values:
                continue
            wanted[matches[0]["label"]] = (state["plan"][index], matches[0])
        # Sub-goals are one way to know a field will be filled; the page is another. Empty fields
        # with names of their own are going to be filled from this goal or not at all, and asking
        # for their values now costs a call that is already being made. A value never used is the
        # waste; a value waited for one field at a time was the cost.
        if decision.get("operation") == "TYPE_TEXT":
            labels = [a["label"] for a in page["actions"] if a["kind"] == "fill"]
            for action in page["actions"]:
                if action["kind"] != "fill" or action.get("value") or action["label"] in wanted:
                    continue
                if action["label"] in self.ready_values:
                    continue
                if labels.count(action["label"]) == 1:
                    wanted[action["label"]] = (state["goal"], action)
        if len(wanted) < 2:
            return  # one field is one call either way
        # Keyed by position, not by the field's name. A name has to survive being echoed back
        # exactly, and a name it tidies on the way is a value dropped and fetched again one at a
        # time - which is what the batch was for.
        numbered = {f"f{n}": pair for n, pair in enumerate(wanted.values())}
        labels = {f"f{n}": label for n, label in enumerate(wanted)}
        try:
            values, helper = field_values(numbered, page, state["history"])
        except (ValueError, RuntimeError):
            return  # each field falls back to its own call
        if helper:
            state["text_calls"].append({**helper, "field": f"{len(values)} fields", "value": None})
        self.ready_values.update({labels[key]: value for key, value in values.items() if key in labels})

    def reuse_held(self, outstanding):
        """A decision held for a sub-goal, re-resolved against the page as it is now.

        The answer is reused, not the element it named: a node id survives a change of meaning, so
        the label and kind are looked up again and must match exactly one control. Because the
        action comes from the current snapshot, the ordinary freshness gate still applies to it.
        """
        state = self.state
        for index in outstanding:
            entry = self.held.get(index)
            if not entry or entry["operation"] in {"DONE", "BLOCKED", "WAIT"}:
                continue
            matches = [
                action
                for action in state["page"]["actions"]
                if action["label"] == entry["label"] and action["kind"] == entry["kind"]
            ]
            if len(matches) != 1:
                continue
            action = matches[0]
            # Sub-goals asked against one page converge on whatever control that page makes
            # obvious, so a held answer is usually the action just taken. Repeating it is not
            # progress, and on the flights task it produced three clicks on the same suggestion.
            if any(
                past["action"] == action["label"] and past["kind"] == action["kind"]
                for past in state["history"][-2:]
            ):
                continue
            if action["kind"] == "fill" and action.get("value"):
                continue  # already carries a value; re-typing it is not progress
            # Retire only the answer being used. The throttle that cleared every held answer
            # after one reuse was protecting against stale satisfaction readings, but an answer is
            # re-resolved against the current page before it is used, and a filled field is
            # skipped, so the guards already cover what the throttle was standing in for.
            self.held.pop(index, None)
            return {
                "choice": action["id"],
                "operation": entry["operation"],
                "target": None,
                "confidence": entry["confidence"],
                "probabilities": {action["id"]: entry["confidence"] or 0.0},
                "operation_probabilities": {},
                "target_probabilities": {},
                "target_confidence": None,
                "raw_answers": {},
                "plan": [],
                "model": "held",
                "usage": {},
                "latency_ms": 0,
                "reused_for": index,
                "request": None,
            }
        return None

    def run(self):
        while self.state["status"] not in {"done", "blocked"}:
            yield self.command("tick")

    def close(self):
        """Close the browser, unless it was passed in or is being kept open.

        A borrowed browser is the caller's. A kept one stays, and so does this worker's claim on
        the daemon: the page is still there to be asked about, and a second run would take the
        browser from under it. `free` gives both back once the caller has finished with the page.
        """
        if getattr(self, "borrowed", False):
            return
        if getattr(self, "handle", None):
            return  # the page is held; closing it is the handle holder's to do
        self.browser.close()

    def free(self):
        """Close a held page and report this worker free.

        Called when the caller has finished with the handle. Safe when the page is already gone,
        because a registry sweeping orphans cannot know what the worker closed on its way out.
        """
        handle = getattr(self, "handle", None)
        closed = session.close_handle(handle) if handle else {}
        self.handle = None
        self.browser.release()
        return closed

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self.close()
