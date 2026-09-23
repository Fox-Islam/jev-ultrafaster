"""The complete agent loop. Typed choices, observable state, bounded execution."""

import base64
import time
from pathlib import Path

from .browser import Browser, StalePage
from .model import action_space, choose, field_context, field_text
from .questions import MAX_STEPS, PLAN_SATISFIED


class Agent:
    def __init__(self, url, goals, *, record_dir=None, screenshots=False, track_plan=False, reuse_held=True):
        steps = [goals] if isinstance(goals, str) else list(goals)
        steps = [step.strip() for step in steps if step and step.strip()]
        task = "\n".join(steps)
        if not task:
            raise ValueError("Supply a task")
        # Tracking asks per sub-goal, so it needs them kept apart rather than joined into one string.
        self.track_plan = track_plan and len(steps) > 1
        plan = steps if self.track_plan else [task]
        # A sub-goal can be satisfied by an action aimed at another, so this is a set, not an index.
        self.plan_satisfied = set()
        # Decisions taken for later sub-goals, kept until they apply or are overwritten.
        self.held = {}
        self.reuse_held_answers = reuse_held
        self.reused = 0
        self.pending_text = None
        self.browser = Browser(url)
        self.record_dir = Path(record_dir) if record_dir else None
        self.screenshots = screenshots or bool(record_dir)
        try:
            page = self.browser.settle(screenshot=self.screenshots)
        except Exception:
            self.browser.close()
            raise
        self.state = dict(
            browser=self.browser,
            goal="\n".join(plan),
            page=page,
            decision=None,
            history=[],
            status="ready",
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

    def snapshot(self):
        return {
            **{k: v for k, v in self.state.items() if k != "browser"},
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
                state["page"] = state["browser"].settle(screenshot=self.screenshots)
                state["elapsed_ms"] = round((time.perf_counter() - state["started_at"]) * 1000)
                return self.snapshot()
        elif name == "predict":
            if not state["browser"]:
                raise ValueError("Start a demo first")
            if state["started_at"] is None:
                state["started_at"] = time.perf_counter()
            if not state["browser"].fresh(state["page"]):
                state["page"] = state["browser"].settle(screenshot=self.screenshots)
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
            # A held answer that still applies replaces this turn's call. Checking costs one local
            # lookup over the current snapshot; being wrong costs the call that would have happened.
            reused = self.reuse_held(outstanding) if self.reuse_held_answers else None
            if reused:
                state["decision"] = reused
                self.reused += 1
                state["status"] = "predicted"
                return self.snapshot()
            state["decision"] = choose(
                state["page"], state["goal"], state["history"], [state["plan"][i] for i in outstanding]
            )
            # These readings describe the page this call saw, so a sub-goal retires in the same call
            # that measured it. Retiring is one-way: the question is not asked again.
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
                if not state["browser"].fresh(page):
                    state["status"] = "ready"
                    raise StalePage("Page changed since the decision. Choose again.")
                state["status"] = "done" if selected == "DONE" else "blocked"
                state["plan_index"] = int(selected == "DONE")
                state["elapsed_ms"] = round((time.perf_counter() - state["started_at"]) * 1000)
                return self.snapshot()
            action = next(a for a in page["actions"] if a["id"] == selected)
            if len(state["history"]) >= MAX_STEPS:
                state["status"] = "blocked"
                raise ValueError(f"Stopped at the {MAX_STEPS}-action demo budget")
            text, helper = None, None
            if action["kind"] == "fill":
                if not state["browser"].fresh(page):
                    raise StalePage("Page changed before text generation. Choose again.")
                context = field_context(state["goal"], action, page, state["history"])
                if self.pending_text and self.pending_text[0] == context:
                    _, text, helper = self.pending_text
                else:
                    text, helper = field_text(context)
                    self.pending_text = (context, text, helper)
                    state["text_calls"].append({**helper, "field": action["label"], "value": text})
            # Browser.act checks freshness immediately before input, including after text generation.
            state["browser"].act(action, page, text=text)
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
            state["page"] = state["browser"].settle(screenshot=self.screenshots)
            state["elapsed_ms"] = round((time.perf_counter() - state["started_at"]) * 1000)
            state["history"][-1].update(
                page_changed=state["page"]["fingerprint"] != page["fingerprint"],
                url=state["page"]["url"],
                elapsed_ms=state["elapsed_ms"],
            )
            if state["record"]:
                (self.record_dir / f"{state['elapsed_ms']:06d}.jpg").write_bytes(
                    base64.b64decode(state["page"]["screenshot"])
                )
            repeated = state["history"][-3:]
            state["status"] = (
                "blocked"
                if len(repeated) == 3 and all(h["page_changed"] is False and h["kind"] != "wait" for h in repeated)
                else "ready"
            )
        else:
            raise ValueError("Unknown command")
        return self.snapshot()

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
            # One reuse per call: satisfaction readings are only refreshed when Jev is asked, so
            # after acting on a held answer nothing knows which sub-goals are still outstanding.
            self.held = {}
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
        self.browser.close()

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self.close()
