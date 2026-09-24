"""Offline contracts for a dynamic operation/target policy. No paid APIs."""

import importlib
import json
import time
from copy import deepcopy
from datetime import date
from unittest.mock import Mock

import pytest

from jev_ultrafast import agent as loop
from jev_ultrafast import browser as browser_module
from jev_ultrafast import model
from jev_ultrafast import replay as replay_module
from jev_ultrafast import session as session_module
from jev_ultrafast.browser import StalePage, browser_operation, fingerprint


def page():
    state = {
        "url": "https://example.test/",
        "title": "Search",
        "text": "Search",
        "scroll": {"y": 0},
        "actions": [
            {"id": "e1", "kind": "fill", "label": "Search", "role": "textbox", "value": "", "node": 10},
            {"id": "e2", "kind": "click", "label": "Open Search", "role": "textbox", "value": "", "node": 10},
            {"id": "e3", "kind": "click", "label": "Go", "role": "button", "value": "", "node": 20},
            {"id": "wait", "kind": "wait", "label": "Wait"},
        ],
    }
    state["fingerprint"] = fingerprint(state)
    return state


def choice(ids, selected):
    return {"choice": selected, "confidence": 1.0, "probabilities": {i: float(i == selected) for i in ids}}


def decision(action="e1"):
    return {
        "choice": action,
        "operation": "TYPE_TEXT",
        "target": "1",
        "confidence": 1.0,
        "probabilities": {action: 1.0},
        "latency_ms": 10,
        "usage": {},
    }


@pytest.mark.parametrize("mutation", ["unknown", "nan", "missing", "negative", "non_max", "confidence"])
def test_invalid_choice_is_rejected(mutation):
    a = choice(["a", "b"], "a")
    if mutation == "unknown":
        a["choice"] = "invented"
    elif mutation == "nan":
        a["probabilities"]["a"] = float("nan")
    elif mutation == "missing":
        del a["probabilities"]["b"]
    elif mutation == "negative":
        a["probabilities"]["b"] = -1
    elif mutation == "non_max":
        a["choice"] = "b"
    else:
        a["confidence"] = 5
    with pytest.raises(ValueError, match="Invalid TypeSafe"):
        model.validate_choice(a, {"a", "b"})


def test_one_index_per_node_with_operation_specific_targets():
    elements, targets, controls = model.action_space(page()["actions"])
    assert len(elements) == 2
    assert elements[0]["operations"] == ["TYPE_TEXT", "CLICK"]
    assert targets["TYPE_TEXT"]["1"]["id"] == "e1"
    assert targets["CLICK"]["1"]["id"] == "e2"
    assert targets["CLICK"]["2"]["id"] == "e3"
    assert "WAIT" in controls


def test_all_heads_are_one_request_and_only_matching_head_executes(monkeypatch):
    calls = []

    def post(_url, _key, body):
        calls.append(body)
        return {
            "model": "test",
            "answers": {
                "operation": choice(body["questions"]["operation"]["criteria"], "TYPE_TEXT"),
                "type_text_target": choice(["1"], "1"),
                "click_target": {"choice": "invented"},
            },
        }

    monkeypatch.setenv("TYPESAFE_API_KEY", "test")
    monkeypatch.setattr(model, "post_json", post)
    d = model.choose(page(), "Find a book", [])
    assert len(calls) == 1
    assert d["operation"] == "TYPE_TEXT" and d["target"] == "1" and d["choice"] == "e1"
    assert set(calls[0]["questions"]) == {"operation", "click_target"}   # type_text_target had one candidate


def test_click_cannot_consume_a_text_target(monkeypatch):
    def post(_url, _key, body):
        return {
            "model": "test",
            "answers": {
                "operation": choice(body["questions"]["operation"]["criteria"], "CLICK"),
                "type_text_target": choice(["1"], "1"),
                "click_target": choice(["1", "2", "999"], "999"),
            },
        }

    monkeypatch.setenv("TYPESAFE_API_KEY", "test")
    monkeypatch.setattr(model, "post_json", post)
    with pytest.raises(ValueError, match="Invalid TypeSafe"):
        model.choose(page(), "Find a book", [])


def test_target_head_receives_control_state_and_full_next_step_rules(monkeypatch):
    p = page()
    p["actions"].insert(0, {
        "id": "toggle", "kind": "click", "label": "Free cancellation", "node": 30,
        "role": "checkbox", "checked": "true", "selected": False,
    })

    def post(_url, _key, body):
        questions = body["questions"]
        target = questions["click_target"]
        # Criteria and instructions are text, as the spec takes them; the control state and the
        # operation rules still have to reach the target head, now inside those strings.
        assert "checked: true" in target["criteria"]["1"]
        assert "selected: False" in target["criteria"]["1"]
        assert model.NEXT_ACTION in questions["operation"]["instructions"]
        assert model.NEXT_ACTION in target["instructions"]
        return {
            "model": "test",
            "answers": {
                "operation": choice(questions["operation"]["criteria"], "CLICK"),
                "click_target": choice(target["criteria"], "3"),
            },
        }

    monkeypatch.setenv("TYPESAFE_API_KEY", "test")
    monkeypatch.setattr(model, "post_json", post)
    d = model.choose(p, "Search with free cancellation", [])
    assert d["choice"] == "e3"


def test_quoted_task_text_still_uses_the_llm(monkeypatch):
    monkeypatch.setenv("TEXT_MODEL_API_KEY", "test")
    post = Mock(return_value={"choices": [{"message": {"content": '{"text":"Zurich"}'}}]})
    monkeypatch.setattr(model, "post_json", post)
    context = model.field_context('Fly from "Zurich" to London', page()["actions"][0], page(), [])
    assert model.field_text(context)[0] == "Zurich"
    assert post.call_count == 1
    sent = json.loads(post.call_args.args[2]["messages"][1]["content"])
    assert sent["goal"] == 'Fly from "Zurich" to London'


def test_missing_text_credential_stops_before_guessing(monkeypatch):
    monkeypatch.delenv("TEXT_MODEL_API_KEY", raising=False)
    with pytest.raises(ValueError, match="TEXT_MODEL_API_KEY"):
        model.field_text({"goal": 'Enter "Zurich"'})


@pytest.fixture
def runner():
    a = loop.Agent.__new__(loop.Agent)
    a.screenshots = False
    a.pending_text = None
    p = page()
    a.state = {
        "browser": Mock(fresh=Mock(return_value=True), observe=Mock(return_value=p), settle=Mock(return_value=p)),
        "page": p,
        "decision": decision(),
        "goal": "Find a book",
        "history": [],
        "decisions": [],
        "status": "predicted",
        "started_at": time.perf_counter(),
        "record": False,
        "text_calls": [],
    }
    return a


def test_stale_decision_is_consumed_before_any_mutation(runner):
    runner.state["browser"].fresh.return_value = False
    with pytest.raises(StalePage):
        runner.command("act", {"fingerprint": runner.state["page"]["fingerprint"]})
    runner.state["browser"].act.assert_not_called()
    assert runner.state["decision"] is None


def test_generated_text_reused_only_for_identical_retry_context(runner, monkeypatch):
    helper = Mock(return_value=("book", {"model": "test", "latency_ms": 10}))
    monkeypatch.setattr(loop, "field_text", helper)
    runner.state["browser"].act.side_effect = [StalePage("Changed before input"), None]
    with pytest.raises(StalePage):
        runner.command("act", {"fingerprint": runner.state["page"]["fingerprint"]})
    runner.state["decision"] = decision()
    runner.command("act", {"fingerprint": runner.state["page"]["fingerprint"]})
    assert helper.call_count == 1
    assert runner.state["browser"].act.call_count == 2  # The first call rejects before any browser input.
    assert runner.pending_text is None


def test_changed_field_context_does_not_reuse_generated_text(runner, monkeypatch):
    helper = Mock(return_value=("book", {"model": "test", "latency_ms": 10}))
    monkeypatch.setattr(loop, "field_text", helper)
    runner.state["browser"].act.side_effect = [StalePage("Changed before input"), None]
    with pytest.raises(StalePage):
        runner.command("act", {"fingerprint": runner.state["page"]["fingerprint"]})
    runner.state["page"]["text"] = "Different page context"
    runner.state["decision"] = decision()
    runner.command("act", {"fingerprint": runner.state["page"]["fingerprint"]})
    assert helper.call_count == 2


def test_loading_waits_do_not_trigger_no_progress_stop(runner):
    for _ in range(5):
        runner.state["decision"] = decision("wait")
        runner.command("act", {"fingerprint": runner.state["page"]["fingerprint"]})
    assert len(runner.state["history"]) == 5 and runner.state["status"] == "ready"


def test_stale_observation_preserves_executed_action(runner):
    runner.state["decision"] = decision("e3")
    runner.state["browser"].settle.side_effect = StalePage("changed")
    with pytest.raises(StalePage):
        runner.command("act", {"fingerprint": runner.state["page"]["fingerprint"]})
    assert runner.state["history"][-1]["action"] == "Go"
    runner.state["browser"].act.assert_called_once()


def test_observation_is_one_atomic_browser_read(monkeypatch):
    import jev_ultrafast.browser as browser

    p = page()
    cdp = Mock(return_value={"result": {"value": p}})
    monkeypatch.setattr(browser, "cdp", cdp)
    actual = browser_operation({"operation": "observe", "session": "test", "screenshot": False})
    assert actual["actions"] == p["actions"]
    assert cdp.call_count == 1
    assert cdp.call_args.args[0] == "Runtime.evaluate"


def test_executor_rejects_a_stale_page_before_browser_input(monkeypatch):
    import jev_ultrafast.browser as browser

    b = browser.Browser.__new__(browser.Browser)
    b.fresh = Mock(return_value=False)
    operation = Mock()
    monkeypatch.setattr(browser, "browser_operation", operation)
    with pytest.raises(StalePage):
        b.act(page()["actions"][0], page(), "book")
    operation.assert_not_called()


@pytest.mark.parametrize("response", [{"exceptionDetails": {}}, {"result": {}}])
def test_interrupted_dropdown_mutation_cannot_be_retried_as_stale(monkeypatch, response):
    import jev_ultrafast.browser as browser

    # A navigation can destroy the evaluation result after the change event already fired.
    if "exceptionDetails" in response:
        response["exceptionDetails"] = {"text": "Execution context destroyed"}
    cdp = Mock(return_value=response)
    monkeypatch.setattr(browser, "cdp", cdp)
    with pytest.raises(RuntimeError, match="Dropdown execution"):
        browser_operation({"operation": "act", "session": "test", "action": {
            "id": "e1", "kind": "select", "node": 1, "value": "Design",
        }})
    assert cdp.call_count == 1


def test_fingerprint_tracks_values_and_identity_not_screenshots():
    p = page()
    other = deepcopy(p)
    other["screenshot"] = "changed"
    assert fingerprint(p) == fingerprint(other)
    other["actions"][0]["node"] = 99
    assert fingerprint(p) != fingerprint(other)


@pytest.mark.parametrize("changed", ["Departure", "Where from?", "Where to?", "year"])
def test_flight_verification_rejects_wrong_trip(changed):
    from examples.flights import verify

    # The task's date rolls forward, so the fixture names its own day and verify() is told which
    # day to check. Otherwise this test would pass only while the fixture matched today + 30.
    departure = date(2026, 9, 20)
    actual = {
        "url": "https://www.google.com/travel/flights/search?tfs=example",
        "text": "Track prices from Zürich to London departing 2026-09-20",
        "actions": [
            {"label": k, "value": v}
            for k, v in [
                ("Change ticket type. One way", "One way"),
                ("Where from?", "Zürich"),
                ("Where to?", "London"),
                ("Departure", "Sun, Sep 20"),
                ("Nonstop flight on Sunday, September 20. Select flight", ""),
            ]
        ],
    }
    assert verify(actual, departure)["passed"]
    if changed == "year":
        actual["text"] = actual["text"].replace("2026", "2027")
    else:
        next(a for a in actual["actions"] if a["label"] == changed)["value"] = "wrong"
    assert not verify(actual, departure)["passed"]


@pytest.mark.parametrize(
    "content", ["Thinking: Zurich", '{"text":null}', '{"text":"Zurich","extra":true}', '{"text":123}']
)
def test_text_helper_rejects_invalid_values(monkeypatch, content):
    monkeypatch.setenv("TEXT_MODEL_API_KEY", "test")
    monkeypatch.setattr(model, "post_json", Mock(return_value={"choices": [{"message": {"content": content}}]}))
    with pytest.raises(ValueError, match="nothing typed"):
        model.field_text({"goal": "Find a flight"})


def test_navigation_during_prediction_reobserves_without_action(runner):
    runner.state["browser"].fresh.side_effect = StalePage("Document navigating")
    runner.command("tick")
    assert runner.state["status"] == "ready"
    assert runner.state["decision"] is None
    runner.state["browser"].act.assert_not_called()


def test_plan_asks_a_check_and_a_decision_per_outstanding_goal():
    _, targets, _ = model.action_space(page()["actions"])
    operations = {"CLICK": "click", "TYPE_TEXT": "type", "DONE": "done", "BLOCKED": "blocked"}
    questions = model.plan_questions(["Set the origin", "Run the search"], operations, targets, {})
    assert {"plan0_satisfied", "plan1_satisfied"} <= set(questions)
    assert questions["plan0_satisfied"]["type"] == "noul"
    assert "Set the origin" in questions["plan0_satisfied"]["instructions"]
    # Each outstanding goal also gets a decision, so its answer can replace a later call.
    assert {"plan0_operation", "plan0_click_target", "plan1_operation"} <= set(questions)


@pytest.mark.parametrize(
    "answer,expected", [({"noul": 0.95}, 0.95), ({"noul": None}, None), ({}, None), ({"noul": 2}, None)]
)
def test_unreadable_satisfaction_is_dropped_not_raised(answer, expected):
    read = model.read_plan_answers({"plan0_satisfied": answer}, ["a"], {}, {}, {})
    assert read[0]["satisfied"] == expected
    assert read[0]["label"] is None  # no readable decision either, and that must not raise


def test_a_fully_satisfied_plan_ends_the_run_without_acting(runner):
    runner.track_plan = True
    runner.plan_satisfied = {0, 1}
    runner.state["plan"] = ["Set the origin", "Run the search"]
    out = runner.command("act", {"fingerprint": runner.state["page"]["fingerprint"]})
    assert out["status"] == "done"
    runner.state["browser"].act.assert_not_called()


def history_step(label, changed, kind="click"):
    return {"action": label, "kind": kind, "page_changed": changed}


@pytest.mark.parametrize(
    "steps,expected",
    [
        ([history_step("Open Return", False), history_step("Open Return", False)], {("Open Return", "click")}),
        ([history_step("Open Return", True), history_step("Open Return", False)], set()),
        ([history_step("Open Return", False), history_step("Search", False)], set()),
        ([history_step("Wait", False, "wait"), history_step("Wait", False, "wait")], set()),
    ],
)
def test_a_control_chosen_twice_without_effect_is_fixation(runner, steps, expected):
    runner.state["history"] = steps
    assert runner.fixated() == expected


def test_a_suppressed_control_cannot_be_chosen(monkeypatch):
    def post(_url, _key, body):
        answers = {"operation": choice(body["questions"]["operation"]["criteria"], "CLICK")}
        target = body["questions"].get("click_target")
        if target:  # one remaining candidate is taken directly, so the head may not be asked
            answers["click_target"] = choice(target["criteria"], next(iter(target["criteria"])))
        return {"model": "test", "answers": answers}

    monkeypatch.setenv("TYPESAFE_API_KEY", "test")
    monkeypatch.setattr(model, "post_json", post)
    # "Go" is e3; withholding it must leave the other click, never the suppressed one.
    assert model.choose(page(), "Find a book", [], (), {("Go", "click")})["choice"] == "e2"


def test_suppression_never_empties_the_action_space(monkeypatch):
    sent = {}

    def post(_url, _key, body):
        sent.update(body)
        return {
            "model": "test",
            "answers": {
                "operation": choice(body["questions"]["operation"]["criteria"], "CLICK"),
                "click_target": choice(body["questions"]["click_target"]["criteria"], "1"),
            },
        }

    monkeypatch.setenv("TYPESAFE_API_KEY", "test")
    monkeypatch.setattr(model, "post_json", post)
    everything = {("Open Search", "click"), ("Go", "click"), ("Search", "fill")}
    model.choose(page(), "Find a book", [], (), everything)
    assert sent["questions"]["click_target"]["criteria"]  # fell back instead of stranding the run


def test_an_unusable_field_value_is_retried_not_fatal(runner, monkeypatch):
    # The helper is called before any input is sent, so re-deciding is not a mutation retry.
    monkeypatch.setattr(loop, "field_text", Mock(side_effect=ValueError("no valid field value")))
    runner.text_failures = 0
    with pytest.raises(StalePage):
        runner.command("act", {"fingerprint": runner.state["page"]["fingerprint"]})
    runner.state["browser"].act.assert_not_called()
    assert runner.state["status"] == "ready"


def test_a_persistently_unusable_field_value_stops_the_run(runner, monkeypatch):
    monkeypatch.setattr(loop, "field_text", Mock(side_effect=ValueError("no valid field value")))
    runner.text_failures = 99
    with pytest.raises(ValueError, match="no valid field value"):
        runner.command("act", {"fingerprint": runner.state["page"]["fingerprint"]})


@pytest.mark.parametrize(
    "content,expected",
    [
        ('{"text":"Zurich"}', "Zurich"),
        ('```json\n{"text":"Zurich"}\n```', "fenced"),
        ('Here you go: {"text":"Zurich"}', "prose before"),
        ('{"text":"Zurich"}\nHope that helps.', "prose after"),
    ],
)
def test_a_wrapped_json_object_is_still_read(monkeypatch, content, expected):
    monkeypatch.setenv("TEXT_MODEL_API_KEY", "test")
    monkeypatch.setattr(model, "post_json", Mock(return_value={"choices": [{"message": {"content": content}}]}))
    assert model.field_text({"goal": "x"})[0] == "Zurich", expected


def test_the_rejected_content_is_named_in_the_error(monkeypatch):
    monkeypatch.setenv("TEXT_MODEL_API_KEY", "test")
    monkeypatch.setattr(model, "post_json", Mock(return_value={"choices": [{"message": {"content": "I cannot"}}]}))
    with pytest.raises(ValueError, match="I cannot"):
        model.field_text({"goal": "x"})


def test_a_reused_fill_is_told_which_sub_goal_it_is_for(runner, monkeypatch):
    helper = Mock(return_value=("Zurich", {"model": "test", "latency_ms": 1}))
    monkeypatch.setattr(loop, "field_text", helper)
    runner.state["plan"] = ["Set the origin", "Set the destination"]
    runner.state["goal"] = "Set the origin\nSet the destination"
    runner.state["decision"] = {**decision(), "reused_for": 1}
    runner.command("act", {"fingerprint": runner.state["page"]["fingerprint"]})
    # The helper must see the sub-goal the held decision was taken for, not the whole task.
    assert helper.call_args.args[0]["goal"] == "Set the destination"


def test_field_values_asks_once_for_several_fields(monkeypatch):
    monkeypatch.setenv("TEXT_MODEL_API_KEY", "test")
    post = Mock(return_value={"choices": [{"message": {"content": '{"a":"Ada","b":"Lovelace"}'}}]})
    monkeypatch.setattr(model, "post_json", post)
    fields = {
        "a": ("Set the first name to Ada", {"label": "First Name", "role": "textbox", "value": ""}),
        "b": ("Set the last name to Lovelace", {"label": "Last Name", "role": "textbox", "value": ""}),
    }
    values, helper = model.field_values(fields, page(), [])
    assert values == {"a": "Ada", "b": "Lovelace"}
    assert post.call_count == 1  # one call, not one per field
    assert helper["fields"] == 2


@pytest.mark.parametrize(
    "content,kept",
    [
        ('{"a":"Ada","b":null}', {"a": "Ada"}),
        ('{"a":"Ada","b":"  "}', {"a": "Ada"}),
        ('{"a":"Ada","unasked":"x"}', {"a": "Ada"}),
        ("not json at all", {}),
    ],
)
def test_a_value_that_does_not_hold_up_is_left_out(monkeypatch, content, kept):
    monkeypatch.setenv("TEXT_MODEL_API_KEY", "test")
    monkeypatch.setattr(model, "post_json", Mock(return_value={"choices": [{"message": {"content": content}}]}))
    fields = {
        "a": ("goal a", {"label": "A", "role": "textbox", "value": ""}),
        "b": ("goal b", {"label": "B", "role": "textbox", "value": ""}),
    }
    # A field left out here falls back to its own call instead of being filled with a guess.
    assert model.field_values(fields, page(), [])[0] == kept


def observation(url="https://example.test/", actions=10, expanded=0):
    acts = [{"id": f"e{i}", "kind": "click", "label": f"c{i}", "node": i} for i in range(actions)]
    for a in acts[:expanded]:
        a["expanded"] = "true"
    return {"url": url, "actions": acts}


@pytest.mark.parametrize(
    "before,after,expected,why",
    [
        (observation(), observation(url="https://example.test/next"), True, "navigation"),
        (observation(expanded=0), observation(expanded=1), True, "a menu or dialog opened"),
        (observation(actions=10), observation(actions=14), True, "what is on offer changed a lot"),
        (observation(actions=10), observation(actions=12), False, "a small change is not arrival"),
        (observation(), observation(), False, "nothing changed"),
    ],
)
def test_only_arriving_pages_are_waited_for(before, after, expected, why):
    assert browser_module.unsettling(before, after) is expected, why


def watched_browser(monkeypatch, events, started=True):
    b = browser_module.Browser.__new__(browser_module.Browser)
    b.session = "test"
    calls = []

    def call(method, **params):
        calls.append(method)
        if method == "Page.startScreencast" and not started:
            raise RuntimeError("no screencast here")
        return {}

    b.call = call
    monkeypatch.setattr(browser_module, "drain_events", lambda: events.pop(0) if events else [])
    return b, calls


def frame():
    return [{"method": "Page.screencastFrame", "params": {"sessionId": 1}}]


def test_a_screen_that_stops_painting_settles_the_page(monkeypatch):
    b, calls = watched_browser(monkeypatch, [frame(), [], [], [], [], [], [], [], []])
    settled = page()
    b.observe = lambda screenshot=False, frames=True: settled
    monkeypatch.setattr(browser_module, "SETTLE_PAINT", 0.02)
    assert b.still_screen(False, time.monotonic() + 2) is settled
    assert "Page.screencastFrameAck" in calls


def test_a_screen_that_never_reports_is_left_to_the_document(monkeypatch):
    b, _ = watched_browser(monkeypatch, [], started=False)
    b.observe = lambda screenshot=False, frames=True: page()
    assert b.still_screen(False, time.monotonic() + 2) is None
    assert b.watching_paint() is False


def test_a_screen_that_never_paints_is_asked_once_and_then_left_alone(monkeypatch):
    b, _ = watched_browser(monkeypatch, [])
    b.observe = lambda screenshot=False, frames=True: page()
    began = time.monotonic()
    assert b.still_screen(False, began + 5) is None
    # Given up on quickly whatever the watching budget is, and not retried on the next wait.
    assert time.monotonic() - began < browser_module.SETTLE_WATCH / 2
    assert b.screencast is False


def test_a_screen_that_never_stops_painting_is_left_to_the_document(monkeypatch):
    b, _ = watched_browser(monkeypatch, [])
    monkeypatch.setattr(browser_module, "drain_events", frame)
    b.observe = lambda screenshot=False, frames=True: page()
    monkeypatch.setattr(browser_module, "SETTLE_WATCH", 0.2)
    began = time.monotonic()
    assert b.still_screen(False, began + 5) is None
    assert time.monotonic() - began >= 0.2


def test_only_a_frame_counts_as_the_screen_having_painted(monkeypatch):
    other = {"method": "Network.responseReceived", "params": {}}
    b, calls = watched_browser(monkeypatch, [[other], [other] + frame()])
    b.observe = lambda screenshot=False, frames=True: page()
    # A drain that found no frame leaves the page never having painted, and acks nothing.
    assert b.painted() is None
    assert calls == []
    assert b.painted() is not None
    assert calls == ["Page.screencastFrameAck"]


def test_a_report_left_running_by_an_earlier_reader_is_accepted(monkeypatch):
    b, _ = watched_browser(monkeypatch, [])

    def refuse(method, **params):
        raise RuntimeError({"code": -32000, "message": "Screencast is already active"})

    b.call = refuse
    assert b.watching_paint() is True


def test_closing_the_page_stops_it_reporting(monkeypatch):
    b, calls = watched_browser(monkeypatch, [])
    b.screencast, b.target = True, None
    monkeypatch.setenv("JEV_KEEP_OPEN", "0")
    b.close()
    assert "Page.stopScreencast" in calls


def restless_agent(fresh_answers):
    runner = loop.Agent.__new__(loop.Agent)
    runner.state = {"browser": Mock(fresh=Mock(side_effect=fresh_answers))}
    runner.unfresh_since = None
    return runner


def test_a_page_that_holds_still_is_never_called_restless():
    runner = restless_agent([True, True])
    assert runner.settled_or_restless(page()) is True
    assert runner.restless() is False
    assert runner.unfresh_since is None


def test_a_page_that_keeps_changing_is_acted_on_once_waiting_is_futile(monkeypatch):
    monkeypatch.setattr(loop, "RESTLESS", 0.05)
    runner = restless_agent([False, False, False])
    # The first refusal starts the clock instead of giving up on the spot.
    assert runner.settled_or_restless(page()) is False
    time.sleep(0.06)
    assert runner.settled_or_restless(page()) is True
    assert runner.restless() is True


def test_holding_still_again_clears_the_restless_clock(monkeypatch):
    monkeypatch.setattr(loop, "RESTLESS", 0.05)
    runner = restless_agent([False, True, False])
    runner.settled_or_restless(page())
    runner.settled_or_restless(page())
    assert runner.unfresh_since is None
    time.sleep(0.06)
    # A fresh refusal starts a fresh clock; earlier unsettledness is not carried over.
    assert runner.settled_or_restless(page()) is False


def harness_daemon():
    """The installed daemon module, or a skip when it has not been taught about headers."""
    daemon = pytest.importorskip("browser_harness.daemon")
    if not hasattr(daemon, "cdp_headers"):
        pytest.skip("browser-harness is unpatched; run scripts/patch_browser_harness.py")
    return daemon


def test_no_setting_sends_no_extra_headers(monkeypatch):
    daemon = harness_daemon()
    monkeypatch.delenv("BU_CDP_HEADERS", raising=False)
    assert daemon.cdp_headers() == {}


def test_a_json_object_becomes_the_headers(monkeypatch):
    daemon = harness_daemon()
    monkeypatch.setenv("BU_CDP_HEADERS", '{"Authorization": "Bearer t", "X-Count": 2}')
    # Values are sent as text, so a number in the JSON is not passed through as one.
    assert daemon.cdp_headers() == {"Authorization": "Bearer t", "X-Count": "2"}


@pytest.mark.parametrize("setting", ["not json", "[1, 2]", '"a string"', "   "])
def test_a_setting_that_is_not_a_json_object_sends_nothing(monkeypatch, setting):
    daemon = harness_daemon()
    monkeypatch.setenv("BU_CDP_HEADERS", setting)
    # Nothing is invented from a malformed value: the endpoint then refuses the handshake, which
    # names the endpoint, where a half-built header would name nothing.
    assert daemon.cdp_headers() == {}


def recorded_state():
    return {
        "url": "https://example.test/start",
        "goal": "Fill the form",
        "history": [
            {"step": 1, "action": "Search", "kind": "fill", "text": "Ada", "url": "https://example.test/start",
             "choice": "0:e1", "probability": 0.9},
            {"step": 2, "action": "Go", "kind": "click", "text": None, "url": "https://example.test/start"},
        ],
    }


def test_a_script_names_controls_and_keeps_no_element_ids():
    document = replay_module.script(recorded_state(), name="demo")
    assert document["url"] == "https://example.test/start"
    assert document["steps"] == [
        {"kind": "fill", "label": "Search", "text": "Ada"},
        {"kind": "click", "label": "Go"},
    ]
    # A node id means nothing on a page loaded again, so nothing carries one.
    assert "choice" not in json.dumps(document) and "e1" not in json.dumps(document)


def test_a_script_falls_back_to_the_first_page_a_run_acted_on():
    state = recorded_state()
    del state["url"]
    assert replay_module.script(state)["url"] == "https://example.test/start"


def test_a_replay_resolves_each_step_against_the_page_in_front_of_it():
    acted, pages = [], [page(), page(), page()]
    browser = Mock(
        settle=Mock(side_effect=lambda **kw: pages.pop(0) if pages else page()),
        act=Mock(side_effect=lambda action, page, text=None: acted.append((action["id"], text))),
    )
    document = replay_module.script(recorded_state())
    document["steps"] = [{"kind": "fill", "label": "Search", "text": "Ada"}, {"kind": "click", "label": "Go"}]
    result = replay_module.replay(document, browser=browser)
    assert acted == [("e1", "Ada"), ("e3", None)]
    assert result["status"] == "done" and result["completed"] == 2
    assert [step["step"] for step in result["steps"]] == [1, 2]
    browser.close.assert_not_called()  # a borrowed browser is left open


def test_a_replay_stops_when_a_step_names_nothing_on_the_page():
    browser = Mock(settle=Mock(return_value=page()))
    document = {"version": replay_module.VERSION, "url": "https://example.test/",
                "steps": [{"kind": "click", "label": "Nowhere"}]}
    with pytest.raises(ValueError, match="matched 0 controls"):
        replay_module.replay(document, browser=browser)
    browser.act.assert_not_called()


def test_a_replay_stops_when_a_step_names_more_than_one_control():
    twice = page()
    twice["actions"].append({**twice["actions"][2], "id": "e9"})
    browser = Mock(settle=Mock(return_value=twice))
    document = {"version": replay_module.VERSION, "url": "https://example.test/",
                "steps": [{"kind": "click", "label": "Go"}]}
    with pytest.raises(ValueError, match="matched 2 controls"):
        replay_module.replay(document, browser=browser)
    browser.act.assert_not_called()


def test_a_script_from_another_version_is_refused(tmp_path):
    path = tmp_path / "script.json"
    path.write_text(json.dumps({"version": replay_module.VERSION + 1, "url": "https://x.test/", "steps": []}))
    with pytest.raises(ValueError, match="is not"):
        replay_module.read(path)


def test_a_script_round_trips_through_a_file(tmp_path):
    path = tmp_path / "script.json"
    replay_module.write(recorded_state(), path, name="demo")
    assert replay_module.read(path)["steps"][0]["text"] == "Ada"


def reported_state():
    state = recorded_state()
    state["plan"] = ["Open the contact page", "Find the message field"]
    state["status"] = "blocked"
    state["page"] = {**page(), "title": "Contact us", "scroll": {"y": 1120, "height": 6097}}
    state["decisions"] = [{
        "operation_probabilities": {"CLICK": 0.31, "BLOCKED": 0.53, "SCROLL_DOWN": 0.12},
        # A decision also carries the request it sent, which holds the page text.
        "request": {"state": {"page": {"text": "x" * 6000}}},
    }]
    for step in state["history"]:
        step["page_changed"] = True
        step["scroll"] = {"y": 560, "height": 6097}
    return state


def test_a_report_carries_positions_and_no_page_content():
    document = replay_module.report(reported_state())
    assert document["status"] == "blocked"
    assert document["goals"] == ["Open the contact page", "Find the message field"]
    assert document["steps"][0]["y"] == 560 and document["steps"][0]["height"] == 6097
    assert document["final"]["y"] == 1120 and document["final"]["height"] == 6097
    assert document["operations"]["BLOCKED"] == 0.53
    # The page's own text reaches the model but must never reach a report.
    assert "x" * 100 not in json.dumps(document)
    assert "e1" not in json.dumps(document) and "fingerprint" not in json.dumps(document)


def test_a_report_carries_a_url_only_when_it_changes():
    state = reported_state()
    long_url = "https://example.test/search?tfs=" + "A" * 200
    state["history"][1]["url"] = long_url
    document = replay_module.report(state)
    # The first step happened where the run started, so it says nothing about the url.
    assert "url" not in document["steps"][0]
    assert document["steps"][1]["url"].startswith("https://example.test/search?tfs=A")
    assert len(document["steps"][1]["url"]) <= replay_module.URL_LENGTH


def test_a_report_is_bounded_by_its_caps_not_by_the_page():
    state = reported_state()
    state["history"] = [
        {"action": f"Some control number {n} with a rather long label", "kind": "click",
         "text": None, "url": "https://example.test/a/fairly/long/path?with=query", "page_changed": n % 2 == 0,
         "scroll": {"y": 560 * n, "height": 6097}}
        for n in range(10)
    ]
    state["page"]["actions"] = [
        {"label": f"Control {n} with a long label that should be cut", "kind": "click"} for n in range(80)
    ]
    document = replay_module.report(state)
    assert len(document["final"]["controls"]) == 25
    # A real eleven-step Google Flights report is 2,249 bytes, most of it the steps. This is the
    # same run with every label at its cap and eighty controls offered, so it bounds the shape
    # rather than describing a typical run.
    size = len(json.dumps(document, separators=(",", ":")))
    assert size < 2600, f"a ten-step report should stay bounded, got {size}"
    # Ten times the page, same report: nothing here scales with what the page contains.
    state["page"]["actions"] = state["page"]["actions"] * 10
    state["page"]["text"] = "x" * 60000
    assert len(json.dumps(replay_module.report(state), separators=(",", ":"))) == size


def borrowed_agent(monkeypatch, url, browser):
    monkeypatch.setattr(loop, "Browser", Mock(side_effect=AssertionError("must not open its own browser")))
    return loop.Agent(url, "Do the thing", browser=browser)


def test_an_agent_can_run_in_a_browser_it_was_given(monkeypatch):
    browser = Mock(settle=Mock(return_value=page()))
    runner = borrowed_agent(monkeypatch, None, browser)
    assert runner.browser is browser
    # No url, so it starts wherever the browser already is.
    browser.navigate.assert_not_called()


def test_a_borrowed_browser_outlives_the_agent(monkeypatch):
    browser = Mock(settle=Mock(return_value=page()))
    borrowed_agent(monkeypatch, None, browser).close()
    browser.close.assert_not_called()


def test_a_url_with_a_borrowed_browser_navigates_it(monkeypatch):
    browser = Mock(settle=Mock(return_value=page()))
    borrowed_agent(monkeypatch, "https://example.test/next", browser)
    browser.navigate.assert_called_once_with("https://example.test/next")


def tall_page(y=1120, height=6097, view=780, scrolled=True):
    state = page()
    state["scroll"] = {"y": y, "height": height, "view": view}
    if scrolled:
        state["actions"] = state["actions"] + [
            {"id": "scroll_down", "kind": "scroll", "label": "Scroll down", "delta": 560}
        ]
    return state


def test_a_page_with_most_of_it_unseen_still_has_somewhere_to_go():
    assert round(model.unseen(tall_page()), 2) == 0.69
    assert model.scrolling_still_helps(tall_page(), []) is True


def test_a_page_read_to_the_bottom_has_nowhere_left():
    assert model.unseen(tall_page(y=5317)) == 0.0
    # No way down is offered once the bottom is reached, which is what ends the scrolling.
    assert model.scrolling_still_helps(tall_page(y=5317, scrolled=False), []) is False


def test_a_page_that_grows_as_it_is_read_is_still_worth_scrolling():
    # Every scroll loads more, so the unseen fraction never falls; only a scroll that moves
    # nothing says the page is finished.
    grew = [{"kind": "scroll", "action": "Scroll down", "page_changed": True}]
    assert model.scrolling_still_helps(tall_page(y=4000, height=12000), grew) is True


def test_scrolling_up_is_not_evidence_that_the_page_is_finished():
    upward = [{"kind": "scroll", "action": "Scroll up", "page_changed": False}]
    assert model.scrolling_still_helps(tall_page(), upward) is True


def test_scrolling_that_stopped_moving_the_page_is_not_worth_more():
    history = [{"kind": "scroll", "action": "Scroll down", "page_changed": False}]
    assert model.scrolling_still_helps(tall_page(), history) is False
    moved = [{"kind": "scroll", "action": "Scroll down", "page_changed": True}]
    assert model.scrolling_still_helps(tall_page(), moved) is True


def test_a_page_that_does_not_report_its_size_can_still_be_scrolled():
    # `unseen` needs the viewport and reports nothing without it, but the scrolling rule does not
    # depend on size: a way down that has not been shown to fail is reason enough to take it.
    bare = tall_page()
    del bare["scroll"]["view"]
    assert model.unseen(bare) == 0.0
    assert model.scrolling_still_helps(bare, []) is True


def asked_operations(monkeypatch, state, history):
    """The operation question `choose` would send for this page."""
    sent = {}

    def post(url, key, body):
        sent.update(body)
        offered = body["questions"]["operation"]["criteria"]
        return {"model": "test", "answers": {
            # Whatever is on offer; this asks what was offered, not what was picked.
            "operation": choice(offered, next(iter(offered))),
            "type_text_target": choice(["1"], "1"),
            "click_target": choice(["1", "2"], "1"),
        }}

    monkeypatch.setenv("TYPESAFE_API_KEY", "test")
    monkeypatch.setattr(model, "post_json", post)
    model.choose(state, "Find the contact form", history)
    return sent["questions"]["operation"]


def test_blocked_is_withheld_while_the_page_still_has_somewhere_to_go(monkeypatch):
    operations = asked_operations(monkeypatch, tall_page(), [])
    assert "BLOCKED" not in operations["criteria"]
    assert "69% of this page is below the viewport" in operations["instructions"]


def test_blocked_returns_once_scrolling_stops_working(monkeypatch):
    stuck = [{"kind": "scroll", "action": "Scroll down", "page_changed": False}]
    operations = asked_operations(monkeypatch, tall_page(), stuck)
    assert "BLOCKED" in operations["criteria"]


def test_blocked_stays_available_on_a_page_with_nothing_below(monkeypatch):
    operations = asked_operations(monkeypatch, tall_page(y=5317, scrolled=False), [])
    assert "BLOCKED" in operations["criteria"]


def test_a_read_only_run_is_offered_no_way_to_write():
    actions = [
        {"id": "e1", "kind": "fill", "label": "Search", "node": 1},
        {"id": "e2", "kind": "select", "label": "Country", "node": 2},
        {"id": "e3", "kind": "click", "label": "Submit", "node": 3},
        {"id": "e4", "kind": "click", "label": "Send message", "node": 4},
        {"id": "e5", "kind": "click", "label": "About us", "node": 5},
        {"id": "e6", "kind": "scroll", "label": "Scroll down", "delta": 560},
    ]
    kept = [a["label"] for a in model.readable_only(actions)]
    assert kept == ["About us", "Scroll down"]


def test_a_submitting_control_is_recognised_by_its_type_as_well_as_its_words():
    assert model.submits({"kind": "click", "label": "Read more", "type": "submit"}) is True
    assert model.submits({"kind": "click", "label": "Continue"}) is True
    assert model.submits({"kind": "click", "label": "Our continuing story"}) is False


def test_a_host_outside_the_allowlist_is_refused():
    runner = loop.Agent.__new__(loop.Agent)
    runner.allowed_hosts = ("example.test",)
    assert runner.off_site({"url": "https://example.test/a"}) is False
    assert runner.off_site({"url": "https://www.example.test/a"}) is False
    assert runner.off_site({"url": "https://example.test.evil.com/a"}) is True
    assert runner.off_site({"url": "https://other.test/a"}) is True


def test_no_allowlist_allows_everywhere():
    runner = loop.Agent.__new__(loop.Agent)
    runner.allowed_hosts = ()
    assert runner.off_site({"url": "https://anywhere.test/"}) is False


def test_faults_are_deduplicated_and_capped():
    browser = browser_module.Browser.__new__(browser_module.Browser)
    browser.faults = {}
    for _ in range(3):
        browser.note_fault({"method": "Log.entryAdded",
                            "params": {"entry": {"level": "error", "text": "same", "url": "u"}}})
    for n in range(40):
        browser.note_fault({"method": "Network.responseReceived",
                            "params": {"response": {"status": 404, "url": f"https://x.test/{n}"}}})
    browser.note_fault({"method": "Network.responseReceived",
                        "params": {"type": "Document", "response": {"status": 503, "url": "https://x.test/"}}})
    browser.note_fault({"method": "Log.entryAdded", "params": {"entry": {"level": "info", "text": "quiet"}}})
    found = browser_module.diagnosis(browser.faults)
    assert len(found["console"]) == 1
    assert len(found["requests"]) == browser_module.FAULTS_KEPT
    assert found["document_status"] == 503


def test_a_run_that_reaches_its_budget_says_so_rather_than_raising(runner):
    runner.max_steps = 1
    runner.state["history"] = [{"step": 1, "action": "Go", "kind": "click", "page_changed": True}]
    got = runner.command("act", {"fingerprint": runner.state["page"]["fingerprint"]})
    assert got["status"] == "budget"
    # The action that would have gone over the budget was not taken.
    assert len(got["history"]) == 1


def test_a_local_connection_keeps_the_screen_and_the_longer_wait(monkeypatch):
    monkeypatch.delenv("BU_CDP_WS", raising=False)
    reloaded = importlib.reload(browser_module)
    try:
        assert reloaded.REMOTE is False
        assert reloaded.SETTLE_TIMEOUT == 6
    finally:
        importlib.reload(browser_module)


def test_a_remote_connection_waits_in_the_page_and_does_not_screencast(monkeypatch):
    monkeypatch.setenv("BU_CDP_WS", "wss://example.test/devtools/browser")
    reloaded = importlib.reload(browser_module)
    try:
        assert reloaded.REMOTE is True
        # Bounded harder, because each look is a round trip rather than a pipe.
        assert reloaded.SETTLE_TIMEOUT == 1.5
        b = reloaded.Browser.__new__(reloaded.Browser)
        assert b.watching_paint() is False
    finally:
        importlib.reload(browser_module)


def test_a_reading_script_is_taken_again_after_the_connection_goes(monkeypatch):
    document = {"version": replay_module.VERSION, "url": "https://example.test/",
                "steps": [{"kind": "click", "label": "Go"}]}
    assert replay_module.mutates(document) is False
    attempts = []

    def run_once(doc, browser, screenshots, on_step, done):
        attempts.append(1)
        if len(attempts) == 1:
            raise RuntimeError("WebSocket connection closed")
        done.append({"step": 1})

    monkeypatch.setattr(replay_module, "run_once", run_once)
    result = replay_module.replay(document)
    assert result["status"] == "done" and len(attempts) == 2


def test_a_script_that_types_is_not_taken_again_and_says_where_it_stopped(monkeypatch):
    document = {"version": replay_module.VERSION, "url": "https://example.test/",
                "steps": [{"kind": "click", "label": "Open"}, {"kind": "fill", "label": "Search", "text": "Ada"}]}
    assert replay_module.mutates(document) is True
    attempts = []

    def run_once(doc, browser, screenshots, on_step, done):
        attempts.append(1)
        done.append({"step": 1})  # the first step went through before the connection went
        raise RuntimeError("WebSocket connection closed")

    monkeypatch.setattr(replay_module, "run_once", run_once)
    result = replay_module.replay(document)
    # Taken once only: repeating it would type into the page a second time.
    assert len(attempts) == 1
    assert result["status"] == "connection_lost"
    assert result["completed"] == 1 and result["repeatable"] is False


def test_a_submitting_click_makes_a_script_unrepeatable():
    document = {"version": replay_module.VERSION, "url": "https://example.test/",
                "steps": [{"kind": "click", "label": "Send message"}]}
    assert replay_module.mutates(document) is True


def test_a_query_reports_a_value_or_what_it_raised(monkeypatch):
    b = browser_module.Browser.__new__(browser_module.Browser)
    b.session = "test"
    answers = iter([
        {"result": {"value": "ok"}},
        {"exceptionDetails": {"exception": {"description": "ReferenceError: nope is not defined"}}},
        {"result": {"value": {"a": 1}}},
    ])
    monkeypatch.setattr(browser_module, "cdp", lambda *a, **kw: next(answers))
    assert b.ask("'ok'") == {"value": "ok"}
    assert b.ask("nope")["exception"].startswith("ReferenceError")
    assert b.ask("({a:1})") == {"value": {"a": 1}}


def test_a_long_answer_is_cut_to_the_cap(monkeypatch):
    b = browser_module.Browser.__new__(browser_module.Browser)
    b.session = "test"
    monkeypatch.setattr(browser_module, "cdp", lambda *a, **kw: {"result": {"value": "x" * 9000}})
    assert len(b.ask("big", cap=2048)["value"]) == 2048


def test_a_question_waits_for_the_goals_named_before_it():
    runner = loop.Agent.__new__(loop.Agent)
    runner.asked = [{"query": "first", "after": 1}, {"query": "last", "after": 2}]
    runner.answered, runner.plan_satisfied = {}, {0}
    asked = []
    state = {"plan": ["a", "b"], "status": "ready",
             "browser": Mock(ask=Mock(side_effect=lambda q: asked.append(q) or {"value": q}))}
    runner.ask_due(state)
    assert asked == ["first"]  # the second waits for the goal it follows
    state["status"] = "done"
    runner.ask_due(state)
    assert asked == ["first", "last"]
    assert state["queries"] == [{"value": "first"}, {"value": "last"}]


def held_browser(target="T1", context="C1"):
    browser = Mock()
    browser.target, browser.context = target, context
    return browser


def test_a_handle_carries_everything_needed_to_close_the_page(monkeypatch):
    monkeypatch.setattr(session_module, "_send", lambda req: {"browser_session_id": "a-uuid-from-the-handshake"})
    handle = session_module.handle_for(held_browser(), worker="worker-3")
    assert handle["worker_id"] == "worker-3"
    assert handle["target_id"] == "T1" and handle["context_id"] == "C1"
    assert handle["browser_session_id"] == "a-uuid-from-the-handshake"
    assert handle["id"] and handle["id"] != session_module.handle_for(held_browser())["id"]


def test_a_harness_that_does_not_keep_the_session_reports_none(monkeypatch):
    def refuses(req):
        raise RuntimeError("unknown meta")

    monkeypatch.setattr(session_module, "_send", refuses)
    assert session_module.browser_session_id() is None


def test_a_handle_closes_the_target_and_the_context_around_it(monkeypatch):
    sent = []
    monkeypatch.setattr(session_module, "cdp", lambda method, **kw: sent.append((method, kw)) or {})
    assert session_module.close_handle({"target_id": "T1", "context_id": "C1"}) == {
        "target": True, "context": True}
    assert [method for method, _ in sent] == ["Target.closeTarget", "Target.disposeBrowserContext"]


def test_closing_a_page_that_is_already_gone_is_not_an_error(monkeypatch):
    def gone(method, **kw):
        raise RuntimeError("No target with given id found")

    monkeypatch.setattr(session_module, "cdp", gone)
    # A registry sweeping orphans cannot know which the worker closed on its way out.
    assert session_module.close_handle({"target_id": "T1", "context_id": "C1"}) == {
        "target": False, "context": False}



def test_freeing_a_worker_closes_its_page_and_gives_back_the_daemon(monkeypatch):
    monkeypatch.setattr(session_module, "cdp", lambda method, **kw: {})
    runner = loop.Agent.__new__(loop.Agent)
    runner.borrowed, runner.browser = False, Mock()
    runner.handle = {"target_id": "T1", "context_id": "C1"}
    assert runner.free() == {"target": True, "context": True}
    assert runner.handle is None
    runner.browser.release.assert_called_once()
