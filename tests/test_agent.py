"""Offline contracts for a dynamic operation/target policy. No paid APIs."""

import json
import time
from copy import deepcopy
from datetime import date
from unittest.mock import Mock

import pytest

from jev_ultrafast import agent as loop
from jev_ultrafast import browser as browser_module
from jev_ultrafast import model
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
    assert sent["questions"]["click_target"]["criteria"]  # fell back rather than stranding the run


def observation(url="https://example.test/", actions=None, expanded=None):
    acts = [{"id": f"e{i}", "kind": "click", "label": f"c{i}", "node": i} for i in range(actions or 10)]
    for a in acts[:expanded or 0]:
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
