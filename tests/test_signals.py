from unittest.mock import Mock

import pytest
from test_agent import choice, decision, page, runner  # noqa: F401

from jev_ultrafast import agent as loop
from jev_ultrafast import browser as browser_module
from jev_ultrafast import model
from jev_ultrafast.browser import Browser


def browser():
    b = Browser.__new__(Browser)
    b.call = Mock(side_effect=RuntimeError("No data found for resource with given identifier"))
    return b


def sent(b, request_id, method, url, kind="Fetch"):
    b.note_fault({"method": "Network.requestWillBeSent",
                  "params": {"requestId": request_id, "type": kind, "request": {"method": method, "url": url}}})


def received(b, request_id, url, status, kind="Fetch"):
    b.note_fault({"method": "Network.responseReceived",
                  "params": {"requestId": request_id, "type": kind, "response": {"url": url, "status": status}}})


def test_a_pages_own_console_error_is_a_fault_and_follows_the_last_action():
    b = browser()
    b.note_fault({"method": "Runtime.consoleAPICalled",
                  "params": {"type": "error", "args": [{"type": "string", "value": "Profile save failed"}]}})
    b.note_fault({"method": "Runtime.consoleAPICalled", "params": {"type": "log", "args": [{"value": "hello"}]}})
    assert b.faults["console"] == [{"level": "error", "text": "Profile save failed", "url": None}]
    assert b.after_action({"url": "https://shop.example.test/"}) == {
        "console_errors_after_last_action": ["Profile save failed"]}


def test_a_decision_is_told_of_the_sites_own_writes_and_failures_and_nothing_else():
    b = browser()
    sent(b, "1", "POST", "https://api.example.test/save")
    received(b, "1", "https://api.example.test/save", 500)
    sent(b, "2", "GET", "https://example.test/items")
    received(b, "2", "https://example.test/items", 200)  # a read that worked
    sent(b, "3", "GET", "https://example.test/stock")
    received(b, "3", "https://example.test/stock", 404)  # a read that failed
    sent(b, "4", "POST", "https://analytics.other.test/collect")
    received(b, "4", "https://analytics.other.test/collect", 500)  # another site's
    told = b.after_action({"url": "https://shop.example.test/profile"})["network_after_last_action"]
    assert told == [
        {"method": "POST", "url": "https://api.example.test/save", "status": 500},
        {"method": "GET", "url": "https://example.test/stock", "status": 404},
    ]


def test_a_page_with_nothing_to_report_adds_nothing_to_the_decision():
    b = browser()
    sent(b, "1", "GET", "https://example.test/items")
    received(b, "1", "https://example.test/items", 200)
    assert b.after_action({"url": "https://example.test/"}) == {}


@pytest.mark.parametrize(("kind", "kept"), [("wait", True), ("scroll", True), ("click", False)])
def test_only_an_input_to_the_page_starts_what_follows_it_afresh(monkeypatch, kind, kept):
    monkeypatch.setattr(browser_module, "browser_operation", lambda request: {"executed": "x"})
    monkeypatch.setattr(browser_module, "drain_events", lambda: [])
    b = browser()
    b.session = "s"
    b.note_fault({"method": "Runtime.exceptionThrown",
                  "params": {"exceptionDetails": {"exception": {"description": "Error: save failed"}}}})
    b.act({"id": "x", "kind": kind, "label": "Go", "node": 1, "delta": 560}, page(), insist=True)
    assert bool(b.after_action({"url": "https://example.test/"})) is kept


def test_what_followed_the_last_action_goes_in_the_decisions_state(monkeypatch):
    seen = []

    def post(_url, _key, body):
        seen.append(body)
        blocked = choice(body["questions"]["operation"]["criteria"], "BLOCKED")
        return {"model": "test", "answers": {"operation": blocked}}

    monkeypatch.setenv("TYPESAFE_API_KEY", "test")
    monkeypatch.setattr(model, "post_json", post)
    p = page()
    p["after_action"] = {"network_after_last_action": [{"method": "POST", "url": "/save", "status": 500}]}
    model.choose(p, "Save the profile", [])
    assert seen[0]["state"]["network_after_last_action"] == [{"method": "POST", "url": "/save", "status": 500}]


def test_an_unnamed_control_is_described_by_its_fill_unless_turned_off(monkeypatch):
    swatch = {"label": "button", "role": "button", "colour": "red"}
    assert "filled red" in model.describe_target("3", swatch)
    monkeypatch.setenv("JEV_COLOURS", "0")
    assert "filled" not in model.describe_target("3", swatch)


def test_a_guess_taken_before_what_followed_the_action_was_known_is_decided_again(runner, monkeypatch):  # noqa: F811
    fresh = {**decision("e3"), "operation": "CLICK", "plan": []}
    monkeypatch.setattr(loop, "choose", Mock(return_value=fresh))
    runner.state["page"] = {**runner.state["page"], "after_action": {"console_errors_after_last_action": ["Error"]}}
    runner.speculating, runner.guesses_used = True, 0
    runner.plan_satisfied, runner.held, runner.ready_values, runner.unfresh_since = set(), {}, {}, None
    runner.guessing, runner.guessed = None, (runner.state["page"]["fingerprint"], {**decision("e1"), "plan": []})
    runner.command("predict")
    assert runner.state["decision"]["choice"] == "e3"
    loop.choose.assert_called_once()
