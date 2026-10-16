from unittest.mock import Mock

import pytest
from test_agent import choice, decision, runner  # noqa: F401

from jev_ultrafast import agent as loop
from jev_ultrafast import model
from jev_ultrafast import replay as replay_module
from jev_ultrafast.browser import fingerprint

SECRET = "hunter2-correct-horse"


def form(password_value=""):
    state = {
        "url": "https://example.test/register",
        "title": "Register",
        "text": "Create your account",
        "scroll": {"y": 0},
        "actions": [
            {"id": "e1", "kind": "fill", "label": "Email", "role": "textbox", "value": "", "node": 1},
            {"id": "e2", "kind": "fill", "label": "Password", "role": "textbox", "value": password_value,
             "node": 2, "secret": True},
            {"id": "e3", "kind": "click", "label": "Create my account", "role": "button", "value": "", "node": 3},
            {"id": "wait", "kind": "wait", "label": "Wait"},
        ],
    }
    state["fingerprint"] = fingerprint(state)
    return state


def asked(monkeypatch, agent, page):
    sent = []

    def post(_url, _key, body):
        sent.append(body)
        blocked = choice(body["questions"]["operation"]["criteria"], "BLOCKED")
        return {"model": "test", "answers": {"operation": blocked}}

    monkeypatch.setenv("TYPESAFE_API_KEY", "test")
    monkeypatch.setattr(model, "post_json", post)
    agent.choose(page, "Register", [], [], set())
    return sent[0]


def test_a_run_without_secrets_is_not_offered_a_password_field(monkeypatch):
    agent = loop.Agent.__new__(loop.Agent)
    agent.secrets = {}
    body = asked(monkeypatch, agent, form())
    assert [e["label"] for e in body["state"]["elements"]] == ["Email", "Create my account"]


def test_a_run_with_secrets_is_offered_one_and_never_shown_a_secret(monkeypatch):
    agent = loop.Agent.__new__(loop.Agent)
    agent.secrets = {"password": SECRET}
    shown = form()
    shown["text"] = f"Create your account. Your password is {SECRET}"  # a page showing it in clear
    body = asked(monkeypatch, agent, shown)
    assert "Password" in [e["label"] for e in body["state"]["elements"]]
    assert "a password field, typed from the caller's secret" in str(body["questions"])
    assert SECRET not in str(body)


def test_a_secret_is_typed_by_the_executor_without_the_text_model_and_never_recorded(runner, monkeypatch):  # noqa: F811
    helper = Mock()
    monkeypatch.setattr(loop, "field_text", helper)
    runner.secrets = {"password": SECRET}
    runner.state["page"] = form()
    runner.state["decision"] = decision("e2")
    runner.state["browser"].act = Mock()
    runner.state["browser"].settle.return_value = form("(filled)")
    runner.text_failures, runner.refused = 0, {}
    runner.command("act", {"fingerprint": runner.state["page"]["fingerprint"]})
    assert runner.state["browser"].act.call_args.kwargs["text"] == SECRET
    helper.assert_not_called()
    step = runner.state["history"][-1]
    assert step["secret"] is True and step["text"] is None
    assert SECRET not in str(runner.state["history"])


@pytest.mark.parametrize(("label", "secrets", "found"), [
    ("Password", {"password": "a"}, "a"),
    ("Confirm password", {"password": "a", "pin": "b"}, "a"),
    ("Passcode", {"only": "c"}, "c"),
    ("Passcode", {"password": "a", "pin": "b"}, None),
    ("Password", {}, None),
])
def test_a_field_takes_the_secret_its_label_names_or_the_only_one(label, secrets, found):
    assert model.secret_for(label, secrets) == found


def test_a_script_marks_a_secret_step_and_replays_it_only_when_given_the_secret():
    state = {"url": "https://example.test/register", "goal": "Register",
             "history": [{"kind": "fill", "action": "Password", "text": None, "secret": True}]}
    document = replay_module.script(state)
    assert document["steps"] == [{"kind": "fill", "label": "Password", "secret": True}]
    with pytest.raises(ValueError, match="types a secret"):
        replay_module.replay(document)
