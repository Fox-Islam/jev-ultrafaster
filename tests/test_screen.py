import base64
import io

import pytest

from jev_ultrafast import model, screen
from jev_ultrafast.browser import fingerprint

Image = pytest.importorskip("PIL.Image")

from test_agent import choice, decision, runner  # noqa: E402,F401


def shot(width=1120, height=780):
    out = io.BytesIO()
    Image.new("RGB", (width, height), "white").save(out, format="JPEG")
    return base64.b64encode(out.getvalue()).decode()


def page(width=1120, height=780):
    state = {
        "url": "https://example.test/",
        "title": "Shop",
        "text": "Shop",
        "w": 1120,
        "h": 780,
        "scroll": {"y": 0},
        "screenshot": shot(width, height),
        "actions": [
            {"id": "e1", "kind": "click", "label": "button", "role": "button", "value": "", "node": 1,
             "rect": {"x": 1000, "y": 10, "w": 40, "h": 40}},
            {"id": "e2", "kind": "click", "label": "Products", "role": "link", "value": "", "node": 2,
             "rect": {"x": 100, "y": 10, "w": 120, "h": 30}},
            {"id": "e3", "kind": "click", "label": "Footer link", "role": "link", "value": "", "node": 3,
             "rect": {"x": 100, "y": 1400, "w": 120, "h": 30}, "offscreen": "below"},
            {"id": "wait", "kind": "wait", "label": "Wait"},
        ],
    }
    state["fingerprint"] = fingerprint(state)
    return state


def answering(target, target_confidence=1.0, done=0.9):
    sent = []

    def post(_url, _key, body, headers=None):
        sent.append((body, headers))
        heads = body["questions"]
        click = choice(heads["click_target"]["criteria"], target)
        click["confidence"] = target_confidence
        answers = {
            "operation": choice(heads["operation"]["criteria"], "CLICK"),
            "click_target": click,
            "screen_done": {"noul": done},
        }
        return {"model": "test", "answers": answers}

    return post, sent


@pytest.fixture
def images(monkeypatch):
    monkeypatch.setenv("TYPESAFE_API_KEY", "test")
    monkeypatch.setenv("JEV_IMAGES", "1")
    monkeypatch.delenv("TYPESAFE_MODEL", raising=False)


def test_the_screenshot_goes_with_its_opt_in_header_and_is_not_kept_in_the_decision(monkeypatch, images):
    post, sent = answering("1")
    monkeypatch.setattr(model, "post_json", post)
    d = model.choose(page(), "Open the cart", [])
    body, headers = sent[0]
    assert headers == screen.HEADERS
    name = body["state"]["screen"]
    assert set(body["assets"]) == {name} and body["assets"][name]["format"] == "jpg"
    assert "`screen`" in body["questions"]["operation"]["instructions"]
    assert body["questions"]["screen_done"]["type"] == "noul"
    assert "data_base64" not in d["request"]["assets"][name] and d["request"]["assets"][name]["bytes"] > 0
    assert d["screen_done"] == 0.9


def test_without_the_setting_nothing_changes(monkeypatch):
    monkeypatch.setenv("TYPESAFE_API_KEY", "test")
    monkeypatch.delenv("JEV_IMAGES", raising=False)
    post, sent = answering("1")
    monkeypatch.setattr(model, "post_json", post)
    model.choose(page(), "Open the cart", [])
    body, headers = sent[0]
    assert headers is None and "assets" not in body and "screen" not in body["state"]
    assert "screen_done" not in body["questions"]


def test_elements_in_view_are_outlined_on_the_image_and_offscreen_ones_are_not():
    _, targets, _ = model.action_space(page()["actions"])
    image = Image.open(io.BytesIO(screen.marked(page(), targets)))
    assert image.size == (1120, 780)
    # Read by the green channel, which the outline has none of and JPEG noise barely moves.
    assert image.getpixel((1000, 30))[1] < 80  # left edge of element 1's outline
    assert image.getpixel((1020, 30))[1] > 240  # its inside is left alone
    assert screen.places(targets).keys() == {"1", "2"}


def test_a_screenshot_too_large_is_scaled_and_one_too_small_is_not_sent():
    _, targets, _ = model.action_space(page()["actions"])
    large = Image.open(io.BytesIO(screen.marked(page(2240, 1560), targets)))
    assert max(large.size) == 1568
    assert screen.marked(page(200, 140), targets) is None


def test_a_done_the_screenshot_does_not_bear_out_is_doubtful(runner):  # noqa: F811
    runner.state["decision"] = {**decision("DONE"), "choice": "DONE", "operation": "DONE", "screen_done": 0.1}
    runner.state["plan"] = ["Submit"]
    runner.unfresh_since = None
    out = runner.command("act", {"fingerprint": runner.state["page"]["fingerprint"]})
    assert out["status"] == "done" and out["doubtful"] is True and out["screen_done"] == 0.1


def named(p):
    """The fixture page with every element given a distinct name, so auto mode has no reason to look."""
    p["actions"][0]["label"] = "Cart"
    return p


def row(p):
    """The fixture page with two unnamed controls side by side, as in an icon toolbar."""
    p["actions"][0]["group"] = 7
    p["actions"][1].update(label="button", role="button", group=7)
    return p


def test_auto_sends_no_image_when_nothing_calls_for_one(monkeypatch, images):
    monkeypatch.setenv("JEV_IMAGES", "auto")
    post, sent = answering("1")
    monkeypatch.setattr(model, "post_json", post)
    d = model.choose(named(page()), "Open the cart", [])
    assert len(sent) == 1 and sent[0][1] is None and "assets" not in sent[0][0]
    assert "looked_again" not in d


@pytest.mark.parametrize(("prepare", "goal"), [
    (row, "Open the cart"),                     # a row of unnamed controls
    (named, "Choose the red colour"),           # the goal names a colour
    (named, "Open the tab right of the highlighted one"),
])
def test_auto_sends_the_image_up_front_where_text_cannot_tell(monkeypatch, images, prepare, goal):
    monkeypatch.setenv("JEV_IMAGES", "auto")
    post, sent = answering("1")
    monkeypatch.setattr(model, "post_json", post)
    model.choose(prepare(page()), goal, [])
    assert len(sent) == 1 and sent[0][1] == screen.HEADERS


def test_one_unnamed_control_alone_is_not_a_reason_to_look(monkeypatch, images):
    monkeypatch.setenv("JEV_IMAGES", "auto")
    p = page()
    p["actions"][0]["group"] = 7
    post, sent = answering("1")
    monkeypatch.setattr(model, "post_json", post)
    model.choose(p, "Open the cart", [])
    assert sent[0][1] is None


@pytest.mark.parametrize(("operation", "confidence", "asked"), [
    ("DONE", 0.65, True),       # a DONE the text was not sure of
    ("DONE", 0.95, False),
    ("BLOCKED", 0.95, True),    # giving up is checked against the picture
    ("CLICK", 0.5, True),       # unsure which operation to take
    ("CLICK", 0.9, False),
])
def test_auto_asks_again_with_the_image_where_the_text_answer_needs_it(
    monkeypatch, images, operation, confidence, asked
):
    monkeypatch.setenv("JEV_IMAGES", "auto")
    sent = []

    def post(_url, _key, body, headers=None):
        sent.append(headers)
        heads = body["questions"]
        looking = headers is not None
        answer = choice(heads["operation"]["criteria"], "WAIT" if looking else operation)
        answer["confidence"] = confidence
        target = choice(heads["click_target"]["criteria"], "2")
        return {"model": "test", "answers": {"operation": answer, "click_target": target}}

    monkeypatch.setattr(model, "post_json", post)
    d = model.choose(named(page()), "Open the cart", [])
    assert sent == ([None, screen.HEADERS] if asked else [None])
    assert (d.get("looked_again") == operation) is asked


@pytest.mark.parametrize(("operation", "confidence", "prepare", "asked"), [
    ("DONE", 0.65, named, True),
    ("BLOCKED", 0.95, named, True),
    ("CLICK", 0.95, row, True),        # the settled page has a row of unnamed controls
    ("CLICK", 0.95, named, False),     # nothing the text could not settle
])
def test_auto_asks_a_guess_again_where_it_would_have_looked(monkeypatch, images, operation, confidence, prepare,
                                                             asked):
    from jev_ultrafast import agent as loop

    monkeypatch.setenv("JEV_IMAGES", "auto")
    calls = []
    monkeypatch.setattr(loop, "choose", lambda *args, look=False: calls.append(look) or {"operation": "BLOCKED"})
    agent = loop.Agent.__new__(loop.Agent)
    guess = {"operation": operation, "choice": "e1", "confidence": confidence, "usage": {}, "latency_ms": 5,
             "request": {}}
    out = agent.looked_at(guess, prepare(page()), ("Open the cart", [], [], set()))
    assert calls == ([True] if asked else [])
    assert (out.get("looked_again") == "guess") is asked


def test_auto_keeps_guessing_and_always_does_not(monkeypatch, images):
    from jev_ultrafast import agent as loop

    monkeypatch.setattr(loop.Browser, "__init__", lambda self, url: None)
    monkeypatch.setattr(loop.Browser, "settle", lambda self, **kw: page())
    monkeypatch.setenv("JEV_IMAGES", "auto")
    assert loop.Agent("https://example.test/", "Open the cart").speculating
    monkeypatch.setenv("JEV_IMAGES", "1")
    assert not loop.Agent("https://example.test/", "Open the cart").speculating


def test_the_readme_states_the_thresholds_the_code_uses():
    from pathlib import Path

    readme = " ".join((Path(__file__).parent.parent / "README.md").read_text().split())
    assert f"DONE below {screen.DONE_SURE} confidence" in readme
    assert f"other operation below {screen.UNCERTAIN}" in readme
