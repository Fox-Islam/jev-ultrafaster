"""Opening, borrowing and giving back a browser, against a fake daemon. No browser, no model."""

import os
import signal
import subprocess
import sys

import pytest

from jev_ultrafast import browser as browser_module
from jev_ultrafast import session as session_module

ANSWERS = {
    "Target.createBrowserContext": {"browserContextId": "C1"},
    "Target.createTarget": {"targetId": "T-new"},
    "Target.attachToTarget": {"sessionId": "S1"},
}
VALUES = {"document.readyState": "complete", "[innerWidth, innerHeight]": [1568, 772]}


@pytest.fixture
def sent(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(browser_module, "LOCKS", tmp_path)

    def cdp(method, session_id=None, **params):
        calls.append((method, params))
        if method == "Runtime.evaluate":
            return {"result": {"value": VALUES.get(params["expression"])}}
        return ANSWERS.get(method, {})

    monkeypatch.setattr(browser_module, "cdp", cdp)
    monkeypatch.setattr(browser_module, "ensure_daemon", lambda: None)
    monkeypatch.delenv("JEV_BROWSER_CONTEXT", raising=False)
    monkeypatch.delenv("JEV_KEEP_OPEN", raising=False)
    return calls


def methods(calls):
    return [method for method, _ in calls]


def test_an_attached_page_is_used_as_it_stands(sent):
    browser = browser_module.Browser.attach("T-mine")
    try:
        assert browser.target == "T-mine" and browser.session == "S1" and browser.context is None
        assert browser.viewport == (1568, 772)  # frames are judged against the window it has
        used = methods(sent)
        assert ("Target.attachToTarget", {"targetId": "T-mine", "flatten": True}) in sent
        assert "Emulation.setFocusEmulationEnabled" in used
        for made in ("Target.createTarget", "Target.createBrowserContext",
                     "Emulation.setDeviceMetricsOverride", "Page.navigate"):
            assert made not in used
    finally:
        browser.close()


def test_closing_an_attached_page_detaches_and_leaves_it_open(sent):
    browser = browser_module.Browser.attach("T-mine")
    browser.close()
    used = methods(sent)
    assert ("Target.detachFromTarget", {"sessionId": "S1"}) in sent
    assert "Target.closeTarget" not in used and "Target.disposeBrowserContext" not in used
    browser.close()  # idempotent
    assert methods(sent).count("Target.detachFromTarget") == 1
    # The daemon is given back: another browser can be had straight away.
    browser_module.Browser.attach("T-mine").close()


def test_an_opened_page_is_still_made_sized_and_closed(sent):
    browser = browser_module.Browser("https://example.test/")
    used = methods(sent)
    for made in ("Target.createBrowserContext", "Target.createTarget",
                 "Emulation.setDeviceMetricsOverride", "Page.navigate"):
        assert made in used
    browser.close()
    assert ("Target.closeTarget", {"targetId": "T-new"}) in sent
    assert ("Target.disposeBrowserContext", {"browserContextId": "C1"}) in sent
    assert "Target.detachFromTarget" not in methods(sent)


def test_a_failed_attach_gives_the_daemon_back(sent, monkeypatch):
    def gone(method, session_id=None, **params):
        raise RuntimeError("No target with given id found")

    monkeypatch.setattr(browser_module, "cdp", gone)
    with pytest.raises(RuntimeError):
        browser_module.Browser.attach("T-gone")
    assert browser_module.IN_USE.acquire(blocking=False)
    browser_module.IN_USE.release()


def test_a_handle_on_an_attached_page_never_closes_it(sent, monkeypatch):
    monkeypatch.setattr(session_module, "_send", lambda req: {})
    closed = []
    monkeypatch.setattr(session_module, "cdp", lambda method, **kw: closed.append(method) or {})
    browser = browser_module.Browser.attach("T-mine")
    try:
        handle = session_module.handle_for(browser)
    finally:
        browser.close()
    assert handle["borrowed"] is True
    assert session_module.close_handle(handle) == {"target": False, "context": False}
    assert closed == []


# A process that takes the daemon and holds it until its stdin closes. It opens no page: the claim
# is taken before anything is sent, so the fake daemon needs no answers.
HOLDER = """
import sys
from jev_ultrafast import browser
browser.ensure_daemon = lambda: None
browser.Browser.open = lambda self, *args, **kwargs: None
try:
    held = browser.Browser("about:blank")
except browser.DaemonBusy:
    print("busy", flush=True)
    sys.exit(3)
print("held", flush=True)
sys.stdin.read()
held.release()
"""


def holder(tmp_path, stdin=subprocess.PIPE):
    env = {**os.environ, "XDG_CACHE_HOME": str(tmp_path), "BU_NAME": "jev-test-lock"}
    env.pop("JEV_DAEMON_PER_RUN", None)
    return subprocess.Popen([sys.executable, "-c", HOLDER], stdin=stdin, stdout=subprocess.PIPE, text=True, env=env)


def test_a_second_process_is_refused_the_daemon(tmp_path):
    first = holder(tmp_path)
    try:
        assert first.stdout.readline().strip() == "held"
        second = holder(tmp_path, stdin=subprocess.DEVNULL)
        assert second.communicate(timeout=30)[0].strip() == "busy" and second.returncode == 3
    finally:
        first.communicate(input="", timeout=30)
    # Given back on close, so the next process gets it.
    after = holder(tmp_path, stdin=subprocess.DEVNULL)
    assert after.communicate(timeout=30)[0].strip() == "held" and after.returncode == 0


@pytest.mark.skipif(sys.platform == "win32", reason="SIGKILL")
def test_a_killed_holder_does_not_keep_the_daemon(tmp_path):
    first = holder(tmp_path)
    assert first.stdout.readline().strip() == "held"
    first.send_signal(signal.SIGKILL)
    first.wait(timeout=30)
    after = holder(tmp_path, stdin=subprocess.DEVNULL)
    assert after.communicate(timeout=30)[0].strip() == "held"


def test_a_claim_held_elsewhere_refuses_this_process_too(sent):
    elsewhere = browser_module.claim_daemon()
    try:
        with pytest.raises(browser_module.DaemonBusy):
            browser_module.Browser.attach("T-mine")
        # The thread lock is not left taken by the refusal.
        assert browser_module.IN_USE.acquire(blocking=False)
        browser_module.IN_USE.release()
    finally:
        browser_module.let_go(elsewhere)
    browser_module.Browser.attach("T-mine").close()


def test_a_daemon_per_run_is_named_for_this_process():
    env = {**os.environ, "JEV_DAEMON_PER_RUN": "1", "BU_NAME": "default"}
    shown = subprocess.run(
        [sys.executable, "-c",
         "import os, jev_ultrafast; from browser_harness import helpers; "
         "from jev_ultrafast import browser; print(os.getpid(), helpers.NAME, browser.DAEMON, browser.PER_RUN)"],
        capture_output=True, text=True, env=env, timeout=30, check=True,
    ).stdout.split()
    pid, name, daemon, per_run = shown
    assert name == daemon and name.startswith(f"jev-{pid}-") and per_run == "True"
