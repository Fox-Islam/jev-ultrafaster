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


# A run that opens a page against a fake daemon, writes each CDP method it sends to a file, and
# then waits to be ended. How it ends is the argument.
RUN = """
import json, sys
from jev_ultrafast import browser
sent = open(sys.argv[1], "a")
answers = {"Target.createBrowserContext": {"browserContextId": "C1"}, "Target.createTarget": {"targetId": "T-new"},
           "Target.attachToTarget": {"sessionId": "S1"}}
def cdp(method, session_id=None, **params):
    sent.write(method + "\\n"); sent.flush()
    if method == "Runtime.evaluate":
        return {"result": {"value": "complete" if params["expression"] == "document.readyState" else [800, 600]}}
    return answers.get(method, {})
browser.cdp = cdp
browser.ensure_daemon = lambda: None
held = browser.Browser.attach("T-mine") if sys.argv[2] == "attach" else browser.Browser("https://example.test/")
if sys.argv[3] == "keep":
    held.keep()
print("open", flush=True)
if sys.argv[3] != "exit":
    sys.stdin.read()
"""


def run(tmp_path, how="open", end="wait"):
    log = tmp_path / "sent.log"
    env = {**os.environ, "XDG_CACHE_HOME": str(tmp_path), "BU_NAME": "jev-test-exit"}
    for name in ("JEV_DAEMON_PER_RUN", "JEV_KEEP_OPEN", "JEV_BROWSER_CONTEXT"):
        env.pop(name, None)
    child = subprocess.Popen([sys.executable, "-c", RUN, str(log), how, end],
                             stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, env=env)
    assert child.stdout.readline().strip() == "open"
    return child, log


def sent_by(log):
    return log.read_text().split()


@pytest.mark.skipif(sys.platform == "win32", reason="SIGTERM")
def test_a_terminated_run_closes_its_page_and_context(tmp_path):
    child, log = run(tmp_path)
    child.send_signal(signal.SIGTERM)
    child.communicate(timeout=30)
    assert child.returncode == 128 + signal.SIGTERM
    assert "Target.closeTarget" in sent_by(log) and "Target.disposeBrowserContext" in sent_by(log)


@pytest.mark.skipif(sys.platform == "win32", reason="SIGINT")
def test_an_interrupted_run_closes_its_page_and_still_raises(tmp_path):
    child, log = run(tmp_path)
    child.send_signal(signal.SIGINT)
    child.communicate(timeout=30)
    assert child.returncode != 0  # KeyboardInterrupt, as without the handler
    assert "Target.closeTarget" in sent_by(log)


@pytest.mark.skipif(sys.platform == "win32", reason="SIGTERM")
def test_a_terminated_run_in_an_attached_page_only_detaches(tmp_path):
    child, log = run(tmp_path, how="attach")
    child.send_signal(signal.SIGTERM)
    child.communicate(timeout=30)
    assert "Target.detachFromTarget" in sent_by(log)
    assert "Target.closeTarget" not in sent_by(log)


def test_a_run_that_exits_without_closing_still_closes(tmp_path):
    child, log = run(tmp_path, end="exit")
    child.communicate(timeout=30)
    assert child.returncode == 0
    assert sent_by(log).count("Target.closeTarget") == 1


def test_a_kept_page_outlives_the_process(tmp_path):
    child, log = run(tmp_path, end="keep")
    child.communicate(input="", timeout=30)
    assert "Target.closeTarget" not in sent_by(log)


def test_closing_hands_the_signals_back(sent):
    before = signal.getsignal(signal.SIGTERM)
    browser = browser_module.Browser("https://example.test/")
    assert signal.getsignal(signal.SIGTERM) == browser.on_signal
    browser.close()
    assert signal.getsignal(signal.SIGTERM) == before


def test_a_failed_open_closes_what_it_made(sent, monkeypatch):
    def navigate(self, url):
        raise RuntimeError("navigation refused")

    monkeypatch.setattr(browser_module.Browser, "navigate", navigate)
    with pytest.raises(RuntimeError):
        browser_module.Browser("https://example.test/")
    assert ("Target.closeTarget", {"targetId": "T-new"}) in sent
    assert ("Target.disposeBrowserContext", {"browserContextId": "C1"}) in sent


@pytest.mark.skipif(sys.platform == "win32", reason="SIGCHLD")
def test_stopping_the_own_daemon_lets_the_system_collect_it(monkeypatch):
    seen = []
    monkeypatch.setattr(browser_module, "restart_daemon", lambda name: seen.append(signal.getsignal(signal.SIGCHLD)))
    before = signal.getsignal(signal.SIGCHLD)
    browser_module.stop_own_daemon()
    assert seen == [signal.SIG_IGN]
    assert signal.getsignal(signal.SIGCHLD) == before


@pytest.mark.skipif(sys.platform == "win32", reason="SIGCHLD")
def test_the_daemon_stop_restores_the_handler_when_it_fails(monkeypatch):
    def fail(name):
        raise RuntimeError("stuck")

    monkeypatch.setattr(browser_module, "restart_daemon", fail)
    before = signal.getsignal(signal.SIGCHLD)
    browser_module.stop_own_daemon()
    assert signal.getsignal(signal.SIGCHLD) == before


def test_a_per_run_lock_file_goes_when_the_browser_does(sent, monkeypatch, tmp_path):
    monkeypatch.setattr(browser_module, "PER_RUN", True)
    browser_module.Browser.attach("T-mine").close()
    assert list(tmp_path.glob("*.lock")) == []


def test_the_shared_lock_file_stays(sent, tmp_path):
    browser_module.Browser.attach("T-mine").close()
    assert [p.name for p in tmp_path.glob("*.lock")] == [f"{browser_module.DAEMON}.lock"]


@pytest.mark.skipif(sys.platform == "win32", reason="SIGTERM")
def test_a_per_run_lock_file_goes_when_the_run_is_terminated(tmp_path):
    env = {**os.environ, "XDG_CACHE_HOME": str(tmp_path), "BU_NAME": "jev-test-perrun"}
    env["JEV_DAEMON_PER_RUN"] = "1"
    code = RUN.replace("browser.ensure_daemon = lambda: None", "browser.ensure_own_daemon = lambda: None")
    child = subprocess.Popen([sys.executable, "-c", code, str(tmp_path / "s.log"), "attach", "wait"],
                             stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, env=env)
    assert child.stdout.readline().strip() == "open"
    assert list((tmp_path / "jev").glob("*.lock"))
    child.send_signal(signal.SIGTERM)
    child.wait(timeout=30)
    assert list((tmp_path / "jev").glob("*.lock")) == []
