"""Pages left open after a run, addressed by a handle.

Verifying a site is rarely one question. A caller that has driven a page into the state it cares
about often wants to look again a moment later, and the page it wants is the one already in front
of it: navigating again would be a different page, with the form not filled in and the cookie
banner back.

No timer is kept here. A handle names a target and the context around it, and closing it is an
ordinary CDP call, so the caller holding the registry can close a page from a different process
from the one that opened it - including one whose worker has since died.
"""

import os
import secrets

from browser_harness.helpers import _send, cdp

WORKER = os.environ.get("BU_NAME") or os.environ.get("JEV_WORKER") or "default"


def handle_for(browser, worker=None):
    """The handle that reaches this page again.

    Everything needed to close the page is in it, so the holder needs nothing of this process:
    `target_id` is the page, `context_id` the storage around it, and `browser_session_id` the
    browser they live in, which is what Browser Run's HTTP API deletes when a worker is lost.
    """
    return {
        "id": secrets.token_urlsafe(12),
        "worker_id": worker or WORKER,
        "browser_session_id": browser_session_id(),
        "target_id": browser.target,
        "context_id": getattr(browser, "context", None),
    }


def browser_session_id():
    """The browser this daemon is attached to, as Browser Run named it.

    Reported in the `cf-browser-session-id` header of the WebSocket handshake, which nothing
    inside CDP repeats, so the daemon keeps it from the moment it connected. None for a local
    browser, and for a daemon whose harness has not been taught to keep it.
    """
    try:
        return _send({"meta": "browser_session"}).get("browser_session_id")
    except (RuntimeError, OSError, KeyError):
        return None


def close_handle(handle):
    """Close the page a handle names, and dispose the context around it.

    A no-op when the page is already gone, because a caller sweeping orphans cannot know which of
    them a worker closed on its way out. Returns what it actually closed.
    """
    closed = {"target": False, "context": False}
    target, context = handle.get("target_id"), handle.get("context_id")
    if target:
        try:
            cdp("Target.closeTarget", targetId=target)
            closed["target"] = True
        except (RuntimeError, KeyError, OSError):
            pass
    if context:
        try:
            cdp("Target.disposeBrowserContext", browserContextId=context)
            closed["context"] = True
        except (RuntimeError, KeyError, OSError):
            pass
    return closed


def ask(handle, queries, browser=None):
    """Run more queries on a page a handle names.

    A browser from this process is used when there is one; otherwise the page is reached by
    attaching to its target, which is what lets another process ask.
    """
    from .browser import Browser

    if browser is not None:
        return [browser.ask(query) for query in queries]
    reached = Browser.__new__(Browser)
    reached.target = handle["target_id"]
    reached.session = cdp("Target.attachToTarget", targetId=handle["target_id"], flatten=True)["sessionId"]
    return [reached.ask(query) for query in queries]
