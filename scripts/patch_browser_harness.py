"""Teach the installed browser-harness daemon to send custom headers with its CDP connection.

A hosted CDP endpoint authenticates the WebSocket handshake with a header. Cloudflare Browser
Rendering is one: the token goes on the request, not in the URL, so a URL-only setting cannot
reach it. browser-harness 0.1.13 builds its client without headers, and `cdp_use.CDPClient`
already accepts them, so the gap is two call sites in the daemon.

This edits the installed package, which `uv sync` will overwrite, so it is idempotent and safe to
re-run. `patches/browser-harness-cdp-headers.patch` is the same change as a diff, for upstreaming.
"""

import json
import pathlib
import sys

import browser_harness

EDITS = (
    (
        '        await self.attach_first_page()',
        '        # Browser Run names the browser it gave us in the handshake response, and nothing\n'
        '        # inside CDP reports it. Read once, here, because the response is gone afterwards.\n'
        '        try:\n'
        '            self.browser_session = dict(self.cdp.ws.response.headers).get("cf-browser-session-id")\n'
        '        except Exception:\n'
        '            self.browser_session = None\n'
        '        await self.attach_first_page()',
    ),
    (
        '        if meta == "session":     return {"session_id": self.session}',
        '        if meta == "session":     return {"session_id": self.session}\n'
        '        if meta == "browser_session":\n'
        '            return {"browser_session_id": getattr(self, "browser_session", None)}',
    ),
    (
        '        self.cdp = _PatientCDPClient(url) if BROWSER_KIND == "local" else CDPClient(url)',
        '        self.cdp = (\n'
        '            _PatientCDPClient(url)\n'
        '            if BROWSER_KIND == "local"\n'
        '            else CDPClient(url, additional_headers=cdp_headers() or None)\n'
        '        )',
    ),
    (
        '                return json.loads(urllib.request.urlopen(f"{base_url}/json/version", timeout=5).read())'
        '["webSocketDebuggerUrl"]',
        '                request = urllib.request.Request(f"{base_url}/json/version", headers=cdp_headers())\n'
        '                return json.loads(urllib.request.urlopen(request, timeout=5).read())["webSocketDebuggerUrl"]',
    ),
    (
        "def get_ws_url():",
        'def cdp_headers():\n'
        '    """Headers for the CDP endpoint, from BU_CDP_HEADERS as a JSON object.\n'
        '\n'
        '    A hosted endpoint authenticates the handshake rather than the URL, so the token cannot\n'
        '    travel in BU_CDP_WS. Anything unreadable is treated as unset: a malformed value must not\n'
        '    silently become a header, and the handshake failure that follows names the endpoint\n'
        '    rather than the setting.\n'
        '    """\n'
        '    raw = os.environ.get("BU_CDP_HEADERS", "").strip()\n'
        '    if not raw:\n'
        '        return {}\n'
        '    try:\n'
        '        headers = json.loads(raw)\n'
        '    except ValueError:\n'
        '        log("BU_CDP_HEADERS is not valid JSON; connecting without extra headers")\n'
        '        return {}\n'
        '    if not isinstance(headers, dict):\n'
        '        log("BU_CDP_HEADERS is not a JSON object; connecting without extra headers")\n'
        '        return {}\n'
        '    return {str(name): str(value) for name, value in headers.items()}\n'
        '\n'
        '\n'
        'def get_ws_url():',
    ),
)


def main():
    daemon = pathlib.Path(browser_harness.__file__).with_name("daemon.py")
    source = daemon.read_text()
    if "def cdp_headers()" in source and "browser_session" in source:
        print(f"already patched: {daemon}")
        return 0
    if "def cdp_headers()" in source:
        print(f"cannot patch {daemon}: partly patched already; reinstall browser-harness first")
        return 1
    for old, new in EDITS:
        if source.count(old) != 1:
            print(f"cannot patch {daemon}: expected exactly one of {old[:60]!r}, found {source.count(old)}")
            return 1
        source = source.replace(old, new)
    # Written as a new file and moved into place, never edited where it lies: uv hardlinks
    # installed files from its global cache, so writing in place edits the cache too and the
    # patch escapes into every other environment that installs this version.
    spare = daemon.with_suffix(".patched")
    spare.write_text(source)
    spare.replace(daemon)
    print(f"patched {daemon}")
    print('set BU_CDP_HEADERS to a JSON object, e.g. ' + json.dumps({"Authorization": "Bearer <token>"}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
