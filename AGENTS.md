# Jev Ultrafaster

Read README.md before editing. Keep the loop small: page -> indexed elements -> operation + target
-> execution.

This fork drives live third-party sites - a customer's WordPress after a plugin update, say - with
no prior knowledge of their markup, and reports what it found to another program. Read-only runs,
host limits, page diagnostics and replay exist for that caller, not for a person watching.

## The loop

- The input is one natural-language goal. Do not add site-specific plans or hardcoded field values.
- TypeSafe chooses an operation and operation-specific target heads in one request. Consume only
  the selected operation's target.
- Targets must map to observed elements and supported operations. Never let the model emit
  selectors or executable code. A caller's own JavaScript is not covered by this: `queries` runs
  what the caller wrote, which is the point of it.
- TYPE_TEXT invokes the text LLM. `field_values` asks for a step's fields in one call; a value that
  does not hold up is left out and that field falls back to its own call.
- Never retry a browser mutation. Log execution before observing its result.
- A decision may start while the page is still settling. Keep it only when the settled page is the
  one it was asked about; acting on an unsettled page is what that rule prevents.
- `fresh` refuses a decision the page has moved under. Past `RESTLESS` a page that will not hold
  still is acted on anyway, because a clock or a progress figure never satisfies that check.
- BLOCKED is withheld while a scroll down still moves the page. What releases it is a scroll that
  moved nothing, never a fraction of the page: a page that loads as it is scrolled grows as fast
  as it is read.

## Reporting to a caller

- A run ends in `done`, `blocked`, `budget` or `off_site`. A budget reached is an outcome, not an
  exception. `blocked` carries `reason`: `not_found` where the page was read to the end, `stuck`
  where part of it never was.
- `evidence` names what answered the goal, taken from the observed node, plus the document's
  status, content type, title and first heading. Never let the model write it.
- `faults` carries the page's console errors and warnings, uncaught exceptions and failed
  requests, collected from CDP events already being drained. No round trip per step.
- `replay.report(state)` is what goes back to a planner: labels, positions and probabilities,
  bounded, with no page text, element ids or markup. Nothing in it may scale with the page.
- `replay.script(state)` names controls by label and kind. A script carries no element ids: a node
  id identifies nothing on a page loaded again.

## Running against a browser

- One browser per daemon. The daemon drains events as one buffer with no per-session filter, so a
  second browser takes the first one's: diagnostics would be attributed to the wrong run. A second
  `Browser` raises `DaemonBusy`.
- Every run gets its own browser context, disposed at close, so one customer's cookies and logins
  do not reach the next.
- `keep_open_s` leaves the page open and puts a handle in the result. The caller keeps the
  registry; jev keeps no timer. `session.close_handle` works from another process and is a no-op
  when the page is already gone. `Agent.free` closes the page and gives the daemon back.
- `BU_CDP_WS` means a hosted endpoint, where every look is a round trip: the wait runs inside the
  page, no screencast is started, and the settle timeout is shorter. Leave local behaviour alone
  when changing either.
- Frames are found for the reading a decision is taken from, never for the readings taken while
  waiting. Their offsets move with the page, so the answer cannot be cached.
- browser-harness 0.1.13 sends no custom headers and does not keep the handshake's
  `cf-browser-session-id`. `scripts/patch_browser_harness.py` adds both; `uv sync` undoes it.
  The patch must be written as a new file and moved into place, because uv hardlinks installed
  files from its cache and editing in place changes every other environment too.

## Limits a caller sets

`read_only` withholds TYPE_TEXT, SELECT and submitting CLICK from the action space, not from the
prompt: an operation that is not offered cannot be chosen. It covers what jev does, not what the
caller's `queries` do. `allowed_hosts` matches on suffix and stops the run `off_site` with the
address that was refused. `max_steps` bounds the actions. The viewport is `JEV_VIEWPORT_WIDTH` and
`JEV_VIEWPORT_HEIGHT`.

## Evidence and claims

- Keep credentials server-side and .env ignored. Tests must not call paid APIs.
- Verify actual final outcomes independently. A DONE choice is not proof of success.
- Compare arms interleaved within one batch. Drift between batches on a live site is larger than
  most differences worth reporting, and a figure from one batch cannot be compared with another's.
- Keep examples, README claims, raw evidence, and model-call counts consistent.
- Do not commit or push unless the user requests it.

Checks: `uv run ruff check .`, `uv run pytest`, `node --check jev_ultrafast/static/app.js`,
`node --check jev_ultrafast/snapshot.js`, `uv build`.
