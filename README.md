# Jev Ultrafaster
If you want a stable, supported version of jev-ultrafast you should head to the [upstream repository](https://github.com/browser-use/jev-ultrafast). 
This fork exists to fill some gaps that I need covered in order to replace an existing internal tool in our suite.

From my limited testing, at the moment my version is faster on static pages and more reliable on pages with a lot of DOM rewrites (SPAs, wizards etc).
| task | arm | runs | passed | mean (passing) | Jev calls | Mercury calls |
|---|---|---|---|---|---|---|
| Google flights | upstream | 20 | 2 | 16.4s | 3-5 (aborts) | 0-2 |
| Google flights | **this** | 5 | **5** | **8.2s** | 13 used (~18 issued) | **1** |
| Static form, 5 fields | upstream | 1 | 1 | 5.39s | 6 | 5 |
| Static form, 5 fields | **this** | 1 | **1** | **2.5s** | 2 | **1** |
| Endless spinner | upstream | 1 | 1 | **1.82s** | 3 | 1 |
| Endless spinner | **this** | 1 | **1** | 1.92s | 3 | 1 |
| DOM-rewriting ticker | upstream | 1 | **0** | (stalls) | 120 | 0 |
| DOM-rewriting ticker | **this** | 1 | **1** | **7.9s** | 3 | 1 |

On top of the speed and reliability changes, I also needed and added
- Support for Cloudflare's [Browser Run CDP](https://developers.cloudflare.com/browser-run/cdp/)
- Post-run reporting
- Replay functionality
- Session isolation handling
- Read-only mode
- Host allow-lists checks during navigation

## Try it

```bash
git clone https://github.com/Fox-Islam/jev-ultrafaster.git
cd jev-ultrafaster
uv sync
cp .env.example .env
# Add TYPESAFE_API_KEY and TEXT_MODEL_API_KEY.
uv run jev
```

Open **http://127.0.0.1:8766** and click **Start demo → Run automatically**. The inspector shows numbered elements, operation probabilities, target probabilities, and executed actions. **Choose next** pauses before execution.

Chrome connects through [Browser Harness](https://github.com/browser-use/browser-harness), installed by `uv sync`. Run `uv run browser-harness --doctor` if it needs connecting. Allow remote debugging in Chrome when prompted.

### A hosted browser

`BU_CDP_WS` points the harness at a remote CDP endpoint instead of local Chrome. An endpoint that
authenticates the WebSocket handshake with a header, such as
[Cloudflare Browser Rendering](https://developers.cloudflare.com/browser-run/cdp/), needs
`BU_CDP_HEADERS` as well:

```bash
export BU_CDP_WS='wss://<endpoint>/devtools/browser'
export BU_CDP_HEADERS='{"Authorization": "Bearer <api-token>"}'
```

browser-harness 0.1.13 builds its CDP connection without headers, so the setting needs
`uv run python scripts/patch_browser_harness.py` first. The script is idempotent, reports whether
it changed anything, and edits the installed package, so `uv sync` undoes it and it has to be run
again. `patches/browser-harness-cdp-headers.patch` is the same change as a diff, for upstreaming.

`TEXT_MODEL_API_KEY` is an OpenRouter key in the example configuration. The current demo uses `inception/mercury-2.5` with reasoning disabled. Gemini, GLM, and DeepSeek can also use the OpenAI-compatible text helper; configure the appropriate model, endpoint, and reasoning setting.

## Use the library

```python
from jev_ultrafast import Agent

with Agent(
    "https://www.google.com/travel/flights?hl=en",
    "Find one-way flights from Zurich to London on September 20, 2026, "
    "for one adult in economy. Stop when matching flight options are visible.",
) as agent:
    for state in agent.run():
        print(state["elapsed_ms"], state["status"])
```

Run with `uv run --env-file .env python your_script.py`. The same policy can run a different task:

```bash
uv run --env-file .env python examples/run.py \
  --url https://en.wikipedia.org/wiki/Main_Page \
  --goal 'Find and open the Wikipedia article about Gödel’s incompleteness theorems.'
```

`uv run --env-file .env python examples/flights.py --keep-open` performs the flight search, checks the actual route/date/results, and saves its trace. It does not select or book a flight.

## Development

```bash
uv run ruff check .
uv run pytest
node --check jev_ultrafast/static/app.js
node --check jev_ultrafast/snapshot.js
uv build
```

Tests are offline. `uv run python scripts/check_guards.py` checks real controls in a local browser without model calls. Live examples and recording scripts make paid API calls. `scripts/record_flights.py <new-folder>` captures original browser timestamps; `scripts/render_demo.py <recording-folder>` renders that verified run at 1× and crops out the Google account strip. Credentials and raw traces stay ignored.
