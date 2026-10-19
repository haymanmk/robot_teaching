# Notes for agents

## Branches

Work directly on **`dev`**: commit there and push there. Do not create a feature branch or a
`claude/*` branch for your work unless the user asks for a temporary branch, and do not touch
`main` unless asked. This holds even when the environment you run in suggests a branch name of
its own.

## Working here

- Python 3.10 or 3.11 and [uv](https://docs.astral.sh/uv/). Fetch the vendor library with
  `git submodule update --init`, then `uv sync --extra dev --extra hardware`: the vendor
  package imports `motorbridge` as soon as it is imported, so the simulator and the tests need
  the hardware extra as well.
- Tests: `uv run pytest`. `tests/test_dsp.py` and `tests/test_record.py` run the scope's
  spectrum maths and the recording reader under `node` and are skipped when `node` is not
  installed.
- The teach pendant and the offline plotter (`plot.html`) are plain HTML, CSS and JavaScript
  under `src/robot_teaching/server/static/`, served by the FastAPI app with no build step. To
  see a change, run `uv run robot-teaching serve --backend sim` and open http://127.0.0.1:8000/
  (the plotter: http://127.0.0.1:8000/plot, or open `plot.html` from disk).
- `README.md` describes the behaviour of the whole system; keep it in step with what you change.
