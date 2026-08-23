# Trying out Pydantic AI, Evals and Logfire

A small playground project to learn [Pydantic AI](https://ai.pydantic.dev/). It contains two agents:

- **Friendly agent** — the smallest possible agent, one model and a set of instructions. It shows that an
  agent is not a chat interface: you send one prompt and you get one reply.
- **Case agent** — a tiny case management sample. The agent has to decide the priority of a case (1 is the
  highest, 5 the lowest) and store it. It uses:
  - two typed outputs, so the run ends either with a `CaseResponse` or with a `PriorityRejected` instead of
    free text;
  - an output function, `store_priority`, that validates the priority the model picked, writes it to the
    repository and ends the run, with a `ModelRetry` to send the model back when the priority is wrong;
  - dependencies (`CaseAgentDeps`) to inject the `case_id`, the user name and the `CaseRepository`, so the
    agent cannot pick another case than the one it was started for;
  - one `async` tool, `get_case_details`, that talks to the repository;
  - dynamic instructions that personalise the run with the name of the user;
  - a variant, `create_case_agent_with_hand_off()`, whose output function hands the case over to a second
    agent that phrases the message for the reporter of the case.

Both runs are instrumented with [Logfire](https://logfire.pydantic.dev/), so you can follow the agent run,
the model calls and every tool call in a trace.

The reasoning behind each step is written down in [notes.md](notes.md), which is the source for a blog post
about this project.

## Project layout

```text
main.py                              entry point, runs the friendly or the case agent
src/pydantic_ai_tryout/
    friendly_agent.py                the minimal agent
    case_agent.py                    Case, CaseResponse, PriorityRejected, CaseRepository, CaseAgentDeps,
                                     create_case_agent(), create_case_agent_with_hand_off()
tests/
    test_case_agent.py               tests for the repository and the agent, no model calls
    conftest.py                      sets a fake OPENAI_API_KEY before the agents are created
notes.md                             the write-up, step by step
Makefile                             sync, test and run shortcuts
```

## Requirements

- Python 3.12 or newer
- [uv](https://docs.astral.sh/uv/) for the dependencies and the virtual environment
- An OpenAI API key, and optionally a Logfire token

## Setup

Install the project and the dev dependencies in `.venv`:

```bash
make sync
# or: uv sync --all-groups
```

Create a `.env` file in the root of the project, `main.py` loads it with `python-dotenv`:

```dotenv
OPENAI_API_KEY=sk-...
LOGFIRE_TOKEN=pylf_...
```

The `LOGFIRE_TOKEN` is only needed to send the traces to Logfire. Without it, `logfire.configure()` prints a
warning and the sample still runs.

## Running the sample

```bash
make run
# or: uv run python main.py
```

This calls `main_case()`, which asks the case agent to prioritise `case_3` (the broken coffee machine).
The agent calls `get_case_details`, decides on a priority and finishes through the `store_priority` output
function, which stores the priority and returns the `CaseResponse`. The trace below was recorded with the
older version, where storing the priority was still a tool:

```text
Logfire project URL: https://logfire-eu.pydantic.dev/jettro/starter-project
09:14:41.040 agent run
09:14:41.042   chat gpt-5.6-luna
09:14:43.458   running tool: get_case_details
09:14:43.460   chat gpt-5.6-luna
09:14:44.982   running tool: store_case_priority
09:14:44.987   chat gpt-5.6-luna
CaseResponse(case=Case(case_id='case_3', ..., case_priority=3), message='The case was not previously
prioritized (priority 0), so I assigned priority 3, ...')
```

![Screenshot of Logfire trace](images/screenshot-logfire-tool.png)

To run the friendly agent instead, switch the two last lines of `main.py`:

```python
asyncio.run(main())
# asyncio.run(main_case())
```

Both runs call a real model, so they cost tokens. The model is configured in the agents themselves
(`model="openai:gpt-5.6-luna"`), change it there if you want to use another one.

## Running the tests

```bash
make test
# or: uv run pytest -q
```

The tests never call OpenAI. They replace the model with `TestModel` or a scripted `FunctionModel` through
`agent.override(...)`, so they are free and fast. Next to the happy path, they cover the `ModelRetry` on an
out of range priority, the `PriorityRejected` output and the hand off to the message agent. A fake
`OPENAI_API_KEY` is set at the top of `tests/conftest.py`, because an agent resolves the OpenAI model the
moment it is created, and the module level message agent does that on import.

## Make targets

Run `make` without arguments to see the list:

| Target             | What it does                                          |
|--------------------|-------------------------------------------------------|
| `make sync`        | Install the project and the dev dependencies in `.venv` |
| `make lock`        | Refresh `uv.lock` without touching the environment    |
| `make upgrade`     | Upgrade the locked dependencies and sync them         |
| `make test`        | Run the tests                                         |
| `make test-verbose`| Run the tests with the full output                    |
| `make run`         | Run `main.py`, calls the case agent with the real model |
| `make clean`       | Remove the caches and the build artifacts             |
