# Trying out Pydantic AI/Evals/Logfire

## Installation

```bash
# installs through the slim package only what we need
uv add "pydantic-ai-slim[logfire,openai]"

# dotenv for loading the environment variables from the .env file
uv add python-dotenv
```

## The first step

In general, an Agent does not implement a chat interface. With a chat, you send multiple messages and get a reply to the message you send. The agent does not work like that out of the box. You only provide a piece of text, and the agent replies to that piece of text.

```python
from pydantic_ai import Agent

Agent(
    name="Friendly Agent",
    description="A friendly agent that helps users with their tasks.",
    model="openai:gpt-5.6-luna",
    instructions="Always reply with a friendly message, you can never answer in a bad langauge."
)
```

## Case management and the case repository

For the next sample, I create a basic case management system. The goal is to determine the priority of a case. A case repository is available as two tools to the agent. The incoming request asks the agent to determine the priority of a case. Below is the outline of the repository class, the git project has the complete implementation if you are curious.

```python
class CaseRepository:
    def __init__(self):
        self.cases = {
        }

    def get_case(self, case_id: str):
        ...
    def update_case_priority(self, case_id: str, priority: int) -> Case:
        ...
```

## Dependencies

Sometimes you want to keep specific data away from the agent, but make it available to tools. In the example, I want to provide the case_id, this should not be changeable by the Agent. This is one example of data that I can inject as a dependency.  

Another example is database clients, repositories, objects that are use but not maintained by the agent. I want to provide a case repository to the agent. This way, the agent can access the cases without having to know about the repository.

The dependencies are specified by an object. When you call the agent, you provide an instance of the object.

```python
from dataclasses import dataclass
from pydantic_ai import Agent, RunContext

# The dependency object definition
@dataclass
class CaseAgentDeps:
    case_id: str
    user_name: str
    case_repository: CaseRepository


# Tell the agent about the dependency
agent = Agent(
    model="openai:gpt-5.6-luna",
    deps_type=CaseAgentDeps,
    output_type=CaseResponse
)

```

Each method, like a tool call or instructions, receives the RunContext with the dependencies. In the next code block, you can see how to use the RunContext to access the dependencies.

```python
from pydantic_ai import Agent, RunContext

@agent.instructions
def personalize(ctx: RunContext[CaseAgentDeps]):
    return (f"You are handling the case together with {ctx.deps.user_name}."
            f"In your response, always return the case details and a message in the {CaseResponse.__name__} format.")

```

## Using tools, and introduce annotations for configuration

Another way of initializing an Agent is using annotations, these allow you to present more advanced configurations a lot better. Lets dive into a new example. I want to start an Agent by a specific user for a fixed case. So this instance of the agent is not allowed to switch cases. Through the dependency mechanism, I can provide an object with data used by the agent, but also provided to the tools and the mentioned annotations.

The next code block shows the complete construction of the agent.

```python
def create_case_agent():
    agent = Agent(
        model="openai:gpt-5.6-luna",
        deps_type=CaseAgentDeps,
        output_type=CaseResponse
    )

    @agent.instructions
    def personalize(ctx: RunContext[CaseAgentDeps]):
        return (f"You are handling the case together with {ctx.deps.user_name}."
                f"In your response, always return the case details and a message in the {CaseResponse.__name__} format.")

    @agent.tool()
    def get_case_details(ctx: RunContext[CaseAgentDeps]) -> Case:
        case = ctx.deps.case_repository.get_case(ctx.deps.case_id)
        return case

    @agent.tool()
    def store_case_priority(ctx: RunContext[CaseAgentDeps], new_priority: int) -> None:
        ctx.deps.case_repository.update_case_priority(ctx.deps.case_id, new_priority)

    return agent
```

## Calling the agent

Next, you need to call the agent with the user instructions and the required dependencies. Notice how I create the instances in the code block for `CaseAgentDeps` and `CaseRepository`.

```python
async def main_case():
    agent = create_case_agent()
    result = await agent.run(
        user_prompt=(f"You goal is to prioritise the case, but only if it is not yet prioritized."
                     f"You can determine a case is prioritized by checking if case_priority is greater than 0."
                     f"A case can have a priority of 1 to 5, where 1 is highest priority."
                     f"Explain on what basis you prioritize the case."
                     f"Use your tools to get the case details and determine if it is prioritized."
                     f"After prioritizing the case, use the update_case_priority tool to update the case priority."),
        deps=CaseAgentDeps(
            case_id="case_3",
            user_name="Jettro",
            case_repository=CaseRepository()
        )
    )
    print(result.output)
```



## Adding Logfire instrumentation

I prefer to use the environment property for the access to Logfire. This way, I only have to add the key to the .env file.

The following lines are required to ingest telemetry into Logfire. Below is the output of the run.

```text
/opt/homebrew/bin/uv run /Users/jettrocoenradie/Development/personal/pydantic-ai-tryout/.venv/bin/python /Users/jettrocoenradie/Development/personal/pydantic-ai-tryout/main.py 
Logfire project URL: https://logfire-eu.pydantic.dev/jettro/starter-project
09:14:41.040 agent run
09:14:41.042   chat gpt-5.6-luna
09:14:43.458   running tool: get_case_details
09:14:43.460   chat gpt-5.6-luna
09:14:44.982   running tool: store_case_priority
09:14:44.987   chat gpt-5.6-luna
CaseResponse(case=Case(case_id='case_3', case_description='The coffee machine broke down, no coffee for our software engineers.', case_creation_date='2023-09-17', case_creator='Bob Johnson', case_priority=3), message='The case was not previously prioritized (priority 0), so I assigned priority 3, reflecting a moderate operational impact: the broken coffee machine affects the software engineering team but does not indicate a critical business or safety issue.')
```

```python
import logfire

logfire.configure()
logfire.instrument_system_metrics()
logfire.instrument_pydantic_ai()
```

![Screenshot of Logfire trace](images/screenshot-logfire-tool.png)

