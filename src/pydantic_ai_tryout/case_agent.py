import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from pydantic import BaseModel
from pydantic_ai import Agent, ModelRetry, RunContext, ToolOutput, UnexpectedModelBehavior


class Case(BaseModel):
    case_id: str
    case_description: str
    case_creation_date: str
    case_creator: str
    case_priority: int = 0


@dataclass
class CaseResponse:
    case: Case
    message: str


class PriorityRejected(BaseModel):
    """Use me when the case must not be prioritized, for example because it already has a priority."""

    reason: str


class CaseRepository:
    def __init__(self):
        self.cases = {
            "case_1": Case(case_id="case_1",
                           case_description="There is a problem with the printer on the first floor, there is no paper coming out of it.",
                           case_creation_date="2023-09-15", case_creator="John Doe"),
            "case_2": Case(case_id="case_2", case_description="The network is down on the second floor.",
                           case_creation_date="2023-09-16", case_creator="Jane Smith"),
            "case_3": Case(case_id="case_3",
                           case_description="The coffee machine broke down, no coffee for our software engineers.",
                           case_creation_date="2023-09-17", case_creator="Bob Johnson"),
            "case_4": Case(case_id="case_4",
                           case_description="My laptop is to slow to work with local models.",
                           case_creation_date="2023-09-17", case_creator="Unhappy Engineer",
                           case_priority=2)
        }

    async def get_case(self, case_id: str) -> Case:
        await asyncio.sleep(0.1)  # simulate the latency of a remote case store
        found_case = self.cases.get(case_id)
        if found_case is None:
            raise ValueError(f"Case with ID {case_id} not found.")
        return found_case.model_copy(deep=True)

    async def update_case_priority(self, case_id: str, priority: int) -> Case:
        await asyncio.sleep(0.1)  # simulate the latency of a remote case store
        found_case = self.cases.get(case_id)
        if found_case is None:
            raise ValueError(f"Case with ID {case_id} not found.")
        found_case.case_priority = priority
        return found_case.model_copy(deep=True)


@dataclass
class CaseAgentDeps:
    case_id: str
    user_name: str
    case_repository: CaseRepository


message_agent = Agent(
    model="openai:gpt-5.6-luna",
    output_type=str,
    defer_model_check=True,
    instructions="You turn an internal case note into a short, friendly message for the reporter of the case.",
)


async def case_to_prioritize(ctx: RunContext[CaseAgentDeps], new_priority: int) -> Case:
    """Validate the priority and the case, raise a ModelRetry when the model has to try again."""
    if not 1 <= new_priority <= 5:
        raise ModelRetry(f"A priority of {new_priority} is not allowed, pick a number from 1 to 5.")

    case = await ctx.deps.case_repository.get_case(ctx.deps.case_id)
    if case.case_priority > 0:
        raise ModelRetry(f"Case {case.case_id} already has priority {case.case_priority}, "
                         f"return a {PriorityRejected.__name__} instead.")
    return case


def build_case_agent(
        output_function: Callable[[RunContext[CaseAgentDeps], int, str], Awaitable[CaseResponse]]
) -> Agent[CaseAgentDeps, CaseResponse | PriorityRejected]:
    agent = Agent[CaseAgentDeps, CaseResponse | PriorityRejected](
        model="openai:gpt-5.6-luna",
        deps_type=CaseAgentDeps,
        output_type=[ToolOutput(output_function, name="store_priority", max_retries=2), PriorityRejected],
        retries={"output": 2},
    )

    @agent.instructions
    def personalize(ctx: RunContext[CaseAgentDeps]):
        return (f"You are handling the case together with {ctx.deps.user_name}."
                f"Determine the priority with the case details, then finish the run by storing the priority."
                f"When the case must not be prioritized, finish with a {PriorityRejected.__name__} instead.")

    @agent.tool()
    async def get_case_details(ctx: RunContext[CaseAgentDeps]) -> Case:
        case = await ctx.deps.case_repository.get_case(ctx.deps.case_id)
        return case

    return agent


def create_case_agent() -> Agent[CaseAgentDeps, CaseResponse | PriorityRejected]:
    async def store_priority(ctx: RunContext[CaseAgentDeps], new_priority: int, motivation: str) -> CaseResponse:
        """Store the priority you determined for the case and finish the run."""
        case = await case_to_prioritize(ctx, new_priority)
        updated_case = await ctx.deps.case_repository.update_case_priority(case.case_id, new_priority)
        return CaseResponse(case=updated_case, message=motivation)

    return build_case_agent(store_priority)


def create_case_agent_with_hand_off() -> Agent[CaseAgentDeps, CaseResponse | PriorityRejected]:
    async def hand_off_to_message_agent(ctx: RunContext[CaseAgentDeps], new_priority: int,
                                       motivation: str) -> CaseResponse:
        """Store the priority and let the message agent phrase the reply to the reporter of the case."""
        case = await case_to_prioritize(ctx, new_priority)
        updated_case = await ctx.deps.case_repository.update_case_priority(case.case_id, new_priority)

        # drop the output tool call, it should not be passed on to the other agent
        messages = ctx.messages[:-1]
        try:
            result = await message_agent.run(motivation, message_history=messages)
        except UnexpectedModelBehavior as e:
            if (cause := e.__cause__) and isinstance(cause, ModelRetry):
                raise ModelRetry(f"The message agent failed: {cause.message}") from e
            raise

        return CaseResponse(case=updated_case, message=result.output)

    return build_case_agent(hand_off_to_message_agent)
