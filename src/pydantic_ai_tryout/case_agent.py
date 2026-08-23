import asyncio
from dataclasses import dataclass

from pydantic import BaseModel
from pydantic_ai import Agent, RunContext


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
                           case_creation_date="2023-09-17", case_creator="Bob Johnson")
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
    async def get_case_details(ctx: RunContext[CaseAgentDeps]) -> Case:
        case = await ctx.deps.case_repository.get_case(ctx.deps.case_id)
        return case

    @agent.tool()
    async def store_case_priority(ctx: RunContext[CaseAgentDeps], new_priority: int) -> None:
        await ctx.deps.case_repository.update_case_priority(ctx.deps.case_id, new_priority)

    return agent
