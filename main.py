import argparse
import asyncio

import logfire
from dotenv import load_dotenv

from pydantic_ai_tryout.case_agent import (CaseAgentDeps, CaseRepository, CaseResponse, PriorityRejected,
                                           create_case_agent)
from pydantic_ai_tryout.friendly_agent import create_friendly_agent


async def main():
    agent = create_friendly_agent()
    result = await agent.run(
        user_prompt="I am disappointed in you, you do not remember anything that I tell you. Can you do better?"
    )
    print(result.output)


async def main_case(case_id: str):
    agent = create_case_agent()
    result = await agent.run(
        user_prompt=(f"You goal is to prioritise the case, but only if it is not yet prioritized."
                     f"You can determine a case is prioritized by checking if case_priority is greater than 0."
                     f"A case can have a priority of 1 to 5, where 1 is highest priority."
                     f"Explain on what basis you prioritize the case."
                     f"Use your tools to get the case details and determine if it is prioritized."
                     f"Finish by storing the priority you determined, or by rejecting the prioritization."),
        deps=CaseAgentDeps(
            case_id=case_id,
            user_name="Jettro",
            case_repository=CaseRepository()
        )
    )
    match result.output:
        case CaseResponse() as response:
            print(f"Priority {response.case.case_priority}: {response.message}")
        case PriorityRejected() as rejected:
            print(f"Not prioritized: {rejected.reason}")

    print("\n--- Details ---")
    usage = result.usage
    print(f"Cost: {usage.cost}")
    print(f"reasoning tokens: {usage.details['reasoning_tokens']}")
    print(f"input tokens: {usage.input_tokens}")
    print(f"output tokens: {usage.output_tokens}")
    print(f"output reasoning tokens: {usage.output_reasoning_tokens}")
    print(f"Requests: {usage.requests}")
    print(f"tool calls: {usage.tool_calls}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prioritise a case with the case agent.")
    parser.add_argument("case_id", nargs="?", default="case_1",
                        help="The id of the case to prioritize, for example case_4 (default: case_1).")
    return parser.parse_args()


if __name__ == "__main__":
    load_dotenv()

    logfire.configure()
    logfire.instrument_system_metrics()
    logfire.instrument_pydantic_ai()

    # asyncio.run(main())

    args = parse_args()

    asyncio.run(main_case(args.case_id))
