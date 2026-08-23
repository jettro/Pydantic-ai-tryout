import asyncio

import logfire
from dotenv import load_dotenv

from pydantic_ai_tryout.case_agent import create_case_agent, CaseAgentDeps, CaseRepository
from pydantic_ai_tryout.friendly_agent import create_friendly_agent


async def main():
    agent = create_friendly_agent()
    result = await agent.run(
        user_prompt="I am disappointed in you, you do not remember anything that I tell you. Can you do better?"
    )
    print(result.output)


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


if __name__ == "__main__":
    load_dotenv()

    logfire.configure()
    logfire.instrument_system_metrics()
    logfire.instrument_pydantic_ai()

    # asyncio.run(main())
    asyncio.run(main_case())
