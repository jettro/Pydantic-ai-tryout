import pytest
from pydantic_ai.messages import ModelMessage, ModelResponse, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel

from pydantic_ai_tryout.case_agent import CaseAgentDeps, CaseRepository, CaseResponse, create_case_agent

USER_PROMPT = ("You goal is to prioritise the case, but only if it is not yet prioritized."
               "Use your tools to get the case details and determine if it is prioritized.")


@pytest.fixture
def case_repository() -> CaseRepository:
    return CaseRepository()


@pytest.fixture
def case_agent_deps(case_repository: CaseRepository) -> CaseAgentDeps:
    return CaseAgentDeps(case_id="case_3", user_name="Jettro", case_repository=case_repository)


def called_tools(messages: list[ModelMessage]) -> set[str]:
    return {part.tool_name
            for message in messages if isinstance(message, ModelResponse)
            for part in message.parts if isinstance(part, ToolCallPart)}


@pytest.mark.asyncio
async def test_get_case_returns_a_copy_of_the_stored_case(case_repository: CaseRepository):
    case = await case_repository.get_case("case_3")
    assert case.case_creator == "Bob Johnson"
    assert case.case_priority == 0

    case.case_priority = 5

    assert (await case_repository.get_case("case_3")).case_priority == 0


@pytest.mark.asyncio
async def test_get_case_raises_for_an_unknown_case(case_repository: CaseRepository):
    with pytest.raises(ValueError):
        await case_repository.get_case("case_42")


@pytest.mark.asyncio
async def test_update_case_priority_stores_the_new_priority(case_repository: CaseRepository):
    updated_case = await case_repository.update_case_priority("case_3", 2)

    assert updated_case.case_priority == 2
    assert (await case_repository.get_case("case_3")).case_priority == 2


@pytest.mark.asyncio
async def test_update_case_priority_raises_for_an_unknown_case(case_repository: CaseRepository):
    with pytest.raises(ValueError):
        await case_repository.update_case_priority("case_42", 2)


@pytest.mark.asyncio
async def test_agent_run_calls_both_async_tools(case_agent_deps: CaseAgentDeps):
    agent = create_case_agent()

    with agent.override(model=TestModel()):
        result = await agent.run(user_prompt=USER_PROMPT, deps=case_agent_deps)

    assert isinstance(result.output, CaseResponse)
    assert called_tools(result.all_messages()) >= {"get_case_details", "store_case_priority"}


@pytest.mark.asyncio
async def test_agent_run_stores_the_priority_chosen_by_the_model(case_agent_deps: CaseAgentDeps,
                                                                case_repository: CaseRepository):
    prioritized_case = (await case_repository.get_case("case_3")).model_copy(update={"case_priority": 3})

    async def scripted_model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        step = len([message for message in messages if isinstance(message, ModelResponse)])
        if step == 0:
            return ModelResponse(parts=[ToolCallPart("get_case_details", {})])
        if step == 1:
            return ModelResponse(parts=[ToolCallPart("store_case_priority", {"new_priority": 3})])
        return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name,
                                                 {"case": prioritized_case.model_dump(),
                                                  "message": "The coffee machine blocks the engineers."})])

    agent = create_case_agent()

    with agent.override(model=FunctionModel(scripted_model)):
        result = await agent.run(user_prompt=USER_PROMPT, deps=case_agent_deps)

    assert result.output.case.case_priority == 3
    assert result.output.message == "The coffee machine blocks the engineers."
    assert (await case_repository.get_case("case_3")).case_priority == 3
