import pytest
from pydantic_ai.messages import (ModelMessage, ModelRequest, ModelResponse, RetryPromptPart, TextPart, ToolCallPart)
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel

from pydantic_ai_tryout.case_agent import (CaseAgentDeps, CaseRepository, CaseResponse, PriorityRejected,
                                           create_case_agent, create_case_agent_with_hand_off, message_agent)

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


def retry_prompts(messages: list[ModelMessage]) -> list[RetryPromptPart]:
    return [part
            for message in messages if isinstance(message, ModelRequest)
            for part in message.parts if isinstance(part, RetryPromptPart)]


def rejection_tool_name(info: AgentInfo) -> str:
    return next(tool.name for tool in info.output_tools if tool.name != "store_priority")


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
async def test_the_agent_offers_both_outputs_and_the_case_details_tool(case_agent_deps: CaseAgentDeps):
    output_tools: list[str] = []
    tools: list[str] = []

    async def inspecting_model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        output_tools.extend(tool.name for tool in info.output_tools)
        tools.extend(tool.name for tool in info.function_tools)
        return ModelResponse(parts=[ToolCallPart("store_priority", {"new_priority": 3, "motivation": "no coffee"})])

    agent = create_case_agent()

    with agent.override(model=FunctionModel(inspecting_model)):
        await agent.run(user_prompt=USER_PROMPT, deps=case_agent_deps)

    assert "store_priority" in output_tools
    assert len(output_tools) == 2
    assert tools == ["get_case_details"]


@pytest.mark.asyncio
async def test_agent_run_stores_the_priority_chosen_by_the_model(case_agent_deps: CaseAgentDeps,
                                                                case_repository: CaseRepository):
    async def scripted_model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        step = len([message for message in messages if isinstance(message, ModelResponse)])
        if step == 0:
            return ModelResponse(parts=[ToolCallPart("get_case_details", {})])
        return ModelResponse(parts=[ToolCallPart("store_priority",
                                                 {"new_priority": 3,
                                                  "motivation": "The coffee machine blocks the engineers."})])

    agent = create_case_agent()

    with agent.override(model=FunctionModel(scripted_model)):
        result = await agent.run(user_prompt=USER_PROMPT, deps=case_agent_deps)

    assert isinstance(result.output, CaseResponse)
    assert result.output.case.case_priority == 3
    assert result.output.message == "The coffee machine blocks the engineers."
    assert (await case_repository.get_case("case_3")).case_priority == 3


@pytest.mark.asyncio
async def test_agent_retries_an_out_of_range_priority(case_agent_deps: CaseAgentDeps,
                                                     case_repository: CaseRepository):
    async def scripted_model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        step = len([message for message in messages if isinstance(message, ModelResponse)])
        if step == 0:
            return ModelResponse(parts=[ToolCallPart("get_case_details", {})])
        if step == 1:
            return ModelResponse(parts=[ToolCallPart("store_priority",
                                                     {"new_priority": 9, "motivation": "too high"})])
        return ModelResponse(parts=[ToolCallPart("store_priority",
                                                 {"new_priority": 3, "motivation": "no coffee"})])

    agent = create_case_agent()

    with agent.override(model=FunctionModel(scripted_model)):
        result = await agent.run(user_prompt=USER_PROMPT, deps=case_agent_deps)

    assert result.output.case.case_priority == 3
    assert (await case_repository.get_case("case_3")).case_priority == 3
    assert any("not allowed" in prompt.model_response() for prompt in retry_prompts(result.all_messages()))


@pytest.mark.asyncio
async def test_agent_rejects_a_case_that_is_already_prioritized(case_agent_deps: CaseAgentDeps,
                                                               case_repository: CaseRepository):
    await case_repository.update_case_priority("case_3", 2)

    async def scripted_model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        step = len([message for message in messages if isinstance(message, ModelResponse)])
        if step == 0:
            return ModelResponse(parts=[ToolCallPart("get_case_details", {})])
        return ModelResponse(parts=[ToolCallPart(rejection_tool_name(info),
                                                 {"reason": "The case already has priority 2."})])

    agent = create_case_agent()

    with agent.override(model=FunctionModel(scripted_model)):
        result = await agent.run(user_prompt=USER_PROMPT, deps=case_agent_deps)

    assert isinstance(result.output, PriorityRejected)
    assert result.output.reason == "The case already has priority 2."
    assert (await case_repository.get_case("case_3")).case_priority == 2


@pytest.mark.asyncio
async def test_the_message_agent_phrases_the_response(case_agent_deps: CaseAgentDeps,
                                                     case_repository: CaseRepository):
    async def scripted_model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        step = len([message for message in messages if isinstance(message, ModelResponse)])
        if step == 0:
            return ModelResponse(parts=[ToolCallPart("get_case_details", {})])
        return ModelResponse(parts=[ToolCallPart("store_priority",
                                                 {"new_priority": 3, "motivation": "no coffee for the engineers"})])

    async def phrasing_model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[TextPart("We are on it, coffee is coming back soon!")])

    agent = create_case_agent_with_hand_off()

    with agent.override(model=FunctionModel(scripted_model)), \
            message_agent.override(model=FunctionModel(phrasing_model)):
        result = await agent.run(user_prompt=USER_PROMPT, deps=case_agent_deps)

    assert result.output.message == "We are on it, coffee is coming back soon!"
    assert (await case_repository.get_case("case_3")).case_priority == 3


@pytest.mark.asyncio
async def test_test_model_finishes_through_one_of_the_output_tools(case_agent_deps: CaseAgentDeps):
    agent = create_case_agent()

    with agent.override(model=TestModel(custom_output_args={"new_priority": 3, "motivation": "made up"})):
        result = await agent.run(user_prompt=USER_PROMPT, deps=case_agent_deps)

    assert isinstance(result.output, CaseResponse)
    assert called_tools(result.all_messages()) >= {"get_case_details", "store_priority"}
