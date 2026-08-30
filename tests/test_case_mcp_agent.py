import pytest
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

from pydantic_ai_tryout.case_agent import CaseRepository
from pydantic_ai_tryout.case_mcp_agent import create_case_mcp_agent
from pydantic_ai_tryout.case_server import create_case_server


@pytest.fixture
def custom_repo() -> CaseRepository:
    return CaseRepository()


@pytest.fixture
def case_server(custom_repo: CaseRepository):
    return create_case_server(custom_repo)


@pytest.mark.asyncio
async def test_case_mcp_agent_retrieves_and_updates_case(case_server, custom_repo):
    async def scripted_model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        step = len([m for m in messages if isinstance(m, ModelResponse)])
        if step == 0:
            return ModelResponse(parts=[ToolCallPart("get_case", {"case_id": "case_1"})])
        if step == 1:
            return ModelResponse(parts=[ToolCallPart("update_case_priority", {"case_id": "case_1", "priority": 2})])
        return ModelResponse(parts=[TextPart("Case 1 has been prioritized with level 2.")])

    agent = create_case_mcp_agent(server=case_server)

    with agent.override(model=FunctionModel(scripted_model)):
        result = await agent.run("Please prioritize case_1")

    assert "Case 1 has been prioritized with level 2." in result.output
    updated_case = await custom_repo.get_case("case_1")
    assert updated_case.case_priority == 2
