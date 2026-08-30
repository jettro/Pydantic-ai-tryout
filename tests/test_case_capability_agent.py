import pytest
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

from pydantic_ai_tryout.case_agent import CaseRepository
from pydantic_ai_tryout.case_capability_agent import (
    create_case_capability_agent,
    create_case_mcp_capability,
    create_case_mcp_capability_agent,
)
from pydantic_ai_tryout.case_server import create_case_server


@pytest.fixture
def custom_repo() -> CaseRepository:
    return CaseRepository()


@pytest.fixture
def case_server(custom_repo: CaseRepository):
    return create_case_server(custom_repo)


def test_create_case_mcp_capability(case_server):
    capability = create_case_mcp_capability(server=case_server)
    assert capability.native is False
    assert capability.local is not None


def test_create_case_mcp_capability_with_url():
    capability = create_case_mcp_capability(url="http://localhost:8000/sse")
    assert capability.native is False
    assert capability.url == "http://localhost:8000/sse"


@pytest.mark.asyncio
async def test_case_capability_agent_retrieves_and_updates_case(case_server, custom_repo):
    async def scripted_model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        step = len([m for m in messages if isinstance(m, ModelResponse)])
        if step == 0:
            return ModelResponse(parts=[ToolCallPart("get_case", {"case_id": "case_1"})])
        if step == 1:
            return ModelResponse(parts=[ToolCallPart("update_case_priority", {"case_id": "case_1", "priority": 2})])
        return ModelResponse(parts=[TextPart("Case 1 has been prioritized with level 2.")])

    agent = create_case_capability_agent(server=case_server)

    with agent.override(model=FunctionModel(scripted_model)):
        result = await agent.run("Please prioritize case_1")

    assert "Case 1 has been prioritized with level 2." in result.output
    updated_case = await custom_repo.get_case("case_1")
    assert updated_case.case_priority == 2


@pytest.mark.asyncio
async def test_case_mcp_capability_alias(case_server):
    agent = create_case_mcp_capability_agent(server=case_server)
    assert agent is not None
    mcp_caps = [cap for cap in agent.root_capability.capabilities if hasattr(cap, "native")]
    assert len(mcp_caps) == 1
    assert mcp_caps[0].native is False
