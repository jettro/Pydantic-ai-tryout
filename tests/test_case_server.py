import json
import pytest
from pydantic_ai_tryout.case_agent import Case, CaseRepository
from pydantic_ai_tryout.case_server import create_case_server


@pytest.fixture
def custom_repo() -> CaseRepository:
    return CaseRepository()


@pytest.fixture
def case_server(custom_repo: CaseRepository):
    return create_case_server(custom_repo)


@pytest.mark.asyncio
async def test_server_lists_all_tools(case_server):
    tools = await case_server.list_tools()
    tool_names = {tool.name for tool in tools}
    assert tool_names == {"get_case", "update_case_priority", "list_cases"}


@pytest.mark.asyncio
async def test_get_case_tool_success(case_server):
    content, data = await case_server.call_tool("get_case", {"case_id": "case_1"})
    assert data["case_id"] == "case_1"
    assert data["case_creator"] == "John Doe"
    assert data["case_priority"] == 0


@pytest.mark.asyncio
async def test_get_case_tool_not_found(case_server):
    with pytest.raises(Exception):
        await case_server.call_tool("get_case", {"case_id": "case_999"})


@pytest.mark.asyncio
async def test_update_case_priority_tool_success(case_server, custom_repo):
    content, data = await case_server.call_tool("update_case_priority", {"case_id": "case_1", "priority": 3})
    assert data["case_id"] == "case_1"
    assert data["case_priority"] == 3

    # Verify repository was updated
    case = await custom_repo.get_case("case_1")
    assert case.case_priority == 3


@pytest.mark.asyncio
async def test_update_case_priority_tool_invalid_priority(case_server):
    with pytest.raises(Exception):
        await case_server.call_tool("update_case_priority", {"case_id": "case_1", "priority": 9})

    with pytest.raises(Exception):
        await case_server.call_tool("update_case_priority", {"case_id": "case_1", "priority": 0})


@pytest.mark.asyncio
async def test_list_cases_tool(case_server):
    content, data = await case_server.call_tool("list_cases", {})
    cases = data["result"]
    assert len(cases) == 4
    case_ids = {c["case_id"] for c in cases}
    assert case_ids == {"case_1", "case_2", "case_3", "case_4"}


@pytest.mark.asyncio
async def test_get_all_cases_resource(case_server):
    resources = await case_server.read_resource("cases://all")
    assert len(resources) == 1
    raw_json = resources[0].content
    parsed = json.loads(raw_json)
    assert isinstance(parsed, list)
    assert len(parsed) == 4
    assert parsed[0]["case_id"] == "case_1"


@pytest.mark.asyncio
async def test_get_single_case_resource(case_server):
    resources = await case_server.read_resource("cases://case_2")
    assert len(resources) == 1
    parsed = json.loads(resources[0].content)
    assert parsed["case_id"] == "case_2"
    assert parsed["case_creator"] == "Jane Smith"


@pytest.mark.asyncio
async def test_get_single_case_resource_not_found(case_server):
    with pytest.raises(Exception):
        await case_server.read_resource("cases://case_nonexistent")


@pytest.mark.asyncio
async def test_prioritize_case_prompt(case_server):
    prompt = await case_server.get_prompt("prioritize_case", {"case_id": "case_3"})
    assert prompt.description == "Generate a prompt template to prioritize a specific support case."
    assert len(prompt.messages) == 1
    message_text = prompt.messages[0].content.text
    assert "case_3" in message_text
