from mcp.server.fastmcp import FastMCP
from pydantic_ai import Agent
from pydantic_ai.mcp import MCPToolset

from pydantic_ai_tryout.case_server import create_case_server


def create_case_mcp_agent(
    server: FastMCP | None = None,
    model: str = "openai:gpt-5.6-luna",
) -> Agent:
    """Create a Pydantic AI agent configured to use the Case MCP server via MCPToolset."""
    mcp_server = server if server is not None else create_case_server()
    toolset = MCPToolset(mcp_server)

    agent = Agent(
        model=model,
        toolsets=[toolset],
        instructions=(
            "You are a case prioritization assistant. "
            "Use the MCP tools to inspect support cases and update their priority when needed. "
            "Priority must be between 1 (highest) and 5 (lowest). "
            "If a case already has a priority > 0, do not modify it."
        ),
    )
    return agent
