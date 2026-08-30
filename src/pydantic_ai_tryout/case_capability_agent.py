from mcp.server.fastmcp import FastMCP
from pydantic_ai import Agent
from pydantic_ai.capabilities import MCP

from pydantic_ai_tryout.case_server import create_case_server


def create_case_mcp_capability(
    server: FastMCP | None = None,
    *,
    url: str | None = None,
) -> MCP:
    """Create a local MCP capability configured for the Case MCP server without provider-native dispatch."""
    if url is not None:
        return MCP(url=url, native=False)

    mcp_server = server if server is not None else create_case_server()
    return MCP(local=mcp_server, native=False)


def create_case_capability_agent(
    server: FastMCP | None = None,
    capability: MCP | None = None,
    model: str = "openai:gpt-5.6-luna",
) -> Agent:
    """Create a Pydantic AI agent configured to use the Case MCP server via the MCP Capability."""
    mcp_capability = (
        capability if capability is not None else create_case_mcp_capability(server=server)
    )

    agent = Agent(
        model=model,
        capabilities=[mcp_capability],
        instructions=(
            "You are a case prioritization assistant. "
            "Use the MCP tools to inspect support cases and update their priority when needed. "
            "Priority must be between 1 (highest) and 5 (lowest). "
            "If a case already has a priority > 0, do not modify it."
        ),
    )
    return agent


# Alias for alternative naming
create_case_mcp_capability_agent = create_case_capability_agent
