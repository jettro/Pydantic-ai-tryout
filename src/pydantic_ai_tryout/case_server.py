import json
from mcp.server.fastmcp import FastMCP

from pydantic_ai_tryout.case_agent import Case, CaseRepository


def create_case_server(repository: CaseRepository | None = None) -> FastMCP:
    """Create a FastMCP server instance backed by a CaseRepository."""
    repo = repository if repository is not None else CaseRepository()
    server = FastMCP(
        name="Case Repository Server",
        instructions="MCP server providing access to support cases and case prioritization tools and resources.",
    )

    @server.tool()
    async def get_case(case_id: str) -> Case:
        """Retrieve details of a support case by its unique case_id."""
        return await repo.get_case(case_id)

    @server.tool()
    async def update_case_priority(case_id: str, priority: int) -> Case:
        """Update the priority of a case (1 to 5, where 1 is highest priority)."""
        if not 1 <= priority <= 5:
            raise ValueError(f"A priority of {priority} is not allowed, must be between 1 and 5.")
        return await repo.update_case_priority(case_id, priority)

    @server.tool()
    async def list_cases() -> list[Case]:
        """List all support cases currently in the repository."""
        return [c.model_copy(deep=True) for c in repo.cases.values()]

    @server.resource("cases://all")
    async def get_all_cases() -> str:
        """Resource listing all cases in JSON format."""
        cases = [c.model_dump() for c in repo.cases.values()]
        return json.dumps(cases, indent=2)

    @server.resource("cases://{case_id}")
    async def get_case_resource(case_id: str) -> str:
        """Resource returning JSON representation of a single case by case_id."""
        case = await repo.get_case(case_id)
        return case.model_dump_json(indent=2)

    @server.prompt()
    def prioritize_case(case_id: str) -> str:
        """Generate a prompt template to prioritize a specific support case."""
        return (
            f"Please review support case '{case_id}'. "
            f"Use the get_case tool to fetch its details, evaluate its urgency, "
            f"and assign an appropriate priority (1 to 5) if it is not yet prioritized."
        )

    return server


mcp = create_case_server()


if __name__ == "__main__":
    mcp.run()
