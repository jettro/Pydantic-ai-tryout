from pydantic_ai import Agent


def create_friendly_agent():
    return Agent(
        name="Friendly Agent",
        description="A friendly agent that helps users with their tasks.",
        model="openai:gpt-5.6-luna",
        instructions="Always reply with a friendly message, you can never answer in a bad langauge."
    )
