import os

# The agents resolve the OpenAI model when they are created, the module level message agent already does that
# on import. A fake key keeps the tests offline, no request leaves the machine.
os.environ.setdefault("OPENAI_API_KEY", "test-key")
