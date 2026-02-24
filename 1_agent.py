# =============================================================
# CONCEPT 1: AGENT
# =============================================================
# File: 1_agent.py
# Run: python 1_agent.py
#
# WHAT IS AN AGENT?
# -----------------
# An Agent is an AI entity powered by a Large Language Model (LLM).
# It receives a set of INSTRUCTIONS (its "personality" or "role")
# and a USER MESSAGE (the task to complete).
#
# Think of an Agent like hiring a specialist employee:
#   - You give them a JOB DESCRIPTION (instructions)
#   - You give them a TASK (user message)
#   - They return a RESULT using the LLM's intelligence
#
# KEY PROPERTIES OF AN AGENT:
#   name         → A label to identify the agent
#   instructions → System-level directions that shape its behavior
#   model        → The LLM model it uses to generate responses
#
# In this file we create a simple "Explainer Agent" that explains
# any topic in plain, simple language.
# =============================================================

import asyncio
# Import our shared Groq configuration (sets up the LLM client)
import groq_setup 
from groq_setup import MODEL

# Import the two core classes from the OpenAI Agents SDK
from agents import Agent, Runner


# ── Define the Agent ──────────────────────────────────────────
# `instructions` is the SYSTEM PROMPT — it tells the LLM how to
# behave. This is like a job description for your AI employee.
explainer_agent = Agent(
    name="Simple Explainer",                        # Agent's label
    instructions=(
        "You are a friendly teacher who explains topics "
        "in very simple English. Always use one real-world "
        "analogy and keep your answer under 5 sentences."
    ),
    model=MODEL,                                    # Groq LLM model
)


# ── Run the Agent ─────────────────────────────────────────────
# Runner.run() is the function that sends the user's message
# to the agent and waits for the LLM's response.
# It is an ASYNC function, so we wrap it in asyncio.run().
async def main():
    print("An Agent = LLM + Instructions + a Task to do\n")

    topic = "What is an API?"  # The message we send to the agent

    print(f"[User]  → {topic}")

    # Runner.run() sends the message to the agent and returns a result
    result = await Runner.run(
        starting_agent=explainer_agent,   # Which agent handles this
        input=topic,                       # The user's message / task
    )

    # result.final_output contains the agent's text response
    print(f"[Agent Response]\n{result.final_output}")


# ── Entry point ───────────────────────────────────────────────
# This guard ensures the script only runs when executed directly.
# It won't run if this file is imported by another script.
if __name__ == "__main__":
    asyncio.run(main())
