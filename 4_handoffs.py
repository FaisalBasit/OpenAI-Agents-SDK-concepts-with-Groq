# =============================================================
# CONCEPT 4: HANDOFFS
# =============================================================
# File: 4_handoffs.py
# Run: python 4_handoffs.py
#
# WHAT ARE HANDOFFS?
# ------------------
# A Handoff is when one agent passes control to another agent.
# This allows you to build MULTI-AGENT SYSTEMS where each agent
# is a specialist, and they collaborate to handle complex tasks.
#
# REAL-WORLD ANALOGY:
#   Imagine a hospital:
#   - A RECEPTIONIST greets you and decides which doctor you need
#   - A GENERAL DOCTOR handles common issues
#   - A CARDIOLOGIST handles heart problems
#   - A NEUROLOGIST handles brain problems
#   The receptionist "hands off" the patient to the right specialist.
#
# HOW HANDOFFS WORK IN THE SDK:
#   1. Agent A is configured with `handoffs=[AgentB, AgentC]`
#   2. When Agent A decides it needs AgentB's expertise,
#      it "calls" the handoff — like transferring a phone call
#   3. AgentB then takes over and produces the final response
#   4. The Runner manages this entire transfer automatically
#
# In this file:
#   A "Language Router" agent detects the user's language and
#   hands off to either an English or Urdu specialist agent.
# =============================================================

import asyncio

import groq_setup  # noqa: F401
from groq_setup import MODEL

from agents import Agent, Runner


# ── Specialist Agent 1: English Responder ─────────────────────
# This agent ONLY handles English-language queries
english_agent = Agent(
    name="English Specialist",
    instructions=(
        "You are a helpful assistant who ONLY responds in English. "
        "Answer the user's question clearly and concisely in English."
    ),
    model=MODEL,
)

# ── Specialist Agent 2: Urdu Responder ────────────────────────
# This agent ONLY handles Urdu-language queries
urdu_agent = Agent(
    name="Urdu Specialist",
    instructions=(
        "You are a helpful assistant who ONLY responds in Urdu (Roman or script). "
        "Answer the user's question clearly and concisely in Urdu."
    ),
    model=MODEL,
)

# ── Router Agent (the one who decides & hands off) ────────────
# This is the ENTRY POINT agent.
# It reads the user message, detects the language, and
# immediately hands off to the correct specialist.
#
# KEY POINT: `handoffs=[...]` is a list of agents this agent
# is ALLOWED to transfer control to.
router_agent = Agent(
    name="Language Router",
    instructions=(
        "You are a language detection agent. "
        "Your ONLY job is to determine what language the user is writing in, "
        "then immediately transfer to the correct specialist:\n"
        "  - If the user writes in English → hand off to 'English Specialist'\n"
        "  - If the user writes in Urdu (Roman or script) → hand off to 'Urdu Specialist'\n"
        "Do NOT answer the question yourself. Always hand off."
    ),
    model=MODEL,
    handoffs=[english_agent, urdu_agent],   # <-- agents it can route to
)


# ── Entry point ───────────────────────────────────────────────
async def main():
    print("\n" + "=" * 60)
    print("  CONCEPT 4: HANDOFFS")
    print("=" * 60)
    print("Handoffs = Passing control from one Agent to another")
    print("Enables multi-agent collaboration & specialist routing.\n")

    # --- Test 1: English message → should go to English Specialist ---
    msg1 = "What is machine learning?"
    print(f"[Test 1] Input: '{msg1}'")
    print("(Router should detect English → hand off to English Specialist)")

    result1 = await Runner.run(
        starting_agent=router_agent,  # Start with the router
        input=msg1,
    )
    print(f"\n[Final Response]\n{result1.final_output}\n")

    print("-" * 60)

    # --- Test 2: Urdu message → should go to Urdu Specialist ---
    msg2 = "Machine learning kya hoti hai?"
    print(f"[Test 2] Input: '{msg2}'")
    print("(Router should detect Urdu → hand off to Urdu Specialist)")

    result2 = await Runner.run(
        starting_agent=router_agent,
        input=msg2,
    )
    print(f"\n[Final Response]\n{result2.final_output}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
