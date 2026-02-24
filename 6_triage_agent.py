# =============================================================
# CONCEPT 6: TRIAGE AGENT
# =============================================================
# File: 6_triage_agent.py
# Run: python 6_triage_agent.py
#
# WHAT IS A TRIAGE AGENT?
# -----------------------
# A Triage Agent is a COORDINATOR or ROUTER agent.
# It is the FIRST agent a user interacts with, and its only
# job is to understand the request and route it to the correct
# specialist agent.
#
# This is a DESIGN PATTERN built on top of Handoffs (Concept 4).
# The triage agent orchestrates a team of specialists.
#
# REAL-WORLD ANALOGY:
#   Think of a hospital emergency room:
#   - The TRIAGE NURSE assesses every incoming patient
#   - Based on the issue, they send the patient to:
#       → Cardiology (heart)
#       → Orthopedics (bones)
#       → Neurology (brain)
#   The nurse does NOT treat the patient — they ROUTE.
#
# THE WORKFLOW:
#   User Message → Triage Agent → Identifies intent
#       ↓
#   Hands off to the right specialist:
#       - Career question?    → Career Advisor Agent
#       - Finance question?   → Finance Advisor Agent
#       - Tech question?      → Tech Support Agent
#
# WHY USE A TRIAGE PATTERN?
#   ✅ Clean separation of concerns
#   ✅ Each specialist agent stays focused
#   ✅ Easy to add more specialists without changing the triage logic
#   ✅ Scales to many domains
# =============================================================

import asyncio

import groq_setup  # noqa: F401
from groq_setup import MODEL

from agents import Agent, Runner


# ── Specialist 1: Career Advisor ──────────────────────────────
career_agent = Agent(
    name="Career Advisor",
    instructions=(
        "You are an expert career counselor. You help people with "
        "job searching, resume tips, interview preparation, skill development, "
        "and career transitions. Give practical, actionable advice."
    ),
    model=MODEL,
)

# ── Specialist 2: Finance Advisor ─────────────────────────────
finance_agent = Agent(
    name="Finance Advisor",
    instructions=(
        "You are a personal finance expert. You help people with "
        "budgeting, saving, investing, debt management, and financial planning. "
        "Be responsible and mention professional consultation when needed."
    ),
    model=MODEL,
)

# ── Specialist 3: Tech Support ────────────────────────────────
tech_agent = Agent(
    name="Tech Support",
    instructions=(
        "You are a friendly technical support specialist. You help with "
        "software issues, programming questions, computer troubleshooting, "
        "and technology recommendations. Use clear, non-technical language when possible."
    ),
    model=MODEL,
)

# ── Triage Agent (the ROUTER) ─────────────────────────────────
# This is the ENTRY POINT. It sees every user message first.
# Its ONLY job is to route — never to answer directly.
# Note: It has handoffs to ALL specialist agents.
triage_agent = Agent(
    name="Triage Agent",
    instructions=(
        "You are a smart routing assistant. Your ONLY job is to identify "
        "the category of the user's request and transfer it immediately "
        "to the correct specialist:\n\n"
        "  - Career-related  (jobs, resume, interviews, skills) → 'Career Advisor'\n"
        "  - Finance-related (money, savings, investing, budget) → 'Finance Advisor'\n"
        "  - Tech-related    (software, coding, computers, apps) → 'Tech Support'\n\n"
        "DO NOT answer any question yourself. "
        "ALWAYS hand off to the appropriate specialist."
    ),
    model=MODEL,
    handoffs=[career_agent, finance_agent, tech_agent],
)


# ── Helper to run one test ─────────────────────────────────────
async def test(question: str, expected_specialist: str):
    print(f"\n[Question]  : {question}")
    print(f"[Expected]  : Should go to → {expected_specialist}")

    result = await Runner.run(
        starting_agent=triage_agent,  # Always start with the triage agent
        input=question,
    )

    print(f"[Response from Specialist]\n{result.final_output}")
    print("-" * 60)


# ── Entry point ───────────────────────────────────────────────
async def main():
    print("\n" + "=" * 60)
    print("  CONCEPT 6: TRIAGE AGENT")
    print("=" * 60)
    print("Triage Agent = Smart router that delegates to specialists")
    print("One entry point → Many experts behind the scenes.\n")
    print("Testing with 3 different question types:\n")

    # Test 1: Should route to Career Advisor
    await test(
        "I have 5 years of experience in sales but want to move into project management. How do I start?",
        "Career Advisor"
    )

    # Test 2: Should route to Finance Advisor
    await test(
        "I earn 80,000 PKR per month. How much should I save and where should I invest?",
        "Finance Advisor"
    )

    # Test 3: Should route to Tech Support
    await test(
        "My Python script keeps crashing with a RecursionError. What does that mean?",
        "Tech Support"
    )

    print("\n" + "=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
