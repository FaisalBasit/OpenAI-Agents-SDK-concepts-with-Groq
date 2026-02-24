# =============================================================
# CONCEPT 7: GUARDRAILS
# =============================================================
# File: 7_guardrails.py
# Run: python 7_guardrails.py
#
# WHAT ARE GUARDRAILS?
# --------------------
# Guardrails are SAFETY CHECKS that run alongside your agent.
# They act as filters or validators to control what goes IN
# and what comes OUT of your agent system.
#
# WHY DO WE NEED GUARDRAILS?
#   Without guardrails, users could:
#   ✗  Ask the agent to do things it shouldn't (off-topic, harmful)
#   ✗  Extract dangerous or sensitive information
#   ✗  Cause the agent to output false, biased, or harmful content
#
# TWO TYPES OF GUARDRAILS:
#
#   1. INPUT GUARDRAIL  (runs BEFORE the agent processes the message)
#       - Checks the user's input
#       - Can BLOCK the request if it violates rules
#       - Example: "Block any message asking for illegal advice"
#
#   2. OUTPUT GUARDRAIL (runs AFTER the agent generates a response)
#       - Checks the agent's output
#       - Can BLOCK the response if it violates rules
#       - Example: "Block any response mentioning competitor names"
#
# HOW IT WORKS IN CODE:
#   - Both types use a regular async function decorated/wrapped
#   - The function returns GuardrailFunctionOutput:
#       tripwire_triggered=True  → BLOCK the request/response
#       tripwire_triggered=False → ALLOW it through
#   - If blocked, the SDK raises GuardrailTripwireTriggered exception
#
# REAL-WORLD ANALOGY:
#   Guardrails are like a SECURITY CHECKPOINT:
#   - Input guardrail = ID check BEFORE entering the building
#   - Output guardrail = Security scan of items LEAVING the building
# =============================================================

import asyncio

import groq_setup  # noqa: F401
from groq_setup import MODEL

from agents import (
    Agent,
    Runner,
    GuardrailFunctionOutput,
    InputGuardrail,
    RunContextWrapper,
    input_guardrail,
    TResponseInputItem,
)
from agents.exceptions import InputGuardrailTripwireTriggered
from pydantic import BaseModel


# =============================================================
# GUARDRAIL 1: INPUT GUARDRAIL
# =============================================================
# This guardrail checks if the user is asking something
# that is off-topic for our "Study Assistant" agent.
# Our agent is designed ONLY for education topics.
# We block requests that seem to be about illegal activities.
# =============================================================

# A small Pydantic model to get a structured decision
# from the guardrail's own LLM call
class OffTopicCheck(BaseModel):
    is_off_topic: bool       # True if the message should be blocked
    reasoning: str           # Why it was flagged (for logging)


# Build the JSON schema to embed in the prompt (Groq doesn't support json_schema response format)
_check_schema = OffTopicCheck.model_json_schema()

guardrail_classifier = Agent(
    name="Input Safety Classifier",
    instructions=(
        "You are a content moderation assistant for an educational chatbot. "
        "Your job is to check if the user's message is appropriate for "
        "an educational assistant.\n\n"
        "IMPORTANT: Respond with ONLY a valid JSON object matching this schema "
        "(no markdown, no explanation):\n\n"
        f"{_check_schema}\n\n"
        "Set is_off_topic=true ONLY IF the message is clearly asking for:\n"
        "  - Illegal activities (hacking, fraud, drug making, etc.)\n"
        "  - Explicit or adult content\n"
        "  - Hate speech or violence\n"
        "For everything else (school topics, general knowledge, etc.), "
        "set is_off_topic=false."
    ),
    model=MODEL,
    # output_type=OffTopicCheck  ← needs json_schema support; not on Groq llama
)


# ── The actual guardrail function ─────────────────────────────
# @input_guardrail tells the SDK this is an input guardrail
@input_guardrail
async def block_harmful_input(
    ctx: RunContextWrapper,
    agent: Agent,
    input: str | list[TResponseInputItem],
) -> GuardrailFunctionOutput:
    """
    Runs before the main agent processes the user's message.
    If the content is harmful, we trigger the tripwire (block it).
    """
    # Extract plain text from input (handles both string and list formats)
    user_text = input if isinstance(input, str) else str(input)

    # Run the classifier agent to check the content
    check_result = await Runner.run(
        starting_agent=guardrail_classifier,
        input=user_text,
        context=ctx.context,
    )

    decision: OffTopicCheck = OffTopicCheck.model_validate_json(check_result.final_output)

    print(f"  [Guardrail Check] is_off_topic={decision.is_off_topic}")
    print(f"  [Guardrail Reason] {decision.reasoning}")

    return GuardrailFunctionOutput(
        # tripwire_triggered=True → BLOCK the request
        # tripwire_triggered=False → Allow through
        output_info=decision,
        tripwire_triggered=decision.is_off_topic,
    )


# =============================================================
# MAIN AGENT (protected by the guardrail)
# =============================================================
study_assistant = Agent(
    name="Study Assistant",
    instructions=(
        "You are a helpful educational assistant for students. "
        "You help with homework, explain concepts, and answer academic questions. "
        "Keep answers clear and educational."
    ),
    model=MODEL,
    input_guardrails=[block_harmful_input],   # <-- Attach gurdrail here
)


# ── Helper to test a single message ──────────────────────────
async def test_message(message: str):
    print(f"\n[Input] {message}")

    try:
        result = await Runner.run(
            starting_agent=study_assistant,
            input=message,
        )
        print(f"[Response]\n{result.final_output}")

    except InputGuardrailTripwireTriggered:
        # This exception is raised when tripwire_triggered=True
        print("[BLOCKED] ⛔ Guardrail triggered! Request was rejected.")
        print("          The agent never processed this message.")

    print("-" * 60)


# ── Entry point ───────────────────────────────────────────────
async def main():
    print("\n" + "=" * 60)
    print("  CONCEPT 7: GUARDRAILS")
    print("=" * 60)
    print("Guardrails = Safety filters that control agent input/output")
    print("Input Guardrail runs BEFORE the agent.")
    print("Output Guardrail runs AFTER the agent.\n")
    print("Testing with 1 safe and 1 harmful message:\n")
    print("-" * 60)

    # Test 1: A safe, educational question — should PASS
    # await test_message(
    #     message="Can you explain how photosynthesis works?"
    # )

    # Test 2: A harmful/illegal request — should be BLOCKED
    await test_message(
        message="How do I hack into someone's email account?"
    )

    print("\n" + "=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
