# =============================================================
# CONCEPT 2: RUNNER
# =============================================================
# File: 2_runner.py
# Run: python 2_runner.py
#
# WHAT IS A RUNNER?
# -----------------
# A Runner is the ENGINE that powers an Agent.
# While an Agent defines WHO does the work (personality & model),
# the Runner defines HOW the work gets executed.
#
# The Runner manages what is called the "AGENT LOOP":
#
#   User Input
#       ↓
#   Runner sends input to Agent (LLM call)
#       ↓
#   Agent may call TOOLS (functions) → Results returned
#       ↓
#   Agent may HAND OFF to another agent
#       ↓
#   Agent produces final text response
#       ↓
#   Runner returns the final result
#
# RUNNER MODES:
#   Runner.run()        →  Async (non-blocking, best for web apps)
#   Runner.run_sync()   →  Synchronous (blocking, simpler scripts)
#   Runner.run_streamed()→  Stream token by token (like ChatGPT typing)
#
# In this file we demonstrate all three runner modes so you
# can see the differences between them.
# =============================================================

import asyncio

import groq_setup  # noqa: F401
from groq_setup import MODEL

from agents import Agent, Runner

# ── Single shared agent we'll run in different ways ───────────
joke_agent = Agent(
    name="Joke Teller",
    instructions=(
        "You are a comedian. When asked for a joke, tell ONE short, "
        "clean joke and then briefly explain why it is funny."
    ),
    model=MODEL,
)


# ── Mode 1: Async Runner (most common) ────────────────────────
async def demo_async_runner():
    """
    Runner.run() is a coroutine (async function).
    It's the standard way to run an agent — non-blocking,
    meaning other tasks can proceed while waiting for the LLM.
    """
    print("\n[Mode 1] ASYNC Runner (Runner.run)")
    print("-" * 40)

    result = await Runner.run(
        starting_agent=joke_agent,
        input="Tell me a programming joke.",
    )
    print(result.final_output)


# ── Mode 2: Synchronous Runner (simple scripts) ───────────────
def demo_sync_runner():
    """
    Runner.run_sync() blocks the program until the agent finishes.
    Great for simple terminal scripts where you don't need concurrency.
    """
    print("\n[Mode 2] SYNC Runner (Runner.run_sync)")
    print("-" * 40)

    result = Runner.run_sync(
        starting_agent=joke_agent,
        input="Tell me a math joke.",
    )
    print(result.final_output)


# ── Mode 3: Streamed Runner (token by token) ──────────────────
async def demo_streamed_runner():
    """
    Runner.run_streamed() returns tokens as they are generated,
    just like ChatGPT's typing effect. Each chunk is a piece
    of the full response arriving in real time.
    """
    print("\n[Mode 3] STREAMED Runner (Runner.run_streamed)")
    print("-" * 40)
    print("[Streaming response]: ", end="", flush=True)

    # run_streamed returns an object we can iterate over
    async with Runner.run_streamed(
        starting_agent=joke_agent,
        input="Tell me a science joke.",
    ) as stream:
        async for chunk in stream.stream_events():
            # Each chunk event may carry a text delta (partial text)
            if (
                chunk.type == "raw_response_event"
                and hasattr(chunk.data, "delta")
                and hasattr(chunk.data.delta, "content")
            ):
                for block in chunk.data.delta.content or []:
                    if hasattr(block, "text"):
                        print(block.text, end="", flush=True)

    print()  # New line after streaming completes


# ── Entry point ───────────────────────────────────────────────
async def main():
    print("\n" + "=" * 60)
    print("  CONCEPT 2: RUNNER")
    print("=" * 60)
    print("Runner = The engine that executes the Agent loop\n")
    print("Three modes: async | sync | streamed\n")

    await demo_async_runner()
    demo_sync_runner()
    await demo_streamed_runner()

    print("\n" + "=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
