# =============================================================
# CONCEPT 3: TOOLS
# =============================================================
# File: 3_tools.py
# Run: python 3_tools.py
#
# WHAT ARE TOOLS?
# ---------------
# By default, an LLM can ONLY generate text — it cannot:
#   ✗  Search the web
#   ✗  Do math precisely
#   ✗  Read files
#   ✗  Call external APIs
#   ✗  Get real-time data
#
# TOOLS bridge this gap!
# A Tool is a regular Python function that the agent can call
# when it needs to perform an action beyond generating text.
#
# HOW IT WORKS:
#   1. You decorate a Python function with @function_tool
#   2. You attach it to an Agent via tools=[...]
#   3. During a run, the LLM decides WHEN to call the tool
#   4. The SDK executes the function and returns the result to the LLM
#   5. The LLM uses the result to form its final answer
# =============================================================

import asyncio
import groq_setup  # noqa: F401
from groq_setup import MODEL, JSON_TOOL_INSTRUCTIONS
from agents import Agent, Runner, function_tool


# ── Tool: Weather Lookup ────────────────────────────────────
@function_tool
def get_weather(city: str) -> str:
    """
    Returns the current weather for a given city.

    Args:
        city: The name of the city to get weather for.
    """
    fake_weather_data = {
        "karachi":   "🌤  29°C, Partly Cloudy, Humidity: 72%",
        "lahore":    "☀️  24°C, Sunny, Humidity: 45%",
        "islamabad": "🌧  18°C, Rainy, Humidity: 88%",
        "london":    "🌥  12°C, Overcast, Humidity: 80%",
        "new york":  "❄️   3°C, Snowy, Humidity: 60%",
    }

    key = city.lower().strip()
    return fake_weather_data.get(key, f"Weather data for '{city}' not available.")


# ── Agent with Weather Tool ─────────────────────────────────
weather_assistant = Agent(
    name="Weather Assistant",
    instructions=(
        "You are a weather specialist. use weather tool for weather related questions "
        "DIAGNOSTIC: You possess a REAL-TIME tool named `get_weather`. "
        "COMMAND: You MUST call `get_weather` for ANY city requested. "
        "DO NOT apologize. DO NOT say you lack real-time data. "
        "DO NOT explain your limitations. Just call the tool and report the result."
        + JSON_TOOL_INSTRUCTIONS
    ),
    model=MODEL,
    tools=[get_weather],   # <-- only weather tool
)


# ── Entry point ─────────────────────────────────────────────
async def main():
    print("  CONCEPT 3: TOOLS")

    # --- Test: Agent uses get_weather tool ---
    print("[Test] Asking about weather in Karachi...")
    result = await Runner.run(
        starting_agent=weather_assistant,
        input="What is the weather like in Karachi right now?",
    )
    print(f"\n[Response]\n{result.final_output}\n")


if __name__ == "__main__":
    asyncio.run(main())