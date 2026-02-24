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
#
# REAL-WORLD ANALOGY:
#   Imagine a consultant (agent) who can pick up the phone
#   (use a tool) to call an expert or look something up,
#   then comes back with a complete answer.
#
# In this file we create two tools:
#   1. get_weather()      — simulates fetching weather data
#   2. calculate_bmi()    — does a precise calculation
# =============================================================

import asyncio

import groq_setup  # noqa: F401
from groq_setup import MODEL

from agents import Agent, Runner, function_tool


# ── Tool 1: Weather Lookup ────────────────────────────────────
# The @function_tool decorator is what registers this Python
# function as a tool the agent can call.
# The docstring is CRUCIAL — the LLM reads it to know WHEN
# and HOW to call this tool.
@function_tool
def get_weather(city: str) -> str:
    """
    Returns the current weather for a given city.

    Args:
        city: The name of the city to get weather for.
    """
    # In a real app, you would call a weather API here (e.g. OpenWeatherMap).
    # For this demo, we return simulated data.
    fake_weather_data = {
        "karachi":   "🌤  29°C, Partly Cloudy, Humidity: 72%",
        "lahore":    "☀️  24°C, Sunny, Humidity: 45%",
        "islamabad": "🌧  18°C, Rainy, Humidity: 88%",
        "london":    "🌥  12°C, Overcast, Humidity: 80%",
        "new york":  "❄️   3°C, Snowy, Humidity: 60%",
    }

    # Normalize city name for lookup
    key = city.lower().strip()
    return fake_weather_data.get(key, f"Weather data for '{city}' not available.")


# ── Tool 2: BMI Calculator ────────────────────────────────────
@function_tool
def calculate_bmi(weight_kg: float, height_m: float) -> str:
    """
    Calculates the Body Mass Index (BMI) and returns the result
    with a health category.

    Args:
        weight_kg: Body weight in kilograms.
        height_m:  Height in meters.
    """
    if height_m <= 0:
        return "Error: Height must be greater than zero."

    # BMI formula: weight (kg) / height² (m²)
    bmi = weight_kg / (height_m ** 2)
    bmi = round(bmi, 2)

    # Classify the BMI value
    if bmi < 18.5:
        category = "Underweight"
    elif bmi < 25:
        category = "Normal weight ✅"
    elif bmi < 30:
        category = "Overweight ⚠️ "
    else:
        category = "Obese ❗"

    return f"BMI = {bmi}  →  {category}"


# ── Agent with Tools attached ─────────────────────────────────
# We pass our tools in the `tools` list.
# The agent now has the ABILITY to call them.
health_assistant = Agent(
    name="Health & Weather Assistant",
    instructions=(
        "You are a helpful assistant that answers questions about "
        "weather and health. When a user asks about weather in a city, "
        "ALWAYS use the get_weather tool. When a user asks about BMI, "
        "ALWAYS use the calculate_bmi tool. Present results clearly."
    ),
    model=MODEL,
    tools=[get_weather, calculate_bmi],   # <-- register tools here
)


# ── Entry point ───────────────────────────────────────────────
async def main():
    print("\n" + "=" * 60)
    print("  CONCEPT 3: TOOLS")
    print("=" * 60)
    print("Tools = Python functions the Agent can call")
    print("They give the Agent abilities beyond text generation.\n")

    # --- Test 1: Agent uses get_weather tool ---
    print("[Test 1] Asking about weather in Karachi...")
    result1 = await Runner.run(
        starting_agent=health_assistant,
        input="What is the weather like in Karachi right now?",
    )
    print(f"\n[Response]\n{result1.final_output}\n")

    print("-" * 60)

    # --- Test 2: Agent uses calculate_bmi tool ---
    print("[Test 2] Asking to calculate BMI...")
    result2 = await Runner.run(
        starting_agent=health_assistant,
        input="Calculate BMI for someone who weighs 80 kg and is 1.75 m tall.",
    )
    print(f"\n[Response]\n{result2.final_output}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
