# =============================================================
# groq_setup.py  -  Shared Groq Configuration
# =============================================================
# This file is NOT a concept file.
# It is a HELPER that every concept file imports so they
# don't repeat the same boilerplate setup code.
#
# WHY GROQ?
#   Groq is a cloud AI provider that exposes an API that is
#   100% compatible with the OpenAI API format. That means
#   the OpenAI Agents SDK works with Groq out of the box —
#   we just change the base_url and api_key.
#
# HOW IT WORKS:
#   1. We create an AsyncOpenAI client pointing at Groq's URL.
#   2. We register that client as the SDK's default client.
#   3. We tell the SDK to use "chat_completions" mode because
#      Groq does NOT support the newer "responses" API yet.
#   4. Every agent file imports `MODEL` from here so the model
#      name is set in one place and easy to change.
# =============================================================

import os
from dotenv import load_dotenv
from openai import AsyncOpenAI
from agents import set_default_openai_client, set_default_openai_api, set_tracing_disabled

# ── Load environment variables from the .env file ────────────
load_dotenv()

# ── Read the Groq API key from the environment ────────────────
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise EnvironmentError(
        "\n[ERROR] GROQ_API_KEY not found!\n"
        "Please open the .env file and paste your Groq API key.\n"
        "Get one free at: https://console.groq.com\n"
    )

# ── Groq's OpenAI-compatible endpoint ────────────────────────
GROQ_BASE_URL = "https://api.groq.com/openai/v1"

# ── Model to use (change here to switch for ALL files) ───────
# Popular Groq models:
#   "llama-3.3-70b-versatile"   <- Powerful, good for most tasks
#   "llama3-8b-8192"            <- Faster, lighter
#   "mixtral-8x7b-32768"        <- Great at reasoning
MODEL = "llama-3.3-70b-versatile"

# ── Create the Groq client ────────────────────────────────────
groq_client = AsyncOpenAI(
    api_key=GROQ_API_KEY,
    base_url=GROQ_BASE_URL,
)

# ── Register Groq client as the SDK default ───────────────────
# Now every Agent we create will automatically use Groq
# instead of OpenAI without any extra configuration.
set_default_openai_client(groq_client)

# ── Tell the SDK to use the chat completions API ─────────────
# Groq supports the standard /chat/completions endpoint.
# The newer OpenAI "responses" API is NOT supported by Groq.
set_default_openai_api("chat_completions")

# ── Disable tracing (not supported outside OpenAI platform) ──
set_tracing_disabled(True)

print(f"[groq_setup] ✅ Groq client configured | Model: {MODEL}")
