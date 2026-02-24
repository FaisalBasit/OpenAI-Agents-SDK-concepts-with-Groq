# =============================================================
# CONCEPT 5: STRUCTURED OUTPUT
# =============================================================
# File: 5_structured_output.py
# Run: python 5_structured_output.py
#
# WHAT IS STRUCTURED OUTPUT?
# ---------------------------
# By default, agents return free-form TEXT — like a paragraph.
# That's great for humans reading it, but hard for code to use.
#
# STRUCTURED OUTPUT forces the agent to return data in a
# SPECIFIC FORMAT — like a Python object or JSON — so your
# code can access individual fields directly.
#
# HOW IT WORKS:
#   1. Define a Pydantic model (a class that describes the shape
#      of the data you want)
#   2. Pass it to Agent as output_type=YourModel
#   3. The SDK guarantees the response matches that exact shape
#   4. result.final_output is now a Python object, not a string!
#
# PYDANTIC:
#   Pydantic is a Python library for data validation.
#   You define a class that inherits from BaseModel,
#   and each field has a type annotation.
#
# REAL-WORLD USE CASES:
#   ✅ Extract product info from a review
#   ✅ Parse a resume into structured fields
#   ✅ Analyze sentiment with a confidence score
#   ✅ Get a structured report from raw data
#
# In this file we extract structured "Movie Info" from
# a user's description of a film.
# =============================================================

import asyncio

import groq_setup  # noqa: F401
from groq_setup import MODEL, JSON_TOOL_INSTRUCTIONS

from agents import Agent, Runner
from pydantic import BaseModel, Field
from typing import List


# ── Step 1: Define your data shape using Pydantic ─────────────
# Each field must be described clearly — the LLM reads these
# descriptions to know what to put in each field.
class MovieInfo(BaseModel):
    """Structured information extracted about a movie."""

    title: str = Field(description="The full title of the movie")

    director: str = Field(description="The director's name")

    year: int = Field(description="The year the movie was released")

    genre: List[str] = Field(description="List of genre tags (e.g. ['Drama', 'Thriller'])")

    main_actors: List[str] = Field(
        description="List of the main actor names (up to 3)"
    )

    plot_summary: str = Field(
        description="A 1-2 sentence summary of the plot"
    )

    rating_out_of_10: float = Field(
        description="An estimated rating based on reputation (e.g. 8.5)"
    )


# ── Step 2: Create an Agent with JSON-in-prompt approach ────────

# Build the JSON schema string to embed in the prompt
_schema_str = MovieInfo.model_json_schema()

movie_extractor = Agent(
    name="Movie Info Extractor",
    instructions=(
        "You are a movie database assistant. Extract movie information into a JSON object.\n\n"
        "Respond with ONLY a JSON object like this example:\n"
        '{\n  "title": "Movie Name",\n  "director": "Director Name",\n  "year": 2024,\n  "genre": ["Action", "Drama"],\n  "main_actors": ["Actor 1", "Actor 2"],\n  "plot_summary": "Short summary.",\n  "rating_out_of_10": 8.5\n}\n\n'
        "Use this schema as reference:\n"
        f"{_schema_str}\n"
        + JSON_TOOL_INSTRUCTIONS
    ),
    model=MODEL,
    # output_type=MovieInfo  ← would need json_schema support; not on Groq llama
)


# ── Entry point ───────────────────────────────────────────────
async def main():
    print("  CONCEPT 5: STRUCTURED OUTPUT")
    print("Structured Output = Agent returns a typed Python object")
    print("instead of unstructured text. Powered by Pydantic.\n")

    # Give the agent a casual description of a movie
    user_input = (
        "Tell me about Inception, the Christopher Nolan sci-fi thriller "
        "from 2010 with Leonardo DiCaprio involving dreams within dreams."
    )

    print(f"[User Input]\n{user_input}\n")
    print("[Agent] → Extracting structured data...\n")

    result = await Runner.run(
        starting_agent=movie_extractor,
        input=user_input,
    )

    # result.final_output is a plain string (JSON text from the model).
    # We validate+parse it into a MovieInfo Pydantic object ourselves.
    raw_output: str = result.final_output.strip()

    # Robust JSON extraction: Find the first '{' and last '}'
    try:
        start_idx = raw_output.find("{")
        end_idx = raw_output.rfind("}") + 1
        if start_idx != -1 and end_idx != -1:
            json_str = raw_output[start_idx:end_idx]
        else:
            json_str = raw_output
        
        movie: MovieInfo = MovieInfo.model_validate_json(json_str)
    except Exception as e:
        print(f"  [Error Parsing JSON] {e}\n  Raw Output: {raw_output}")
        # Create a dummy object or raise
        raise e

    # We can access each field individually like a Python object
    print("[Extracted Structured Data]")
    print(f"  Title          : {movie.title}")
    print(f"  Director       : {movie.director}")
    print(f"  Year           : {movie.year}")
    print(f"  Genre          : {', '.join(movie.genre)}")
    print(f"  Main Actors    : {', '.join(movie.main_actors)}")
    print(f"  Rating         : {movie.rating_out_of_10}/10")
    print(f"  Plot Summary   : {movie.plot_summary}")

    # Also show the raw JSON representation
    print("\n[Raw JSON Output]")
    print(movie.model_dump_json(indent=2))

    print("\n" + "=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
