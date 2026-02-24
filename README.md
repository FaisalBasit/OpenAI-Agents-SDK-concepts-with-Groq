# OpenAI Agents SDK — Concepts with Groq

A hands-on series of concept files demonstrating the **OpenAI Agents SDK** running on **Groq's free API** (no OpenAI account needed).

Each file is self-contained and teaches one concept, from a basic agent all the way to safety guardrails.

---

## 🚀 Setup

### 1. Clone the repository
```bash
git clone https://github.com/FaisalBasit/OpenAI-Agents-SDK-concepts-with-Groq.git
cd OpenAI-Agents-SDK-concepts-with-Groq
```

### 2. Create a Python virtual environment
```bash
python -m venv venv
```

### 3. Activate the virtual environment

**Windows:**
```bash
venv\Scripts\activate
```

**macOS / Linux:**
```bash
source venv/bin/activate
```

### 4. Install dependencies
```bash
pip install -r requirements.txt
```

### 5. Get a free Groq API key
1. Go to [https://console.groq.com](https://console.groq.com) and sign up for free
2. Navigate to **API Keys** and create a new key

### 6. Configure your API key
Create a `.env` file in the project root:
```
GROQ_API_KEY=your_api_key_here
```

---

## 📂 Concept Files

Run each file individually with `python <filename>`:

| File | Concept | What it demonstrates |
|------|---------|----------------------|
| `1_agent.py` | **Agent** | Create a basic agent and get a response |
| `2_tools.py` | **Tools** | Give agents custom Python functions to call |
| `3_handoffs.py` | **Handoffs** | Pass control between agents (language router) |
| `4_structured_output.py` | **Structured Output** | Extract typed data (Pydantic models) from responses |
| `5_triage_agent.py` | **Triage Agent** | Route requests to specialist agents automatically |
| `6_guardrails.py` | **Guardrails** | Block harmful or off-topic inputs before they reach the agent |

### Example
```bash
python 1_agent.py
python 2_tools.py
# ... and so on
```

---

## 🔧 How It Works

All files share `groq_setup.py` — a helper that:
- Loads your `GROQ_API_KEY` from `.env`
- Points the OpenAI Agents SDK at Groq's OpenAI-compatible endpoint
- Sets the default model to `llama-3.3-70b-versatile`

You can change the model for all files at once by editing one line in `groq_setup.py`:
```python
MODEL = "llama-3.3-70b-versatile"   # change this
```

---

## ⚠️ Groq Compatibility Notes

The OpenAI Agents SDK is designed for OpenAI's API. A few patches are applied for Groq compatibility:

- **Handoff schema patch** (in `3_handoffs.py`, `5_triage_agent.py`): Groq rejects the SDK's default empty handoff tool schema, so it's overridden to a compatible format.
- **Structured output** (in `4_structured_output.py`, `6_guardrails.py`): Groq's llama models don't support `json_schema` response format, so the Pydantic schema is embedded in the prompt and parsed manually.

---

## 📋 Requirements

- Python 3.10+
- A free [Groq API key](https://console.groq.com)

Dependencies (installed via `requirements.txt`):
```
openai-agents
openai
pydantic
python-dotenv
```
