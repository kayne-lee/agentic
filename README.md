# LangChain vs LangGraph (agentic examples)

This repo contains two small agentic systems that solve the same task (support ticket triage + reply) in different ways.

- `langchain_agent/app.py`: a single LangChain agent that decides which tools to call.
- `langgraph_agent/app.py`: a LangGraph workflow with explicit steps, state, and routing.

## Quick start

1) Create a virtual environment and install deps:

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

2) Set your model key (example for OpenAI):

```bash
export OPENAI_API_KEY=your_key_here
```

Or create a `.env` file in the repo root:

```bash
OPENAI_API_KEY=your_key_here
```

3) Run either example:

```bash
python langchain_agent/app.py "My order is 12 days late and I want a refund."
python langgraph_agent/app.py "I see a duplicate charge and may file a chargeback."
```

## What each example does

### LangChain agent

- Uses tools (`lookup_policy`, `calculate_refund`) when the LLM decides they are needed.
- Output is one final response based on agent reasoning.
- Great for fast prototyping or open-ended tasks.

### LangGraph workflow

- Turns the same task into explicit steps: classify -> fetch policy -> handoff check -> draft -> review.
- Uses conditional edges to retry or escalate.
- Great for reliability, traceability, and guardrails.

## Key differences at a glance

- Control flow: LangChain is implicit; LangGraph is explicit.
- Debuggability: LangChain gives a final output; LangGraph shows state at each node.
- Reliability: LangGraph can enforce retries, limits, and handoffs.
- Scaling: LangGraph is easier to extend into multi-agent or multi-step systems.
- When to choose: use LangChain for fast agent prototypes, LangGraph for production workflows.

## Notes

- Both scripts use `gpt-4o-mini` by default; swap the model in the code if needed.
- The policy data is intentionally tiny and local so you can see the flow clearly.
