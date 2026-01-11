import argparse
from typing import Dict

from dotenv import load_dotenv
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.tools import tool
from langchain_openai import ChatOpenAI

load_dotenv()

POLICIES: Dict[str, str] = {
    "refund": (
        "Refunds: Full refunds are allowed within 30 days of purchase if the item "
        "is unused. Partial refunds (up to 50%) are allowed for used items within "
        "30 days. No refunds after 30 days."
    ),
    "shipping": (
        "Shipping: Standard shipping is 5-7 business days. Expedited shipping is "
        "2-3 business days. Delays over 10 business days qualify for a 10% credit."
    ),
    "billing": (
        "Billing: Duplicate charges can be reversed within 5 business days. "
        "Chargebacks are escalated to a human agent immediately."
    ),
    "technical": (
        "Technical: Troubleshoot by confirming account access, resetting password, "
        "and clearing cache. Escalate if the user is locked out for more than 24 hours."
    ),
}


def _match_policy(topic: str) -> str:
    topic_lower = topic.lower()
    for key, value in POLICIES.items():
        if key in topic_lower:
            return value
    return "No matching policy found. Use best judgment and keep reply concise."


@tool
def lookup_policy(topic: str) -> str:
    """Return the relevant support policy for a topic."""

    return _match_policy(topic)


@tool
def calculate_refund(amount: float, percent: float) -> str:
    """Calculate a refund amount based on a percent."""

    refund = amount * (percent / 100.0)
    return f"${refund:.2f}"


def build_agent() -> AgentExecutor:
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    tools = [lookup_policy, calculate_refund]
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a support agent. Use tools when they help. "
                "Return a JSON object with keys: category, needs_human, reply. "
                "Keep reply under 120 words.",
            ),
            ("human", "{input}"),
            MessagesPlaceholder("agent_scratchpad"),
        ]
    )
    agent = create_openai_tools_agent(llm, tools, prompt)
    return AgentExecutor(agent=agent, tools=tools, verbose=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="LangChain support agent example")
    parser.add_argument(
        "message",
        nargs="?",
        default=(
            "My order is 12 days late and I want a refund for expedited shipping. "
            "I paid $200."
        ),
        help="Customer message to triage",
    )
    args = parser.parse_args()

    agent = build_agent()
    result = agent.invoke({"input": args.message})
    print(result["output"])


if __name__ == "__main__":
    main()
