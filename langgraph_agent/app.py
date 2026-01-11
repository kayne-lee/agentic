import argparse
from typing import Dict, TypedDict

from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph

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
    "other": "General: Be polite, ask clarifying questions, and offer next steps.",
}

ALLOWED_CATEGORIES = {"refund", "shipping", "billing", "technical", "other"}


class TicketState(TypedDict, total=False):
    message: str
    category: str
    policy: str
    needs_human: bool
    draft: str
    review_passed: bool
    attempts: int


llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


def classify_issue(state: TicketState) -> TicketState:
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Classify the customer message into one of: refund, shipping, billing, "
                "technical, other. Reply with exactly one word.",
            ),
            ("human", "{message}"),
        ]
    )
    response = llm.invoke(prompt.format_messages(message=state["message"]))
    category = response.content.strip().lower()
    if category not in ALLOWED_CATEGORIES:
        category = "other"
    return {"category": category}


def fetch_policy(state: TicketState) -> TicketState:
    policy = POLICIES.get(state["category"], POLICIES["other"])
    return {"policy": policy}


def decide_handoff(state: TicketState) -> TicketState:
    message = state["message"].lower()
    needs_human = "chargeback" in message or "legal" in message or "lawsuit" in message
    return {"needs_human": needs_human}


def draft_reply(state: TicketState) -> TicketState:
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Draft a short support reply that follows the policy, answers the "
                "customer, and offers next steps. Keep it under 120 words.",
            ),
            (
                "human",
                "Message: {message}\nCategory: {category}\nPolicy: {policy}",
            ),
        ]
    )
    response = llm.invoke(
        prompt.format_messages(
            message=state["message"],
            category=state["category"],
            policy=state["policy"],
        )
    )
    attempts = state.get("attempts", 0) + 1
    return {"draft": response.content.strip(), "attempts": attempts}


def review_reply(state: TicketState) -> TicketState:
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Review the draft reply. Reply with PASS if it mentions the policy "
                "and includes a next step. Otherwise reply with FAIL.",
            ),
            (
                "human",
                "Policy: {policy}\nDraft: {draft}",
            ),
        ]
    )
    response = llm.invoke(
        prompt.format_messages(policy=state["policy"], draft=state["draft"])
    )
    decision = response.content.strip().lower()
    review_passed = decision.startswith("pass")
    return {"review_passed": review_passed}


def human_handoff(state: TicketState) -> TicketState:
    message = (
        "This request needs a human agent. A specialist will review your case and "
        "follow up shortly."
    )
    return {"draft": message}


def route_from_handoff(state: TicketState) -> str:
    return "handoff" if state.get("needs_human") else "draft"


def route_after_review(state: TicketState) -> str:
    if state.get("review_passed"):
        return "done"
    if state.get("attempts", 0) >= 2:
        return "handoff"
    return "retry"


def build_graph():
    graph = StateGraph(TicketState)
    graph.add_node("classify", classify_issue)
    graph.add_node("policy", fetch_policy)
    graph.add_node("handoff_check", decide_handoff)
    graph.add_node("draft", draft_reply)
    graph.add_node("review", review_reply)
    graph.add_node("handoff", human_handoff)

    graph.set_entry_point("classify")
    graph.add_edge("classify", "policy")
    graph.add_edge("policy", "handoff_check")
    graph.add_conditional_edges(
        "handoff_check",
        route_from_handoff,
        {"handoff": "handoff", "draft": "draft"},
    )
    graph.add_edge("draft", "review")
    graph.add_conditional_edges(
        "review",
        route_after_review,
        {"retry": "draft", "handoff": "handoff", "done": END},
    )
    graph.add_edge("handoff", END)

    return graph.compile()


def main() -> None:
    parser = argparse.ArgumentParser(description="LangGraph support workflow example")
    parser.add_argument(
        "message",
        nargs="?",
        default=(
            "I see a duplicate charge and I might file a chargeback. "
            "Please fix this."
        ),
        help="Customer message to triage",
    )
    args = parser.parse_args()

    graph = build_graph()
    result = graph.invoke({"message": args.message, "attempts": 0})

    print(f"category: {result.get('category')}")
    print(f"needs_human: {result.get('needs_human')}")
    print(result.get("draft", ""))


if __name__ == "__main__":
    main()
