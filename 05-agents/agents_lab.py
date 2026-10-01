import json
import os
import sys
import asyncio
from pathlib import Path

from dotenv import load_dotenv
from agents import Agent, Runner, SQLiteSession, function_tool

load_dotenv()

open_api_key = os.getenv("OPENAI_API_KEY")

MODEL = "gpt-4o-mini"

TICKETS_PATH = Path(__file__).parent / "data" / "support_tickets.json"
with open(TICKETS_PATH, "r", encoding="utf-8") as f:
    TICKETS = {t["id"]: t for t in json.load(f) }

simple_agent = Agent(
    name="Support Desk Assistant (simple)",
    instructions="""
Context:
You are a support-desk assistant who answers general questions about how
a helpdesk works.

Instructions:
Answer briefly and helpfully. If asked about a specific ticket ID, say you
don't have a way to look that up.
""",
    model=MODEL,
)

memory_agent = Agent(
    name="Support Desk Assistant (with memory)",
    instructions=simple_agent.instructions,
    model=MODEL
)
memory_session = SQLiteSession("support-desk-converstion")


async def run_part1():
    print("Simple agent -> No Memory, No tool")
    question = "What priority levels does this helpdesk use, and what does ticket T002 say?"
    response = await Runner.run(starting_agent=simple_agent, input=question)

    print(f"\n Agent:\n {response.final_output}")

async def run_part2():
    print("PART 2: AGENT WITH MEMORY (+ session)")

    q1 = "I'm looking into ticket T005. What category is it?"
    print(f"You: {q1}")
    resp1 = await Runner.run(
        starting_agent=memory_agent,
        input=q1,
        session=memory_session,
    )
    print(f"\nAgent:\n{resp1.final_output}")

    q2 = "And what priority is it?"
    print(f"\nYou: {q2}")
    resp2 = await Runner.run(
        starting_agent=memory_agent,
        input=q2,
        session=memory_session,
    )
    print(f"\nAgent:\n{resp2.final_output}")
    print(
        "\n(Note: the agent still can't actually look up T005 — it's guessing. "
        "'That' and 'it' resolve correctly because of the session, but the "
        "answer isn't grounded in real data yet. Part 3 fixes that.)"
    )


@function_tool
def lookup_ticket(ticket_id: str) -> str:
    """Look up a support ticket by its ID and return its details.

    Args:
        ticket_id: The ticket ID to look up, e.g. "T005".
    """
    ticket = TICKETS.get(ticket_id.strip().upper())
    if not ticket:
        return f"No ticket found with ID {ticket_id!r}."
    return (
        f"{ticket['id']} | {ticket['subject']} | "
        f"priority={ticket['priority']} | category={ticket['category']}\n"
        f"{ticket['text']}"
    )

tool_agent = Agent(
    name="Support Desk Assistant (with memory + tools)",
    instructions=simple_agent.instructions
    + "\nWhen asked about a specific ticket ID, use the lookup_ticket tool "
      "instead of guessing.",
    model=MODEL,
    tools=[lookup_ticket],
)
tool_session = SQLiteSession("support-desk-conversation-with-tools")

async def run_part3() -> None:
    print("PART 3: AGENT WITH TOOLS (+ memory)")

    q1 = "I'm looking into ticket T005. What category is it?"
    print(f"You: {q1}")
    resp1 = await Runner.run(
        starting_agent=tool_agent,
        input=q1,
        session=tool_session,
    )
    print(f"\nAgent:\n{resp1.final_output}")

    q2 = "And what priority is it?"
    print(f"\nYou: {q2}")
    resp2 = await Runner.run(
        starting_agent=tool_agent,
        input=q2,
        session=tool_session,
    )
    print(f"\nAgent:\n{resp2.final_output}")
    print(
        "\n(Compare this to Part 2: the answer is now grounded in the real "
        "ticket data from data/support_tickets.json, and the session still "
        "lets the second question refer back to 'it'.)"
    )


async def main():
    #await run_part1()
    #await run_part2()
    await run_part3()

if __name__ == "__main__":
    asyncio.run(main())