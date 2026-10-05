import os
import json
import time
from dotenv import load_dotenv

from crewai import Agent, Crew, Process, Task
from crewai.tools import tool
from ddgs import DDGS

MODEL = "gpt-4o-mini"
TOPIC = "India's DPDP Act obligations for AI systems handling customer data"

WEAK_CRITIC = False
BREAK_CONTEXT = True

@tool("Web Search")
def ddg_search(query: str) -> str:
    """Search the web with DuckDuckGo and return the top results as text."""
    with DDGS() as ddgs:
        hits = list(ddgs.text(query, max_results=5))
    if not hits:
        return "No results found."
    return "\n".join(
        f"- {h.get('title', '')}: {h.get('body', '')[:200]} ({h.get('href', '')})"
        for h in hits
    )

critic_goal = (
    "Check the brief's tone is professional"
    if WEAK_CRITIC
    else " Flag any claim in the brief that is not supported by the findings "
)

researcher = Agent(
        role="Research Analyst",
        goal="Find and cite source material on {topic}",
        backstory=(
            "You verify every claim against a source before you report it."
            "You never present an unsourced statement as fact"
        ),
        tools=[ddg_search],
        llm=MODEL, max_iter=5, verbose=True  
)

summarizer = Agent(
    role="Briefing Writer",
    goal="Turn the researcher's findings into a 200-word brief",
    backstory=(
        "You compress findings into clear prose and never add a fact "
        "that is not in the findings you were handed."
    ),
    tools=[],
    llm=MODEL, max_iter=5, verbose=True,
)

critic = Agent(
    role="Quality Critic",
    goal=critic_goal,
    backstory=(
        "You are hard to satisfy. You treat a claim as unsupported "
        "until you have seen it in findings."
    ),
    tools=[],
    llm=MODEL, max_iter=5, verbose=True,
)

find = Task(
    description="Research the topic: {topic}. Gather concrete finddings and list a supporting source URL for each",
    expected_output="A list of findings, each with its supporting source URL ",
    agent=researcher
)

write = Task(
    description="Write a 200-word brief on {topic} using only the researcher's findings. Do not introduce facts that are not in the findings.",
    expected_output="A brief of about 200 words with no invented facts.",
    agent=summarizer,
    context=[] if BREAK_CONTEXT else [find],
)

check = Task(
    description="Compare the brief against the findings. If every "
                "claim is supported, reply with exactly APPROVED. "
                "Otherwise reply with a numbered list of required revisions.",
    expected_output="Either 'APPROVED' or a numbered revision list.",
    agent=critic,
    context=[find, write],
)

def main():
    crew = Crew(
        agents=[researcher, summarizer, critic],
        tasks=[find, write, check],
        process = Process.sequential,
        verbose=True
    )

    start = time.perf_counter()
    result = crew.kickoff(inputs={"topic" : TOPIC})
    wall = time.perf_counter() - start
    print(result)
    print(f"\nTime taken: {wall:.1f}s")


if __name__ == "__main__":
    main()