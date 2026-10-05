import time
from dotenv import load_dotenv
from pydantic import BaseModel

from crewai import Agent, Crew, Task
from crewai.tools import tool
from crewai.flow.flow import Flow, listen, or_, router, start
from ddgs import DDGS

load_dotenv()

MODEL = "gpt-4o-mini"
TOPIC = "India's DPDP Act obligations for AI systems handling customer data"
MAX_ATTEMPTS = 3


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


researcher = Agent(
    role="Research Analyst",
    goal="Find and cite source material for a topic",
    backstory=(
        "You verify every claim against a source before you report it. "
        "You never present an unsourced statement as fact."
    ),
    tools=[ddg_search],
    llm=MODEL, max_iter=5, verbose=True,
)
writer = Agent(
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
    goal="Flag any claim in the brief that is not supported by the findings",
    backstory=(
        "You are hard to satisfy. You treat a claim as unsupported "
        "until you have seen it in findings."
    ),
    tools=[],
    llm=MODEL, max_iter=5, verbose=True,
)


def run_crew(agent, description, expected_output):
    """One Flow step = one small Crew with a single agent and task."""
    task = Task(description=description, expected_output=expected_output, agent=agent)
    crew = Crew(agents=[agent], tasks=[task], verbose=True)
    return str(crew.kickoff()).strip()


class BriefState(BaseModel):
    topic: str = TOPIC
    findings: str = ""
    brief: str = ""
    feedback: str = ""
    verdict: str = ""
    attempts: int = 0


class BriefFlow(Flow[BriefState]):

    @start()
    def research(self):
        self.state.findings = run_crew(
            researcher,
            f"Research the topic: {self.state.topic}. Gather concrete findings "
            "and list a supporting source URL for each.",
            "A list of findings, each with its supporting source URL.",
        )

    # Runs after research, and again each time review routes to "revise".
    @listen(or_(research, "revise"))
    def write(self):
        revision = (
            f"\n\nFix these issues from the last review:\n{self.state.feedback}"
            if self.state.feedback else ""
        )
        self.state.brief = run_crew(
            writer,
            f"Write a 200-word brief on {self.state.topic} using only these findings:\n"
            f"{self.state.findings}\nDo not introduce facts that are not in the findings."
            f"{revision}",
            "A brief of about 200 words with no invented facts.",
        )

    # The router branches on the critic's verdict. Its return value is a label
    # that picks which @listen method runs next.
    @router(write)
    def review(self):
        self.state.attempts += 1
        self.state.verdict = run_crew(
            critic,
            "Compare the brief against the findings.\n\n"
            f"FINDINGS:\n{self.state.findings}\n\nBRIEF:\n{self.state.brief}\n\n"
            "If every claim is supported, reply with exactly APPROVED. "
            "Otherwise reply with a numbered list of required revisions.",
            "Either 'APPROVED' or a numbered revision list.",
        )
        if self.state.verdict.upper().startswith("APPROVED"):
            return "approved"
        if self.state.attempts >= MAX_ATTEMPTS:
            return "escalate"   # recovery branch: stop looping and flag for a human
        self.state.feedback = self.state.verdict
        return "revise"

    @listen("approved")
    def publish(self):
        print("\nBrief approved and published.")

    @listen("escalate")
    def escalate(self):
        print(f"\nStill not supported after {MAX_ATTEMPTS} attempts. Needs human review.")


def main():
    flow = BriefFlow()
    # flow.plot("lab2_flow") draws the branching graph (requires graphviz).

    t0 = time.perf_counter()
    flow.kickoff()
    wall = time.perf_counter() - t0

    print(f"\nFinal verdict after {flow.state.attempts} review(s): {flow.state.verdict}")
    print(f"\n{flow.state.brief}")
    print(f"\nWall time: {wall:.1f}s")


if __name__ == "__main__":
    main()
