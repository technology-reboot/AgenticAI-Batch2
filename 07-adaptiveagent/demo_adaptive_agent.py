"""
RL / RLHF · Demo — An adaptive agent adjusting to new inputs
============================================================

WHAT THIS DEMO ARGUES
---------------------
An agent that has stopped exploring is not adaptive, however well it performs
today. You cannot see that from a performance number. You can only see it by
changing the world after the agent has finished learning, and watching what
happens next.

So this demo is built around a NON-STATIONARY environment. The agent learns a
good policy, then the world changes underneath it — twice — and you watch it
recover, or fail to.

THE SETUP
---------
A support-response agent. Deliberately small enough to hold in your head:

    CONTEXT (the input)   the ticket category
    ACTION  (the policy)  one of three response styles
    REWARD                thumbs up / thumbs down

Algorithm: an epsilon-greedy CONTEXTUAL BANDIT — one action-value row per
context, updated by incremental averaging:

    Q(context, action)  <-  Q + (1/N) * (reward - Q)

This is genuine reinforcement learning: the single-step case. Be upfront with
the room about what it does NOT show — no state transitions, no credit
assignment over a sequence. That is where PPO and the heavier machinery earn
their keep. Say it before someone in the third row says it for you.

RUN OF SHOW
-----------
    PHASE 1   the room is the reward function      (~3 min, interactive)
    PHASE 2   hand over to a simulated user        (~2 min)
    PHASE 3   REGIME CHANGE - preferences flip     (~3 min)
    PHASE 4   REGIME CHANGE - a new context appears
    PHASE 5   the kill shot: epsilon = 0 vs 0.1    (~3 min)
    PHASE 6   the bridge to RLHF - a learned reward

WHERE THE LLM CALLS ARE
-----------------------
The bandit itself needs no LLM — that is the point of a bandit. Two places use
one for real, and both are pedagogically load-bearing:

    PHASE 1   an LLM writes the three candidate responses for each ticket, so
              the room votes on real text rather than on labels like "action 2"

    PHASE 6   an LLM acts as the reward model, replacing the scripted user.
              This is the RLHF bridge made concrete: same loop, learned reward.

Run with --no-llm to skip both. The RL mechanics run identically offline.

SETUP
-----
    pip install langchain-openai matplotlib python-dotenv

    .env:
        OPENAI_API_KEY=sk-...

    python demo_adaptive_agent.py              # full demo
    python demo_adaptive_agent.py --no-llm     # offline, no API key needed
    python demo_adaptive_agent.py --fast       # skip the interactive phase
"""

from __future__ import annotations

import argparse
import os
import random
import sys
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Ensure UTF-8 console output on Windows (arrows, middle dots, etc.).
try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass

# --------------------------------------------------------------------------
# The world
# --------------------------------------------------------------------------

N_RUNS = 20  # the kill-shot chart is averaged over this many runs

ACTIONS = ["terse_fix", "empathetic_fix", "escalate"]

ACTION_LABEL = {
    "terse_fix": "Terse fix",
    "empathetic_fix": "Empathetic + fix",
    "escalate": "Escalate to human",
}

# Contexts present from the start. 'security_incident' is deliberately absent —
# it arrives in Phase 4 as a genuinely new input.
BASE_CONTEXTS = ["billing", "outage", "how_to"]

SAMPLE_TICKETS = {
    "billing": "I was charged twice for my March invoice. Refund it.",
    "outage": "Your API has been returning 503s for the last twenty minutes. "
    "This is taking down our checkout.",
    "how_to": "How do I rotate my API key without breaking the running jobs?",
    "security_incident": "I think our API key has been leaked — I can see calls "
    "from an IP we don't own.",
}

# --------------------------------------------------------------------------
# The reward function — a scripted user, with three regimes.
#
# These are the PROBABILITIES of a thumbs-up. The agent never sees this table;
# it only ever sees a 0 or a 1. Keep it on screen when you explain regret:
# regret is computed against this table, which is exactly the thing you never
# have in real life.
# --------------------------------------------------------------------------

REGIMES: dict[str, dict[str, dict[str, float]]] = {
    # Regime 1 — the world the agent first learns
    "initial": {
        "billing": {"terse_fix": 0.85, "empathetic_fix": 0.55, "escalate": 0.20},
        "outage": {"terse_fix": 0.15, "empathetic_fix": 0.35, "escalate": 0.70},
        "how_to": {"terse_fix": 0.45, "empathetic_fix": 0.85, "escalate": 0.15},
    },
    # Regime 2 — the company hires a 24x7 in-house support team.
    #
    # Read the outage row carefully, because the whole demo turns on it:
    # ESCALATE IS UNCHANGED at 0.70. Customers who get escalated are exactly
    # as happy as they always were. Nothing got worse. What changed is that
    # TERSE_FIX jumped from 0.15 to 0.98 — a much better option now exists.
    #
    # This is the kind of change a greedy agent is structurally blind to. It
    # keeps escalating, keeps collecting the same good reward it always did,
    # and receives no signal whatsoever that the world moved. You cannot learn
    # about an action you never take.
    "flipped": {
        "billing": {"terse_fix": 0.85, "empathetic_fix": 0.55, "escalate": 0.20},
        "outage": {"terse_fix": 0.98, "empathetic_fix": 0.35, "escalate": 0.70},
        "how_to": {"terse_fix": 0.45, "empathetic_fix": 0.85, "escalate": 0.15},
    },
    # Regime 3 — a context the agent has never seen starts arriving.
    "new_context": {
        "billing": {"terse_fix": 0.85, "empathetic_fix": 0.55, "escalate": 0.20},
        "outage": {"terse_fix": 0.98, "empathetic_fix": 0.35, "escalate": 0.70},
        "how_to": {"terse_fix": 0.45, "empathetic_fix": 0.85, "escalate": 0.15},
        "security_incident": {"terse_fix": 0.15, "empathetic_fix": 0.25, "escalate": 0.92},
    },
}


class ScriptedUser:
    """Supplies a 0/1 reward. Swappable — Phase 1 uses hands, Phase 6 an LLM."""

    def __init__(self, regime: str = "initial", seed: int | None = None) -> None:
        self.regime = regime
        self.rng = random.Random(seed)

    @property
    def table(self) -> dict[str, dict[str, float]]:
        return REGIMES[self.regime]

    def contexts(self) -> list[str]:
        return list(self.table.keys())

    def reward(self, context: str, action: str) -> int:
        return 1 if self.rng.random() < self.table[context][action] else 0

    # --- used only to COMPUTE REGRET, never visible to the agent ----------
    def best_expected(self, context: str) -> float:
        return max(self.table[context].values())

    def expected(self, context: str, action: str) -> float:
        return self.table[context][action]


# --------------------------------------------------------------------------
# The agent
# --------------------------------------------------------------------------


class ContextualBandit:
    """Epsilon-greedy contextual bandit with incremental-average updates.

    New contexts need no special handling: the defaultdict creates an empty row
    the first time one is seen, and every action in it has value 0. That is why
    Phase 4 works without a single line of code about 'new inputs'.
    """

    def __init__(
        self,
        epsilon: float = 0.10,
        alpha: float | None = 0.15,
        epsilon_floor: float = 0.10,
        decay_start: int = 0,
        decay_steps: int | None = None,
        seed: int | None = None,
    ) -> None:
        # alpha is the step size, and it matters as much as epsilon.
        #   alpha = None  -> sample average, Q += (r - Q)/N. Correct for a
        #                    STATIONARY world. In a changing one, old evidence
        #                    never fades: an action with 80 stale observations
        #                    barely moves no matter what happens next.
        #   alpha = 0.15  -> constant step, Q += 0.15 * (r - Q). Recent rewards
        #                    outweigh old ones. The agent can FORGET.
        # Exploration finds the change. The step size is what lets the agent
        # act on it. You need both. Run with --sample-average to watch an
        # exploring agent fail anyway.
        self.alpha = alpha
        # Exploration schedule. decay_steps=None keeps epsilon fixed forever.
        # Otherwise epsilon falls linearly to epsilon_floor over decay_steps
        # interactions, beginning at decay_start — which is what most teams
        # actually ship: "it has converged, stop spending traffic on it."
        self.epsilon_start = epsilon
        self.epsilon_floor = epsilon_floor
        self.decay_start = decay_start
        self.decay_steps = decay_steps
        self.t = 0
        self.q: dict[tuple[str, str], float] = defaultdict(float)
        self.n: dict[tuple[str, str], int] = defaultdict(int)
        self.rng = random.Random(seed)
        self.last_was_exploration = False

    @property
    def epsilon(self) -> float:
        if self.decay_steps is None or self.t <= self.decay_start:
            return self.epsilon_start
        frac = min(1.0, (self.t - self.decay_start) / self.decay_steps)
        return self.epsilon_start + frac * (self.epsilon_floor - self.epsilon_start)

    def select(self, context: str) -> str:
        self.t += 1
        if self.rng.random() < self.epsilon:
            self.last_was_exploration = True
            return self.rng.choice(ACTIONS)
        self.last_was_exploration = False
        # argmax with random tie-breaking — important on an unseen context,
        # where every action is still worth 0.
        best = max(self.q[(context, a)] for a in ACTIONS)
        return self.rng.choice([a for a in ACTIONS if self.q[(context, a)] == best])

    def update(self, context: str, action: str, reward: float) -> None:
        key = (context, action)
        self.n[key] += 1
        step = self.alpha if self.alpha is not None else 1.0 / self.n[key]
        self.q[key] += step * (reward - self.q[key])

    def policy(self, context: str) -> str:
        best = max(self.q[(context, a)] for a in ACTIONS)
        for a in ACTIONS:
            if self.q[(context, a)] == best:
                return a
        return ACTIONS[0]


# --------------------------------------------------------------------------
# Display
# --------------------------------------------------------------------------


def banner(text: str, char: str = "=") -> None:
    print(f"\n{char * 78}\n {text}\n{char * 78}")


def value_table(agent: ContextualBandit, contexts: list[str], user: ScriptedUser | None = None):
    """Keep this on screen. People need to watch the numbers move."""
    head = f"  {'context':<20}" + "".join(f"{ACTION_LABEL[a]:>20}" for a in ACTIONS)
    print(head)
    print("  " + "-" * (len(head) - 2))
    for ctx in contexts:
        row = f"  {ctx:<20}"
        for a in ACTIONS:
            q, n = agent.q[(ctx, a)], agent.n[(ctx, a)]
            cell = f"{q:.2f} (n={n})"
            row += f"{cell:>20}"
        chosen = agent.policy(ctx)
        marker = f"   -> {ACTION_LABEL[chosen]}"
        if user is not None:
            ideal = max(user.table[ctx], key=user.table[ctx].get)
            marker += "  [OPTIMAL]" if chosen == ideal else f"  [should be {ACTION_LABEL[ideal]}]"
        print(row + marker)


# --------------------------------------------------------------------------
# The training loop — one function, used by every phase
# --------------------------------------------------------------------------


def run_rounds(
    agent: ContextualBandit,
    user: ScriptedUser,
    rounds: int,
    regret: list[float],
    label: str = "",
    report_every: int = 0,
) -> None:
    """Run `rounds` interactions, appending cumulative regret after each one."""
    contexts = user.contexts()
    for i in range(rounds):
        ctx = agent.rng.choice(contexts)
        action = agent.select(ctx)
        reward = user.reward(ctx, action)
        agent.update(ctx, action, reward)

        # Regret uses the hidden table. It measures what the agent gave up by
        # not knowing what we know. It is a teaching instrument, not something
        # you can compute in production.
        step_regret = user.best_expected(ctx) - user.expected(ctx, action)
        regret.append((regret[-1] if regret else 0.0) + step_regret)

        if report_every and (i + 1) % report_every == 0:
            print(f"    {label}round {i + 1:>4}   cumulative regret {regret[-1]:6.1f}")


# --------------------------------------------------------------------------
# LLM integration — Phase 1 candidate responses, Phase 6 reward model
# --------------------------------------------------------------------------

_llm = None
_response_cache: dict[str, dict[str, str]] = {}
_reward_cache: dict[tuple[str, str], int] = {}


def get_llm(model: str = "gpt-4o-mini"):
    global _llm
    if _llm is None:
        from langchain_openai import ChatOpenAI

        _llm = ChatOpenAI(model=model, temperature=0.3)
    return _llm


def generate_responses(context: str) -> dict[str, str]:
    """One real LLM call per context. Gives the room actual text to vote on."""
    if context in _response_cache:
        return _response_cache[context]

    ticket = SAMPLE_TICKETS[context]
    prompt = f"""A customer has raised this support ticket:

"{ticket}"

Write three candidate replies, one in each style below. Each must be 2-3
sentences. Return them in exactly this format, with no other text:

TERSE: <a direct fix, no pleasantries>
EMPATHETIC: <acknowledges the frustration, then the same fix>
ESCALATE: <hands the issue to a human specialist>"""

    text = get_llm().invoke(prompt).content
    out: dict[str, str] = {}
    for line in text.splitlines():
        for key, action in (("TERSE:", "terse_fix"), ("EMPATHETIC:", "empathetic_fix"), ("ESCALATE:", "escalate")):
            if line.strip().upper().startswith(key):
                out[action] = line.split(":", 1)[1].strip()
    # Fall back to a label if parsing missed one — never crash a live demo.
    for a in ACTIONS:
        out.setdefault(a, f"[{ACTION_LABEL[a]}]")
    _response_cache[context] = out
    return out


REWARD_MODEL_PROMPT = """You are standing in for a customer who raised this \
support ticket:

Ticket: "{ticket}"

You received this reply:

"{reply}"

Would you give it a thumbs up? Reply with exactly one word: UP or DOWN.

Judge as a real customer would. An outage needs urgency and ownership, not
sympathy. A billing error needs the money back, briefly. A how-to question from
someone who is stuck deserves patience."""


def llm_reward(context: str, action: str, responses: dict[str, dict[str, str]]) -> int:
    """Phase 6: an LLM as the reward function.

    Cached on (context, action) because our candidate text is fixed per pair —
    a real reward model would score every generated response individually. Say
    this out loud; it is the difference between a demo and a system.
    """
    key = (context, action)
    if key in _reward_cache:
        return _reward_cache[key]
    verdict = (
        get_llm()
        .invoke(
            REWARD_MODEL_PROMPT.format(
                ticket=SAMPLE_TICKETS[context], reply=responses[context][action]
            )
        )
        .content.strip()
        .upper()
    )
    r = 1 if verdict.startswith("UP") else 0
    _reward_cache[key] = r
    return r


# ==========================================================================
# PHASES
# ==========================================================================


def phase1_human_reward(agent: ContextualBandit, use_llm: bool, rounds: int = 8) -> None:
    banner("PHASE 1 — the room is the reward function")
    print(
        """ Ten seconds of framing before you start:

   'Reinforcement learning needs a reward signal. Right now, that signal is
    you. I am going to show you a ticket and three replies. Hands up for the
    one you would give a thumbs up. That number is the entire training
    signal — there is nothing else.'

 Run this slowly. It is meant to feel laborious. That feeling is the argument
 for Phase 6.
"""
    )

    if not sys.stdin.isatty():
        print(" [non-interactive terminal — skipping the voting phase]")
        return

    user = ScriptedUser("initial", seed=7)
    for i in range(rounds):
        ctx = agent.rng.choice(BASE_CONTEXTS)
        print(f"\n --- round {i + 1} · context: {ctx} ---")
        print(f' Ticket: "{SAMPLE_TICKETS[ctx]}"\n')

        if use_llm:
            replies = generate_responses(ctx)
        else:
            replies = {a: f"[{ACTION_LABEL[a]}]" for a in ACTIONS}

        for n, a in enumerate(ACTIONS, 1):
            print(f"   {n}. {ACTION_LABEL[a]}")
            print(f"      {replies[a]}\n")

        action = agent.select(ctx)
        tag = "EXPLORE" if agent.last_was_exploration else "exploit"
        print(f" Agent chose: {ACTION_LABEL[action]}   [{tag}]")

        raw = input(" Thumbs up? (y/n, or Enter to skip): ").strip().lower()
        if raw in ("y", "n"):
            agent.update(ctx, action, 1 if raw == "y" else 0)
        print()
        value_table(agent, BASE_CONTEXTS, user)


def phase2_converge(agent: ContextualBandit, regret: list[float]) -> ScriptedUser:
    banner("PHASE 2 — hand the reward function to a simulated user")
    print(
        " Same loop, 300 rounds, one second. The agent is now learning from a\n"
        " scripted preference model instead of from hands.\n"
    )
    user = ScriptedUser("initial", seed=11)
    run_rounds(agent, user, 300, regret, report_every=100)
    print()
    value_table(agent, BASE_CONTEXTS, user)
    print(
        "\n Ask the room to read the policy off the table before you say it:\n"
        "   billing -> terse, outage -> escalate, how-to -> empathetic.\n"
        " The agent was told none of that."
    )
    return user


def phase3_flip(agent: ContextualBandit, regret: list[float]) -> ScriptedUser:
    banner("PHASE 3 — REGIME CHANGE · the world moves under the agent")
    print(
        """ Nothing about the agent changes. The WORLD changes:

   'Meridian has just hired a 24x7 in-house support team.'

 Now read the outage row of the preference table before you run it. Escalating
 an outage is EXACTLY as good as it always was — 0.70, unchanged. Nobody is
 unhappier than before. What changed is that a terse in-house fix went from
 0.15 to 0.98.

 Nothing got worse. Something got better. Ask the room how an agent that only
 ever escalates outages is supposed to find that out.
"""
    )
    user = ScriptedUser("flipped", seed=13)
    print(" BEFORE the change, the agent's policy is:")
    value_table(agent, BASE_CONTEXTS, user)

    run_rounds(agent, user, 600, regret, report_every=200)
    print("\n AFTER 600 more rounds in the new world:")
    value_table(agent, BASE_CONTEXTS, user)
    print(
        f"""
 Nobody retrained anything. Two things had to be true for that recovery, and
 the room should be able to name both:

   1. EXPLORATION found the change. epsilon = {agent.epsilon} means about
      {agent.epsilon:.0%} of actions were deliberately not the best-known one.
      A greedy agent would never have tried 'terse fix' on an outage again.

   2. A CONSTANT STEP SIZE let it act on what it found. alpha = {agent.alpha}
      means each new reward moves Q by a fixed fraction, so recent evidence
      outweighs old evidence and the agent can forget.

 Re-run with --sample-average to see an agent that explores just as hard,
 finds the change just as fast, and still cannot escape the old policy —
 because 80 stale observations drown 40 fresh ones."""
    )
    return user


def phase4_new_context(agent: ContextualBandit, regret: list[float]) -> ScriptedUser:
    banner("PHASE 4 — REGIME CHANGE · a genuinely new input arrives")
    print(
        """ A ticket category the agent has never seen starts arriving:

   security_incident: "I think our API key has been leaked."

 There is no code in this program that handles new contexts. The value table
 simply grows a row of zeros, and the agent learns it the same way it learned
 the others.
"""
    )
    user = ScriptedUser("new_context", seed=17)
    print(" The new row starts empty:")
    value_table(agent, ["security_incident"], user)

    run_rounds(agent, user, 600, regret, report_every=300)
    print("\n After 600 rounds:")
    value_table(agent, user.contexts(), user)
    return user


def phase5_kill_shot(alpha: float | None = 0.15, seed: int = 3):
    banner("PHASE 5 — THE KILL SHOT · exploration that stops, versus exploration that doesn't")
    print(
        """ Two agents, identical in every way except one line of configuration.

   AGENT A  epsilon held at 0.10 for life. Pays the exploration tax every
            single day, long after it has converged.

   AGENT B  same agent, but epsilon is decayed to 0 once it has converged
            (rounds 150-250). This is not a strawman — it is what most teams
            ship, and the argument for it is perfectly reasonable: "it has
            converged, stop spending live traffic on actions we already know
            are worse."

 Both reach the same policy before the world changes, and B does it slightly
 more cheaply. Watch the dashed lines — the whole demo is what happens after
 the first one.

 Averaged over 20 runs, so this is evidence rather than an anecdote.
"""
    )

    configs = {
        "A · exploration held at 0.10": dict(epsilon=0.10, epsilon_floor=0.10, decay_steps=None),
        "B · exploration decayed to 0": dict(
            epsilon=0.10, epsilon_floor=0.0, decay_start=150, decay_steps=100
        ),
    }

    # Averaged over N_RUNS seeds rather than shown for one. A single run of a
    # stochastic system is an anecdote — and on an unlucky seed, agent B gets
    # away with it. Averaging is both better evidence and one less thing to go
    # wrong in front of a room.
    n_runs = N_RUNS
    curves: dict[str, list[float]] = {}
    for name, cfg in configs.items():
        totals: list[float] | None = None
        recovered = 0
        for run in range(n_runs):
            agent = ContextualBandit(alpha=alpha, seed=seed + run, **cfg)
            regret: list[float] = []
            run_rounds(agent, ScriptedUser("initial", seed=11 + run), 300, regret)
            run_rounds(agent, ScriptedUser("flipped", seed=13 + run), 600, regret)
            run_rounds(agent, ScriptedUser("new_context", seed=17 + run), 600, regret)
            recovered += agent.policy("outage") == "terse_fix"
            totals = regret if totals is None else [a + b for a, b in zip(totals, regret)]
        mean = [v / n_runs for v in totals]
        curves[name] = mean
        print(f"   {name}")
        print(f"     found the better outage response in {recovered}/{n_runs} runs")
        print(f"     mean cumulative regret: {mean[-1]:.1f}\n")

    plot_regret(curves)
    return curves


def plot_regret(curves: dict[float, list[float]], path: str = "adaptive_agent_regret.png") -> None:
    plt.figure(figsize=(10, 5.5))
    colors = ["#2E5D72", "#C0392B"]
    for (label, regret), color in zip(curves.items(), colors):
        plt.plot(regret, color=color, linewidth=2, label=label)

    plt.axvline(300, color="#888888", linestyle="--", linewidth=1)
    plt.axvline(900, color="#888888", linestyle="--", linewidth=1)
    top = plt.ylim()[1]
    plt.text(308, top * 0.78, "preferences flip", fontsize=9, color="#555555")
    plt.text(908, top * 0.78, "new context appears", fontsize=9, color="#555555")

    plt.xlabel("interaction")
    plt.ylabel("mean cumulative regret  (lower is better)")
    plt.title(f"An agent that stops exploring stops adapting  "
              f"(mean of {N_RUNS} runs)")
    plt.legend(loc="upper left")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(path, dpi=140)
    print(f"   chart written to {path}")
    print(
        "\n   Put this chart on the screen and say the line the demo exists for:\n"
        "   'Exploration is what you pay for adaptability. Both agents were\n"
        "    good at their job. Only one of them was still able to learn.'"
    )


def phase6_rlhf_bridge(use_llm: bool, rounds: int = 60) -> None:
    banner("PHASE 6 — the bridge to RLHF · a learned reward function")
    print(
        """ Ask the room where the reward came from in each phase:

   Phase 1   your hands
   Phase 2-5 a hard-coded probability table
   Phase 6   a MODEL of human preference

 That third one is RLHF in one sentence: you cannot keep humans in the loop for
 every decision, so you collect enough of their judgements to train a model of
 them, and use that model as the reward signal. Same loop. Learned reward.
"""
    )

    if not use_llm:
        print(" [--no-llm: skipping. This phase needs real model calls to mean anything.]")
        return

    print(" Generating candidate responses for each context (real LLM calls)...")
    responses = {ctx: generate_responses(ctx) for ctx in BASE_CONTEXTS}

    print(" Scoring each (context, action) pair with the LLM reward model...\n")
    agent = ContextualBandit(epsilon=0.10, alpha=0.15, seed=5)
    for _ in range(rounds):
        ctx = agent.rng.choice(BASE_CONTEXTS)
        action = agent.select(ctx)
        agent.update(ctx, action, llm_reward(ctx, action, responses))

    print("  reward model verdicts (1 = thumbs up):")
    for (ctx, action), r in sorted(_reward_cache.items()):
        print(f"    {ctx:<20} {ACTION_LABEL[action]:<20} {r}")

    print("\n  policy learned from the LLM reward model:")
    value_table(agent, BASE_CONTEXTS)
    print(
        """
 Now the honest part. Compare this policy against the scripted one from
 Phase 2. Where they disagree, ask: which is wrong — the LLM's model of a
 customer, or our hand-written table? Neither is the truth. That gap IS the
 reward-hacking problem, and it is why RLHF pipelines keep humans auditing
 the reward model rather than trusting it.
"""
    )


# ==========================================================================


def main() -> None:
    parser = argparse.ArgumentParser(description="Adaptive agent demo for the RL/RLHF session.")
    parser.add_argument("--no-llm", action="store_true", help="run without any API calls")
    parser.add_argument("--fast", action="store_true", help="skip the interactive voting phase")
    parser.add_argument(
        "--sample-average",
        action="store_true",
        help="use Q += (r-Q)/N instead of a constant step size — shows why "
        "exploration alone does not make an agent adaptive",
    )
    args = parser.parse_args()

    use_llm = not args.no_llm
    if use_llm and not os.getenv("OPENAI_API_KEY"):
        try:
            from dotenv import load_dotenv

            load_dotenv()
        except ImportError:
            pass
    if use_llm and not os.getenv("OPENAI_API_KEY"):
        print("No OPENAI_API_KEY found — falling back to --no-llm.\n")
        use_llm = False

    alpha = None if args.sample_average else 0.10
    if args.sample_average:
        print(
            "Running with SAMPLE-AVERAGE updates (alpha = 1/N).\n"
            "Expect the agent to explore normally and still fail to recover "
            "in Phase 3.\n"
        )

    agent = ContextualBandit(epsilon=0.10, alpha=alpha, seed=3)
    regret: list[float] = []

    if not args.fast:
        phase1_human_reward(agent, use_llm)

    phase2_converge(agent, regret)
    phase3_flip(agent, regret)
    phase4_new_context(agent, regret)
    phase5_kill_shot(alpha=alpha)
    phase6_rlhf_bridge(use_llm)

    banner("DISCUSSION", "-")
    print(
        """
 1. Before the first dashed line, the greedy agent was slightly AHEAD. If you
    were reviewing these two agents on a dashboard in week one, which would
    you have shipped?

 2. Run the demo again with --sample-average. The agent explores exactly as
    much, and still never recovers in Phase 3. Explain to someone why, in one
    sentence, without using the word "alpha".

 3. Epsilon is fixed at 0.10 forever here, so the agent never stops paying the
    exploration tax. What would you actually do in production — decay it, and
    accept that you lose adaptability? Or keep a floor, and pay forever?

 4. Regret is computed against the hidden preference table. You will never have
    that table in production. What do you monitor instead to notice that your
    world has changed?

 5. This is a contextual bandit — one decision, immediate reward. Name a
    support-agent decision where the reward arrives five steps later, and say
    what breaks in this code if you try to handle it.

 6. In Phase 6 the LLM judged its own three candidate replies. What would you
    need to collect, and from whom, before you would trust that judgement to
    train a policy you actually deploy?
"""
    )


if __name__ == "__main__":
    main()
