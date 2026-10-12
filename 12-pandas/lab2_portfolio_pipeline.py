"""
Lab 2 - Pandas data pipeline: clean and prepare the capstone portfolio CSV

Objective
---------
Use pandas to clean, transform, and summarise a messy portfolio CSV, then
format the output as structured JSON/text ready for injection into the
capstone's Analysis Agent context.

This file is written with "# %%" cell markers so it can be run top-to-bottom
as a plain script (`python lab2_portfolio_pipeline.py`) or opened in VS
Code / Jupyter as an interactive notebook (each "# %%" becomes a cell).

Steps covered
-------------
1. Load messy_portfolio.csv (generated automatically if not already present)
2. Clean: strip currency symbols/commas, parse mixed date formats,
   fill missing sectors, drop duplicate rows
3. Enrich: current_value, pnl, weight_pct
4. Aggregate: sector_allocation, top_5 holdings, total_pnl_pct, count_of_losers
5. Format: portfolio_to_context(df) -> structured string for LLM injection
6. Test: send the context string to GPT-4o-mini and compare it against
   dumping the raw CSV straight into the prompt
"""

# %% [1] Imports -------------------------------------------------------------
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Windows terminals default to cp1252, which can't print the Rupee symbol.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

CSV_PATH = Path(__file__).with_name("messy_portfolio.csv")


# %% [2] Step 1: generate + load the messy CSV --------------------------------
def generate_messy_csv(path: Path = CSV_PATH, seed: int = 42) -> Path:
    """Create a realistic messy_portfolio.csv for the lab if one isn't present.

    Deliberately messy: Rupee symbols + thousands separators on numeric
    columns, a handful of missing sectors, duplicate rows, and purchase
    dates written in three different formats.
    """
    rng = np.random.default_rng(seed)

    holdings = [
        ("RELIANCE", "Reliance Industries", "Energy"),
        ("TCS", "Tata Consultancy Services", "IT"),
        ("INFY", "Infosys", "IT"),
        ("HDFCBANK", "HDFC Bank", "Financials"),
        ("ICICIBANK", "ICICI Bank", "Financials"),
        ("HINDUNILVR", "Hindustan Unilever", "FMCG"),
        ("ITC", "ITC Limited", "FMCG"),
        ("SBIN", "State Bank of India", "Financials"),
        ("BHARTIARTL", "Bharti Airtel", "Telecom"),
        ("LT", "Larsen & Toubro", "Industrials"),
        ("ASIANPAINT", "Asian Paints", "Materials"),
        ("MARUTI", "Maruti Suzuki", "Auto"),
        ("SUNPHARMA", "Sun Pharma", "Healthcare"),
        ("TITAN", "Titan Company", "Consumer Durables"),
        ("WIPRO", "Wipro", "IT"),
        ("ADANIENT", "Adani Enterprises", None),
        ("TATAMOTORS", "Tata Motors", "Auto"),
        ("ONGC", "Oil & Natural Gas Corp", None),
        ("NTPC", "NTPC Limited", "Energy"),
        ("ULTRACEMCO", "UltraTech Cement", None),
    ]

    date_formats = ["%Y-%m-%d", "%d/%m/%Y", "%b %d, %Y"]
    rows = []
    for ticker, name, sector in holdings:
        shares = int(rng.integers(10, 500))
        cost_per_share = round(float(rng.uniform(150, 3500)), 2)
        cost_basis = round(shares * cost_per_share, 2)
        price_stub = round(cost_per_share * float(rng.uniform(0.7, 1.4)), 2)

        start = pd.Timestamp("2021-01-01")
        end = pd.Timestamp("2024-06-30")
        random_days = int(rng.integers(0, (end - start).days))
        purchase_date = start + pd.Timedelta(days=random_days)
        fmt = date_formats[rng.integers(0, len(date_formats))]
        date_str = purchase_date.strftime(fmt)

        # format numeric fields the messy way: "₹1,23,456.78"
        cost_str = f"₹{cost_basis:,.2f}"
        price_str = f"₹{price_stub:,.2f}"

        rows.append(
            {
                "ticker": ticker,
                "company_name": name,
                "sector": sector,
                "shares": shares,
                "cost_basis": cost_str,
                "price_stub": price_str,
                "purchase_date": date_str,
            }
        )

    df = pd.DataFrame(rows)

    # inject a few exact duplicate rows
    dupes = df.sample(n=3, random_state=seed)
    df = pd.concat([df, dupes], ignore_index=True)
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)

    df.to_csv(path, index=False, encoding="utf-8")
    return path


def load_portfolio(path: Path = CSV_PATH) -> pd.DataFrame:
    if not path.exists():
        generate_messy_csv(path)
    return pd.read_csv(path)


# %% [3] Step 2: clean ---------------------------------------------------------
def clean_portfolio(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # drop exact duplicate rows
    df = df.drop_duplicates().reset_index(drop=True)

    # strip currency symbols/commas from numeric columns and cast to float
    currency_cols = ["cost_basis", "price_stub"]
    for col in currency_cols:
        df[col] = (
            df[col]
            .astype(str)
            .apply(lambda v: re.sub(r"[^\d.\-]", "", v))
            .replace("", np.nan)
            .astype(float)
        )

    df["shares"] = pd.to_numeric(df["shares"], errors="coerce").astype(int)

    # parse mixed date formats (ISO, DD/MM/YYYY, "Mon DD, YYYY")
    df["purchase_date"] = pd.to_datetime(
        df["purchase_date"], format="mixed", dayfirst=True, errors="coerce"
    )

    # fill missing sectors
    df["sector"] = df["sector"].fillna("Unclassified")

    df = df.dropna().reset_index(drop=True)
    return df


# %% [4] Step 3: enrich ---------------------------------------------------------
def enrich_portfolio(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["current_value"] = df["shares"] * df["price_stub"]
    df["pnl"] = df["current_value"] - df["cost_basis"]
    total_value = df["current_value"].sum()
    df["weight_pct"] = (df["current_value"] / total_value * 100).round(2)
    return df


# %% [5] Step 4: aggregate -------------------------------------------------------
def aggregate_portfolio(df: pd.DataFrame) -> dict:
    sector_allocation = (
        df.groupby("sector")["current_value"]
        .sum()
        .div(df["current_value"].sum())
        .mul(100)
        .round(2)
        .sort_values(ascending=False)
        .to_dict()
    )

    top_5 = (
        df.sort_values("current_value", ascending=False)
        .head(5)[["ticker", "company_name", "current_value", "weight_pct", "pnl"]]
        .round(2)
        .to_dict(orient="records")
    )

    total_cost = df["cost_basis"].sum()
    total_pnl = df["pnl"].sum()
    total_pnl_pct = round((total_pnl / total_cost) * 100, 2) if total_cost else 0.0
    count_of_losers = int((df["pnl"] < 0).sum())

    summary = {
        "as_of": pd.Timestamp.today().strftime("%Y-%m-%d"),
        "total_holdings": int(len(df)),
        "total_current_value": round(float(df["current_value"].sum()), 2),
        "total_cost_basis": round(float(total_cost), 2),
        "total_pnl": round(float(total_pnl), 2),
        "total_pnl_pct": total_pnl_pct,
        "count_of_losers": count_of_losers,
        "sector_allocation": sector_allocation,
        "top_5_holdings": top_5,
    }
    return summary


# %% [6] Step 5: format for LLM injection ----------------------------------------
def portfolio_to_context(df: pd.DataFrame, summary: dict | None = None) -> str:
    """Render the cleaned portfolio + summary as a structured text block
    suitable for injecting into an LLM's context window."""
    if summary is None:
        summary = aggregate_portfolio(df)

    lines = []
    lines.append("### PORTFOLIO SNAPSHOT")
    lines.append(f"As of: {summary['as_of']}")
    lines.append(f"Total holdings: {summary['total_holdings']}")
    lines.append(f"Total current value: ₹{summary['total_current_value']:,.2f}")
    lines.append(f"Total cost basis: ₹{summary['total_cost_basis']:,.2f}")
    lines.append(
        f"Total P&L: ₹{summary['total_pnl']:,.2f} ({summary['total_pnl_pct']}%)"
    )
    lines.append(f"Positions currently at a loss: {summary['count_of_losers']}")

    lines.append("\n### SECTOR ALLOCATION (% of portfolio value)")
    for sector, pct in summary["sector_allocation"].items():
        lines.append(f"- {sector}: {pct}%")

    lines.append("\n### TOP 5 HOLDINGS")
    for h in summary["top_5_holdings"]:
        lines.append(
            f"- {h['ticker']} ({h['company_name']}): "
            f"value ₹{h['current_value']:,.2f}, "
            f"weight {h['weight_pct']}%, "
            f"P&L ₹{h['pnl']:,.2f}"
        )

    lines.append("\n### FULL HOLDINGS TABLE")
    table_cols = [
        "ticker",
        "sector",
        "shares",
        "cost_basis",
        "current_value",
        "pnl",
        "weight_pct",
    ]
    lines.append(df[table_cols].round(2).to_csv(index=False))

    return "\n".join(lines)


# %% [7] Step 6: test with GPT-4o-mini --------------------------------------------
def ask_llm(prompt: str, system: str = "You are a portfolio analyst.") -> str:
    """Send a prompt to GPT-4o-mini and return the response text.
    Requires OPENAI_API_KEY to be set in the environment."""
    from openai import OpenAI

    client = OpenAI()
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
    )
    return response.choices[0].message.content


def compare_raw_vs_context(raw_csv_text: str, context: str) -> None:
    """Send the raw CSV and the structured context to the LLM separately
    and print both 3-bullet summaries so students can compare quality."""
    question = "Summarise this portfolio in exactly 3 bullet points."

    print("\n=== LLM summary from RAW CSV ===")
    try:
        raw_answer = ask_llm(f"{question}\n\n{raw_csv_text}")
        print(raw_answer)
    except Exception as exc:  # pragma: no cover - depends on network/API key
        print(f"[skipped] could not call LLM with raw CSV: {exc}")

    print("\n=== LLM summary from STRUCTURED CONTEXT ===")
    try:
        context_answer = ask_llm(f"{question}\n\n{context}")
        print(context_answer)
    except Exception as exc:  # pragma: no cover - depends on network/API key
        print(f"[skipped] could not call LLM with structured context: {exc}")


# %% [8] Main ----------------------------------------------------------------------
def main():
    # Step 1: load
    raw_df = load_portfolio()
    print(f"Loaded {len(raw_df)} raw rows from {CSV_PATH.name}")

    # Step 2: clean
    clean_df = clean_portfolio(raw_df)
    print(f"Cleaned to {len(clean_df)} rows, dtypes:\n{clean_df.dtypes}\n")

    # Step 3: enrich
    enriched_df = enrich_portfolio(clean_df)

    # Step 4: aggregate
    portfolio_summary = aggregate_portfolio(enriched_df)
    print("portfolio_summary JSON:")
    print(json.dumps(portfolio_summary, indent=2, default=str))

    # Step 5: format
    context = portfolio_to_context(enriched_df, portfolio_summary)
    print("\nContext string preview (first 500 chars):")
    print(context[:500])

    # Step 6: test against GPT-4o-mini (skips gracefully without an API key)
    if os.getenv("OPENAI_API_KEY"):
        raw_csv_text = raw_df.to_csv(index=False)
        compare_raw_vs_context(raw_csv_text, context)
    else:
        print(
            "\n[info] Set OPENAI_API_KEY to run Step 6 "
            "(raw CSV vs structured context comparison)."
        )


if __name__ == "__main__":
    main()
