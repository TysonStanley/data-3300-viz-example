"""
Simulated datasets for the DATA 3300 Data Visuals site
(https://tysonbarrett.com/data-3300-viz-example/).

Every dataset has a clear pattern baked in, and every dataset is shuffled
so the pattern is NOT visible from row order. That is the whole point:
the spreadsheet hides it, the chart reveals it.
"""
import numpy as np
import pandas as pd

SEED = 2026


def wait_times(n=240, seed=SEED):
    """Customer support wait times (minutes). Hidden pattern: bimodal.
    Chat customers wait ~4 min, phone customers wait ~17 min."""
    rng = np.random.default_rng(seed)
    n_chat = int(n * 0.55)
    chat = rng.gamma(shape=6, scale=0.7, size=n_chat)
    phone = rng.normal(loc=17, scale=2.6, size=n - n_chat)
    minutes = np.clip(np.concatenate([chat, phone]), 0.5, None).round(1)
    ids = rng.permutation(np.arange(10001, 10001 + n))
    df = pd.DataFrame({"ticket_id": ids, "wait_minutes": minutes})
    return df.sample(frac=1, random_state=seed).reset_index(drop=True)


def ticket_categories(n=200, seed=SEED):
    """Support tickets by category. Hidden pattern: billing is ~45% of all tickets."""
    rng = np.random.default_rng(seed + 1)
    cats = ["Billing", "Login / Password", "Shipping", "Product defect", "Other"]
    probs = [0.45, 0.20, 0.17, 0.11, 0.07]
    df = pd.DataFrame({
        "ticket_id": np.arange(20001, 20001 + n),
        "date": pd.Timestamp("2026-09-01") + pd.to_timedelta(rng.integers(0, 30, n), unit="D"),
        "category": rng.choice(cats, size=n, p=probs),
        "priority": rng.choice(["Low", "Medium", "High"], size=n, p=[0.4, 0.4, 0.2]),
    })
    return df.sort_values("date").reset_index(drop=True)


def store_sales(seed=SEED):
    """Daily sales ($) for 4 regions. Hidden pattern: averages look similar,
    but the West region is wildly inconsistent and the South has extreme outliers."""
    rng = np.random.default_rng(seed + 2)
    n = 60
    frames = [
        pd.DataFrame({"region": "North", "daily_sales": rng.normal(5200, 350, n)}),
        pd.DataFrame({"region": "East",  "daily_sales": rng.normal(5500, 400, n)}),
        pd.DataFrame({"region": "West",  "daily_sales": rng.normal(5300, 1600, n)}),
        pd.DataFrame({"region": "South", "daily_sales": np.concatenate([
            rng.normal(4700, 300, n - 5), rng.normal(10500, 800, 5)])}),
    ]
    df = pd.concat(frames, ignore_index=True)
    df["daily_sales"] = df["daily_sales"].clip(lower=800).round(0).astype(int)
    df["day"] = np.tile(np.arange(1, n + 1), 4)
    df = df[["day", "region", "daily_sales"]]
    return df.sample(frac=1, random_state=seed).reset_index(drop=True)


def study_scores(n=150, seed=SEED):
    """Hours studied vs exam score. Hidden patterns: strong positive but
    curved (diminishing returns) relationship, plus a small cluster of
    students who studied a lot but scored poorly (they skipped the review session)."""
    rng = np.random.default_rng(seed + 3)
    hours = rng.uniform(0, 20, n)
    score = 45 + 50 * (1 - np.exp(-hours / 6)) + rng.normal(0, 4, n)
    attended = np.ones(n, dtype=bool)
    odd = rng.choice(np.where(hours > 11)[0], size=10, replace=False)
    score[odd] -= rng.uniform(25, 35, size=10)
    attended[odd] = False
    df = pd.DataFrame({
        "student_id": np.arange(1, n + 1),
        "hours_studied": hours.round(1),
        "exam_score": np.clip(score, 0, 100).round(0).astype(int),
        "attended_review": np.where(attended, "Yes", "No"),
    })
    return df.sample(frac=1, random_state=seed).reset_index(drop=True)


def monthly_visits(seed=SEED):
    """Monthly website visits for 3 product lines over 3 years. Hidden patterns:
    - Outdoor gear: strong summer seasonality
    - Smart home: steady growth
    - Office supplies: flat, then drops sharply after a site redesign (Jan 2025)"""
    rng = np.random.default_rng(seed + 4)
    months = pd.date_range("2023-01-01", "2025-12-01", freq="MS")
    t = np.arange(len(months))
    outdoor = 9000 + 4500 * np.sin(2 * np.pi * (months.month - 4) / 12) + rng.normal(0, 500, len(t))
    smart = 4000 + 230 * t + rng.normal(0, 450, len(t))
    office = np.where(months < "2025-01-01", 11000, 6500) + rng.normal(0, 500, len(t))
    df = pd.DataFrame({
        "month": np.tile(months, 3),
        "product_line": np.repeat(["Outdoor gear", "Smart home", "Office supplies"], len(t)),
        "visits": np.concatenate([outdoor, smart, office]).round(0).astype(int),
    })
    # Wide format, like most real spreadsheets people get emailed
    wide = df.pivot(index="month", columns="product_line", values="visits").reset_index()
    wide.columns.name = None
    wide["month"] = wide["month"].dt.strftime("%Y-%m")
    return df, wide
