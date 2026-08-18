"""Mock HR data and policy knowledge base.

Everything here is static and deterministic on purpose: the agent's tools read
from these dicts so that, given the same inputs, the agent produces the same
tool outputs every run. Deterministic tools make evaluations reproducible —
you're measuring the *model's* behavior, not flaky downstream systems.
"""

from __future__ import annotations

# --- Employee directory --------------------------------------------------

# Keyed by lowercase full name for simple lookups.
EMPLOYEES: dict[str, dict] = {
    "jordan lee": {
        "employee_id": "E1007",
        "full_name": "Jordan Lee",
        "title": "Software Engineer",
        "department": "Engineering",
        "start_date": "2026-06-15",
        "manager": "Priya Anand",
        "benefits_plan": "standard",
    },
    "sam rivera": {
        "employee_id": "E1008",
        "full_name": "Sam Rivera",
        "title": "Product Designer",
        "department": "Design",
        "start_date": "2026-06-22",
        "manager": "Dana Kim",
        "benefits_plan": "standard",
    },
    "alex chen": {
        "employee_id": "E1009",
        "full_name": "Alex Chen",
        "title": "Engineering Manager",
        "department": "Engineering",
        "start_date": "2026-07-01",
        "manager": "Morgan Doyle",
        "benefits_plan": "executive",
    },
}


# --- HR policy knowledge base -------------------------------------------

# Keyed by topic. These are the "ground truth" facts the agent should cite
# when answering policy questions. Single-turn evals check the agent's answer
# against these.
HR_POLICIES: dict[str, str] = {
    "vacation": (
        "New full-time employees accrue 15 paid vacation days (PTO) per year, "
        "accruing at 1.25 days per month. PTO begins accruing on the employee's "
        "start date and can be used after the 90-day probationary period."
    ),
    "sick_leave": (
        "Employees receive 10 paid sick days per calendar year. Sick days do "
        "not roll over to the following year and cannot be cashed out."
    ),
    "remote_work": (
        "The company follows a hybrid policy: employees are expected on-site "
        "Tuesday, Wednesday, and Thursday, and may work remotely on Monday and "
        "Friday. Fully-remote arrangements require VP approval."
    ),
    "health_insurance": (
        "Health, dental, and vision coverage begin on the first day of the "
        "month following the start date. Employees have 30 days from their "
        "start date to enroll or make changes."
    ),
    "401k": (
        "The company offers a 401(k) with a 4% match. Employees are eligible "
        "to enroll after 60 days of employment. The company match vests "
        "immediately."
    ),
    "parental_leave": (
        "Parental leave provides 12 weeks of paid leave for the primary "
        "caregiver and 6 weeks for the secondary caregiver, available after "
        "6 months of employment."
    ),
    "expenses": (
        "Business expenses must be submitted within 30 days via the expense "
        "portal. Receipts are required for any expense over $25. Reimbursement "
        "is issued in the next payroll cycle after approval."
    ),
}


# --- Benefits plans ------------------------------------------------------

BENEFITS_PLANS: dict[str, dict] = {
    "standard": {
        "health": "PPO medical, dental, vision",
        "401k_match": "4%",
        "pto_days": 15,
        "equity": "RSU grant per offer letter",
    },
    "executive": {
        "health": "PPO medical, dental, vision + executive health program",
        "401k_match": "6%",
        "pto_days": 20,
        "equity": "RSU grant + annual refresh",
    },
}


# Systems an IT account can be created for. Used to validate tool arguments.
VALID_IT_SYSTEMS = {"email", "slack", "github", "vpn", "hris"}

# Equipment the company provisions.
VALID_EQUIPMENT = {"laptop", "monitor", "keyboard", "headset", "phone"}


# --- Topic resolution ----------------------------------------------------

# Keyword -> policy topic. Used to pick the right policy for a free-text
# question when the caller doesn't already know the topic (see
# module_2_single_turn/structured_output.py). Deliberately tiny and
# deterministic: this is a lookup table, not retrieval.
_TOPIC_KEYWORDS: dict[str, tuple[str, ...]] = {
    "vacation": ("vacation", "pto", "paid time off", "time off", "holiday"),
    "sick_leave": ("sick", "illness", "unwell"),
    "remote_work": ("remote", "wfh", "work from home", "hybrid", "on-site", "onsite", "office"),
    "health_insurance": ("health insurance", "health", "medical", "dental", "vision", "coverage", "enroll"),
    "401k": ("401k", "401(k)", "retirement", "match", "vest"),
    "parental_leave": ("parental", "maternity", "paternity", "caregiver", "baby"),
    "expenses": ("expense", "reimburse", "receipt", "spend"),
}


def resolve_topic(text: str) -> str | None:
    """Best-effort map free text to one HR policy topic key, else None.

    Returns a key that is guaranteed to be in ``HR_POLICIES`` — never
    caller-supplied text — so the result is always safe to use as a lookup key.
    """
    lowered = (text or "").lower()
    best: tuple[int, str] | None = None
    for topic, keywords in _TOPIC_KEYWORDS.items():
        for kw in keywords:
            if kw in lowered:
                # Longest matching keyword wins ("health insurance" > "health").
                if best is None or len(kw) > best[0]:
                    best = (len(kw), topic)
    if best is None:
        return None
    topic = best[1]
    return topic if topic in HR_POLICIES else None


def policy_corpus() -> str:
    """The whole policy corpus as one labeled block (fallback context)."""
    return "\n".join(f"[{topic}] {text}" for topic, text in HR_POLICIES.items())
