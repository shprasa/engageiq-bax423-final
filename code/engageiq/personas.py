"""Built-in and session custom user profiles for EngageIQ."""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any

PROFILE_FIELDS: tuple[str, ...] = (
    "background",
    "interests",
    "goal",
    "platforms",
    "time_budget",
)

PROFILE_FIELD_LABELS: dict[str, str] = {
    "background": "Background",
    "interests": "Interests",
    "goal": "Goal",
    "platforms": "Platforms",
    "time_budget": "Time budget",
}


@dataclass(frozen=True)
class UserProfile:
    background: str
    interests: str
    goal: str
    platforms: str
    time_budget: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> UserProfile:
        return cls(
            background=str(data.get("background", "") or ""),
            interests=str(data.get("interests", "") or ""),
            goal=str(data.get("goal", "") or ""),
            platforms=str(data.get("platforms", "") or ""),
            time_budget=str(data.get("time_budget", "") or ""),
        )


def profile_to_interest_text(profile: UserProfile | dict[str, str]) -> str:
    """Flatten structured profile into one retrieval query string."""
    if isinstance(profile, UserProfile):
        data = profile.to_dict()
    else:
        data = profile
    parts: list[str] = []
    for key in PROFILE_FIELDS:
        value = str(data.get(key, "") or "").strip()
        if value:
            label = PROFILE_FIELD_LABELS[key]
            parts.append(f"{label}: {value}")
    return "\n".join(parts)


# Wording matches BAX423 Final Project instructions (Test Personas tables).
BUILTIN_PROFILES: dict[str, UserProfile] = {
    "Sofia (ML Student / Portfolio Builder)": UserProfile(
        background=(
            "MSBA student, graduating soon. Wants to build a visible open-source portfolio before job hunting."
        ),
        interests="Machine learning, NLP, data pipelines. Comfortable with Python and pandas.",
        goal=(
            "Find beginner-friendly GitHub repos to contribute to and Reddit/blog discussions "
            "to engage with for visibility."
        ),
        platforms="GitHub (primary), Reddit r/MachineLearning, ML blogs.",
        time_budget="5 hours/week.",
    ),
    "David (DevOps / Niche Community)": UserProfile(
        background=(
            "Mid-career DevOps engineer (5 years). Wants to establish thought leadership in cloud-native infrastructure."
        ),
        interests="Kubernetes, Terraform, CI/CD, observability. Reads infrastructure blogs heavily.",
        goal=(
            "Find high-signal GitHub projects, Reddit threads, and blog posts where expert commentary adds value."
        ),
        platforms="GitHub (Kubernetes ecosystem), Reddit r/devops and r/kubernetes, DevOps blogs.",
        time_budget="3 hours/week.",
    ),
    "Lina (Data Journalist / Trend Spotter)": UserProfile(
        background=(
            "Data journalist at a tech publication. Monitors open-source and tech communities for story leads."
        ),
        interests="Trending repos, viral discussions, emerging tools, community drama. Breadth over depth.",
        goal=(
            "Surface fast-growing repos, trending Reddit threads, and blog posts gaining traction, "
            "before they go mainstream."
        ),
        platforms="GitHub trending, Reddit (multiple subreddits), Hacker News-style blogs.",
        time_budget="10 hours/week (this is her job).",
    ),
    "Raj (Startup Founder / Marketing-Focused)": UserProfile(
        background=(
            "Technical co-founder of a developer tools startup. Wants to grow awareness by engaging in relevant communities."
        ),
        interests="Developer productivity, APIs, CLI tools, open-source business models.",
        goal=(
            "Find Reddit threads and blog posts where his product is relevant to the discussion, "
            "and GitHub repos where integration/partnership makes sense."
        ),
        platforms="Reddit r/programming, r/SideProject, r/startups. GitHub (developer tools). Tech blogs.",
        time_budget="4 hours/week.",
    ),
}

BUILTIN_PERSONAS: dict[str, str] = {
    name: profile_to_interest_text(profile) for name, profile in BUILTIN_PROFILES.items()
}

CUSTOM_SENTINEL = "— Custom (edit profile below) —"

PROFILE_WIDGET_KEYS: dict[str, str] = {
    "background": "profile_background",
    "interests": "profile_interests",
    "goal": "profile_goal",
    "platforms": "profile_platforms",
    "time_budget": "profile_time_budget",
}


def read_profile_from_session(session_state: Any) -> UserProfile:
    return UserProfile.from_dict(
        {field: str(session_state.get(PROFILE_WIDGET_KEYS[field], "") or "") for field in PROFILE_FIELDS}
    )


def apply_profile_to_session(session_state: Any, profile: UserProfile | dict[str, str]) -> None:
    data = profile.to_dict() if isinstance(profile, UserProfile) else profile
    for field in PROFILE_FIELDS:
        session_state[PROFILE_WIDGET_KEYS[field]] = str(data.get(field, "") or "")
