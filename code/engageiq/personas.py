"""Built-in and session custom user profiles for EngageIQ."""
from __future__ import annotations

BUILTIN_PERSONAS: dict[str, str] = {
    "Sofia (ML Student / Portfolio Builder)": (
        "Machine learning, NLP, data pipelines, beginner-friendly open source, good first issues, "
        "Python, pandas, GitHub issues, GitHub Archive ML issue events."
    ),
    "David (DevOps / Niche Community)": (
        "Kubernetes, Terraform, CI/CD, observability, cloud-native infra, high-activity repos, "
        "few contributors, GitHub Archive DevOps issue and PR events."
    ),
    "Lina (Data Journalist / Trend Spotter)": (
        "Trending repos, viral discussions, emerging tools, fast-growing communities, recency, velocity, "
        "GitHub Archive public timeline events, GitHub trending, multi-domain velocity."
    ),
    "Raj (Startup Founder / Marketing-Focused)": (
        "Developer tools, APIs, CLI tools, open-source business, B2B SaaS, discussions where devtools are relevant, "
        "GitHub Archive and GitHub developer-tools communities."
    ),
}

CUSTOM_SENTINEL = "— Custom (edit interests below) —"
