from __future__ import annotations

# All 15 required technical domains from the EngageIQ brief.
DOMAINS: list[str] = [
    "Machine Learning",
    "DevOps/K8s",
    "Trending Open-Source",
    "Developer Tools",
    "Cybersecurity",
    "Frontend (React/Web)",
    "B2B SaaS",
    "Blockchain",
    "Python Data Eng",
    "GameDev (C++)",
    "AI Research",
    "Embedded Systems (C/RTOS)",
    "Cloud APIs",
    "Mobile Dev (iOS/Flutter)",
    "Beginner Coding",
]

# Search keywords per domain for GitHub and Hacker News scrapers.
DOMAIN_QUERIES: dict[str, dict[str, list[str]]] = {
    "Machine Learning": {
        "github": ["machine-learning", "pytorch", "scikit-learn"],
        "hn": ["machine learning", "pytorch", "llm"],
    },
    "DevOps/K8s": {
        "github": ["kubernetes", "terraform", "devops"],
        "hn": ["kubernetes", "terraform", "devops"],
    },
    "Trending Open-Source": {
        "github": ["open-source", "trending"],
        "hn": ["open source", "github"],
    },
    "Developer Tools": {
        "github": ["developer-tools", "cli", "vscode"],
        "hn": ["developer tools", "cli"],
    },
    "Cybersecurity": {
        "github": ["security", "cybersecurity"],
        "hn": ["security", "cve"],
    },
    "Frontend (React/Web)": {
        "github": ["react", "frontend", "nextjs"],
        "hn": ["react", "frontend"],
    },
    "B2B SaaS": {
        "github": ["saas", "b2b"],
        "hn": ["saas", "b2b"],
    },
    "Blockchain": {
        "github": ["blockchain", "ethereum"],
        "hn": ["blockchain", "ethereum"],
    },
    "Python Data Eng": {
        "github": ["data-engineering", "apache-spark", "airflow"],
        "hn": ["data engineering", "spark"],
    },
    "GameDev (C++)": {
        "github": ["gamedev", "unreal-engine"],
        "hn": ["gamedev", "unity"],
    },
    "AI Research": {
        "github": ["deep-learning", "transformers", "llm"],
        "hn": ["ai research", "transformer"],
    },
    "Embedded Systems (C/RTOS)": {
        "github": ["embedded", "rtos", "arduino"],
        "hn": ["embedded", "rtos"],
    },
    "Cloud APIs": {
        "github": ["cloud-api", "aws", "serverless"],
        "hn": ["aws", "cloud api"],
    },
    "Mobile Dev (iOS/Flutter)": {
        "github": ["flutter", "ios", "swiftui"],
        "hn": ["flutter", "ios"],
    },
    "Beginner Coding": {
        "github": ["good-first-issue", "beginner-friendly"],
        "hn": ["learn programming", "beginner"],
    },
}
