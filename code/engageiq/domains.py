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

# Search keywords per domain for GitHub API and GitHub Archive scrapers.
DOMAIN_QUERIES: dict[str, dict[str, list[str]]] = {
    "Machine Learning": {
        "github": ["machine-learning", "pytorch", "scikit-learn"],
        "gharchive": ["machine-learning", "pytorch", "llm", "ml"],
    },
    "DevOps/K8s": {
        "github": ["kubernetes", "terraform", "devops"],
        "gharchive": ["kubernetes", "terraform", "devops", "k8s"],
    },
    "Trending Open-Source": {
        "github": ["open-source", "trending"],
        "gharchive": ["open-source", "trending", "github"],
    },
    "Developer Tools": {
        "github": ["developer-tools", "cli", "vscode"],
        "gharchive": ["developer-tools", "cli", "vscode"],
    },
    "Cybersecurity": {
        "github": ["security", "cybersecurity"],
        "gharchive": ["security", "cve", "vulnerability"],
    },
    "Frontend (React/Web)": {
        "github": ["react", "frontend", "nextjs"],
        "gharchive": ["react", "frontend", "nextjs"],
    },
    "B2B SaaS": {
        "github": ["saas", "b2b"],
        "gharchive": ["saas", "b2b"],
    },
    "Blockchain": {
        "github": ["blockchain", "ethereum"],
        "gharchive": ["blockchain", "ethereum", "web3"],
    },
    "Python Data Eng": {
        "github": ["data-engineering", "apache-spark", "airflow"],
        "gharchive": ["data-engineering", "spark", "airflow"],
    },
    "GameDev (C++)": {
        "github": ["gamedev", "unreal-engine"],
        "gharchive": ["gamedev", "unity", "unreal"],
    },
    "AI Research": {
        "github": ["deep-learning", "transformers", "llm"],
        "gharchive": ["transformer", "llm", "research"],
    },
    "Embedded Systems (C/RTOS)": {
        "github": ["embedded", "rtos", "arduino"],
        "gharchive": ["embedded", "rtos", "arduino"],
    },
    "Cloud APIs": {
        "github": ["cloud-api", "aws", "serverless"],
        "gharchive": ["aws", "cloud", "serverless"],
    },
    "Mobile Dev (iOS/Flutter)": {
        "github": ["flutter", "ios", "swiftui"],
        "gharchive": ["flutter", "ios", "swift"],
    },
    "Beginner Coding": {
        "github": ["good-first-issue", "beginner-friendly"],
        "gharchive": ["good-first-issue", "beginner", "first issue"],
    },
}
