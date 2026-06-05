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

# Search keywords / subreddits per domain for scrapers.
DOMAIN_QUERIES: dict[str, dict[str, list[str]]] = {
    "Machine Learning": {
        "github": ["machine-learning", "pytorch", "scikit-learn"],
        "reddit": ["MachineLearning", "learnmachinelearning"],
        "hn": ["machine learning", "pytorch", "llm"],
    },
    "DevOps/K8s": {
        "github": ["kubernetes", "terraform", "devops"],
        "reddit": ["devops", "kubernetes"],
        "hn": ["kubernetes", "terraform", "devops"],
    },
    "Trending Open-Source": {
        "github": ["open-source", "trending"],
        "reddit": ["opensource", "coolgithubprojects"],
        "hn": ["open source", "github"],
    },
    "Developer Tools": {
        "github": ["developer-tools", "cli", "vscode"],
        "reddit": ["programming", "commandline"],
        "hn": ["developer tools", "cli"],
    },
    "Cybersecurity": {
        "github": ["security", "cybersecurity"],
        "reddit": ["cybersecurity", "netsec"],
        "hn": ["security", "cve"],
    },
    "Frontend (React/Web)": {
        "github": ["react", "frontend", "nextjs"],
        "reddit": ["reactjs", "webdev"],
        "hn": ["react", "frontend"],
    },
    "B2B SaaS": {
        "github": ["saas", "b2b"],
        "reddit": ["SaaS", "startups"],
        "hn": ["saas", "b2b"],
    },
    "Blockchain": {
        "github": ["blockchain", "ethereum"],
        "reddit": ["CryptoCurrency", "ethdev"],
        "hn": ["blockchain", "ethereum"],
    },
    "Python Data Eng": {
        "github": ["data-engineering", "apache-spark", "airflow"],
        "reddit": ["dataengineering", "Python"],
        "hn": ["data engineering", "spark"],
    },
    "GameDev (C++)": {
        "github": ["gamedev", "unreal-engine"],
        "reddit": ["gamedev", "unrealengine"],
        "hn": ["gamedev", "unity"],
    },
    "AI Research": {
        "github": ["deep-learning", "transformers", "llm"],
        "reddit": ["MachineLearning", "LocalLLaMA"],
        "hn": ["ai research", "transformer"],
    },
    "Embedded Systems (C/RTOS)": {
        "github": ["embedded", "rtos", "arduino"],
        "reddit": ["embedded", "arduino"],
        "hn": ["embedded", "rtos"],
    },
    "Cloud APIs": {
        "github": ["cloud-api", "aws", "serverless"],
        "reddit": ["aws", "googlecloud"],
        "hn": ["aws", "cloud api"],
    },
    "Mobile Dev (iOS/Flutter)": {
        "github": ["flutter", "ios", "swiftui"],
        "reddit": ["FlutterDev", "iOSProgramming"],
        "hn": ["flutter", "ios"],
    },
    "Beginner Coding": {
        "github": ["good-first-issue", "beginner-friendly"],
        "reddit": ["learnprogramming", "coding"],
        "hn": ["learn programming", "beginner"],
    },
}
