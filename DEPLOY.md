# Deploy EngageIQ to Streamlit Community Cloud

## 1) Push to GitHub

Repo: https://github.com/shprasa/engageiq-bax423-final

## 2) Create Streamlit app

1. Go to https://share.streamlit.io
2. **New app** → connect GitHub → select `engageiq-bax423-final`
3. **Main file path:** `code/app.py`
4. **Requirements file:** `code/requirements.txt`

## 3) Secrets (optional live API refresh)

In Streamlit **Settings → Secrets**, paste:

```toml
GITHUB_TOKEN = "your_token"
```

Reddit keys optional.

## 4) Verify

- App loads ranked opportunities
- Persona selector works
- Export CSV/PDF works
- Trend charts render

## Local run

```bat
cd code
py -m pip install -r requirements.txt
py -m streamlit run app.py
```

## Submission URL

After deploy, confirm public URL in `brief.pdf` and Canvas submission.
