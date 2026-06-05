from __future__ import annotations

import os
import time

import certifi
import requests

try:
    import truststore

    truststore.inject_into_ssl()
except ImportError:
    pass


def get(url: str, **kwargs) -> requests.Response:
    kwargs.setdefault("timeout", 30)
    if os.getenv("ENGAGEIQ_SSL_VERIFY", "true").lower() in {"0", "false", "no"}:
        kwargs["verify"] = False
    else:
        kwargs.setdefault("verify", certifi.where())

    last_err: Exception | None = None
    for attempt in range(5):
        try:
            r = requests.get(url, **kwargs)
            if r.status_code in {403, 429, 502, 503, 504}:
                time.sleep(min(60, 2 ** attempt * 3))
                continue
            return r
        except requests.RequestException as e:
            last_err = e
            time.sleep(min(30, 2 ** attempt * 2))
    if last_err:
        raise last_err
    raise RuntimeError(f"Failed to GET {url}")
