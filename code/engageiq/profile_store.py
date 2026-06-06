"""Persist custom user profiles to disk."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .config import Paths, get_paths
from .personas import BUILTIN_PROFILES, UserProfile, profile_to_interest_text


def custom_profiles_path(paths: Paths | None = None) -> Path:
    paths = paths or get_paths()
    if paths.is_cloud:
        return Path("/tmp/engageiq_custom_personas.json")
    return paths.data_dir / "custom_personas.json"


def validate_profile(profile: UserProfile) -> list[str]:
    from .personas import PROFILE_FIELD_LABELS, PROFILE_FIELDS

    errors: list[str] = []
    for field in PROFILE_FIELDS:
        if not str(getattr(profile, field, "") or "").strip():
            errors.append(PROFILE_FIELD_LABELS[field])
    return errors


def load_custom_profiles(path: Path | None = None) -> dict[str, UserProfile]:
    path = path or custom_profiles_path()
    if not path.exists():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}
    if not isinstance(raw, dict):
        return {}
    out: dict[str, UserProfile] = {}
    for name, data in raw.items():
        if name in BUILTIN_PROFILES:
            continue
        if isinstance(data, dict):
            out[str(name)] = UserProfile.from_dict(data)
        elif isinstance(data, str) and data.strip():
            out[str(name)] = UserProfile(
                background="",
                interests=data.strip(),
                goal="",
                platforms="",
                time_budget="",
            )
    return out


def save_custom_profiles(profiles: dict[str, UserProfile], path: Path | None = None) -> Path:
    paths = get_paths()
    path = path or custom_profiles_path(paths)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {name: prof.to_dict() for name, prof in profiles.items() if name not in BUILTIN_PROFILES}
    text = json.dumps(payload, indent=2)
    path.write_text(text, encoding="utf-8")
    if not paths.is_cloud:
        alt = paths.project_root / "data" / "custom_personas.json"
        if alt.resolve() != path.resolve():
            alt.parent.mkdir(parents=True, exist_ok=True)
            alt.write_text(text, encoding="utf-8")
    return path


def upsert_custom_profile(name: str, profile: UserProfile, path: Path | None = None) -> Path:
    name = name.strip()
    if not name or name in BUILTIN_PROFILES:
        raise ValueError("Invalid custom profile name")
    errors = validate_profile(profile)
    if errors:
        raise ValueError(f"Complete all profile fields: {', '.join(errors)}")
    profiles = load_custom_profiles(path)
    profiles[name] = profile
    return save_custom_profiles(profiles, path)


def delete_custom_profile(name: str, path: Path | None = None) -> Path:
    profiles = load_custom_profiles(path)
    profiles.pop(name, None)
    return save_custom_profiles(profiles, path)


def all_persona_interest_texts(custom_path: Path | None = None) -> dict[str, str]:
    """Built-in + saved custom personas as flattened retrieval queries."""
    texts = {name: profile_to_interest_text(prof) for name, prof in BUILTIN_PROFILES.items()}
    for name, prof in load_custom_profiles(custom_path).items():
        texts[name] = profile_to_interest_text(prof)
    return texts


def profiles_for_export() -> dict[str, Any]:
    custom = load_custom_profiles()
    return {
        "builtin": {name: prof.to_dict() for name, prof in BUILTIN_PROFILES.items()},
        "custom": {name: prof.to_dict() for name, prof in custom.items()},
    }
