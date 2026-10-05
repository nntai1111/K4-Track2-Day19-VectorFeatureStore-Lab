"""Feast definitions for the bonus memory agent (stable profile + recent activity).

Self-contained: builds its own registry + SQLite online store in a scratch
directory, so the demo never touches the NB4 repo in `app/feast_repo`.

Two feature views, two freshness tiers (see ARCHITECTURE.md, decision 3):
  user_profile    ttl=30d  written by a daily batch job (here: seed_profiles)
  recent_activity ttl=1h   pushed on every query (streaming path, sub-second)
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
from feast import Entity, FeatureStore, FeatureView, Field, FileSource, ValueType
from feast.infra.online_stores.sqlite import SqliteOnlineStoreConfig
from feast.repo_config import RepoConfig
from feast.types import Float32, Int64, String

PROFILE_FEATURES = [
    "user_profile:preferred_language",
    "user_profile:reading_speed_wpm",
    "user_profile:topic_affinity",
    "user_profile:active_hours",
]
ACTIVITY_FEATURES = [
    "recent_activity:queries_last_hour",
    "recent_activity:recent_topics",
    "recent_activity:late_night_ratio",
]


def build_store(workdir: Path) -> FeatureStore:
    workdir.mkdir(parents=True, exist_ok=True)
    data = workdir / "data"
    data.mkdir(exist_ok=True)
    config = RepoConfig(
        project="bonus_memory",
        provider="local",
        registry=str(workdir / "registry.db"),
        online_store=SqliteOnlineStoreConfig(path=str(workdir / "online_store.db")),
        entity_key_serialization_version=3,
    )
    fs = FeatureStore(config=config)

    user = Entity(name="user", join_keys=["user_id"], value_type=ValueType.STRING)
    profile = FeatureView(
        name="user_profile",
        entities=[user],
        ttl=timedelta(days=30),
        schema=[
            Field(name="preferred_language", dtype=String),   # vi | en | mix
            Field(name="reading_speed_wpm", dtype=Int64),
            Field(name="topic_affinity", dtype=String),
            Field(name="active_hours", dtype=String),          # e.g. "20-23"
        ],
        source=FileSource(path=str(data / "user_profile.parquet"),
                          timestamp_field="event_timestamp"),
        online=True,
    )
    activity = FeatureView(
        name="recent_activity",
        entities=[user],
        ttl=timedelta(hours=1),
        schema=[
            Field(name="queries_last_hour", dtype=Int64),
            Field(name="recent_topics", dtype=String),         # comma-joined, most recent first
            Field(name="late_night_ratio", dtype=Float32),     # share of queries 22h-5h
        ],
        source=FileSource(path=str(data / "recent_activity.parquet"),
                          timestamp_field="event_timestamp"),
        online=True,
    )
    fs.apply([user, profile, activity])
    return fs


def seed_profiles(fs: FeatureStore, profiles: list[dict]) -> None:
    """Stand-in for the daily batch job (offline compute -> online store)."""
    now = datetime.now(timezone.utc)
    df = pd.DataFrame([{**p, "event_timestamp": now} for p in profiles])
    fs.write_to_online_store(feature_view_name="user_profile", df=df)


def push_activity(fs: FeatureStore, user_id: str, queries_last_hour: int,
                  recent_topics: list[str], late_night_ratio: float) -> None:
    """Streaming path: one row per query event, visible to the next read."""
    df = pd.DataFrame([{
        "user_id": user_id,
        "queries_last_hour": queries_last_hour,
        "recent_topics": ",".join(recent_topics),
        "late_night_ratio": float(late_night_ratio),
        "event_timestamp": datetime.now(timezone.utc),
    }])
    fs.write_to_online_store(feature_view_name="recent_activity", df=df)


def read_features(fs: FeatureStore, user_id: str) -> dict:
    out = fs.get_online_features(features=PROFILE_FEATURES + ACTIVITY_FEATURES,
                                 entity_rows=[{"user_id": user_id}]).to_dict()
    return {k: v[0] for k, v in out.items()}
