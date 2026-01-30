"""Feature engineering pipeline for encounter-duration prediction."""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

# --- CRITICAL FIX: Add project root to system path ---
# This allows the script to find the 'src' module when run directly.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

# Fixed Import: Use absolute import instead of relative
from src.config import PipelineConfig, default_config

PROCESSED_FILENAME = "patient_journey_log.csv"
FEATURES_FILENAME = "encounter_features.parquet"


class FeaturePipelinePort:
    """Interface for feature engineering pipelines."""

    def build_features(self):
        raise NotImplementedError


def _load_events(config):
    """
    Loads the processed event log created by the ingestion step.
    
    Args:
        config: The pipeline configuration object.
    Returns:
        pd.DataFrame: The cleaned event log.
    """
    processed_path = config.processed_data_dir / PROCESSED_FILENAME
    
    if not processed_path.exists():
        raise FileNotFoundError(
            f"❌ Processed log not found at: {processed_path}. "
            "Please run 'src/ingest_data.py' first."
        )

    print(f"   Loading events from {processed_path.name}...")
    df = pd.read_csv(processed_path)
    
    # Ensure timestamps are correctly parsed
    df["start:timestamp"] = pd.to_datetime(df["start:timestamp"], utc=True, errors="coerce")
    df["end:timestamp"] = pd.to_datetime(df["end:timestamp"], utc=True, errors="coerce")
    
    # Remove rows with missing critical information
    df = df.dropna(subset=["case:concept:name", "concept:name", "start:timestamp", "end:timestamp"])
    return df


def _summarize_encounter_duration(events):
    """Calculates the total duration and start/end times for each patient encounter."""
    
    summary = (
        events.groupby("case:concept:name")
        .agg(
            encounter_start=("start:timestamp", "min"),
            encounter_end=("end:timestamp", "max"),
            event_count=("concept:name", "count"),
        )
    )
    
    # Calculate duration in minutes
    summary["encounter_duration_minutes"] = (
        (summary["encounter_end"] - summary["encounter_start"]).dt.total_seconds().div(60)
    )
    
    # Filter out invalid durations (negative or zero)
    summary = summary[summary["encounter_duration_minutes"] > 0]
    summary["total_hours"] = summary["encounter_duration_minutes"] / 60.0
    
    return summary


def _procedure_matrix(events, top_k=50):
    """
    Creates a matrix of procedures performed during each encounter.
    """
    events["concept:name"] = events["concept:name"].astype(str).str.strip()
    
    # Select only the most common procedures to avoid noise
    top_procedures = events["concept:name"].value_counts().nlargest(top_k).index
    filtered = events[events["concept:name"].isin(top_procedures)]
    
    # Pivot table: Rows = Patients, Columns = Procedures
    matrix = filtered.groupby(["case:concept:name", "concept:name"]).size().unstack(fill_value=0)
    
    # Rename columns to be model-friendly
    matrix.columns = [f"proc_count__{col}" for col in matrix.columns]
    
    return matrix


def _save_features(config, features):
    """Saves the final feature set to the disk."""
    
    output_path = config.feature_store_dir / FEATURES_FILENAME
    os.makedirs(output_path.parent, exist_ok=True)
    features.to_parquet(output_path)
    return output_path


@dataclass
class DefaultFeaturePipeline(FeaturePipelinePort):
    """
    Main pipeline class that orchestrates the feature engineering process.
    """
    config: PipelineConfig

    def build_features(self):
        print("Starting Feature Engineering...")
        events = _load_events(self.config)
        durations = _summarize_encounter_duration(events)
        procedures = _procedure_matrix(events)
        features = durations.join(procedures, how="left")
        features.fillna(0, inplace=True)
        output_path = _save_features(self.config, features)
        print(f"Features successfully saved at: {output_path}")

# --- EXECUTION BLOCK ---
if __name__ == "__main__":
    cfg = default_config(PROJECT_ROOT)
    pipeline = DefaultFeaturePipeline(cfg)
    pipeline.build_features()