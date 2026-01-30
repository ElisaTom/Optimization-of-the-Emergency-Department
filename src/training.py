"""Model training module outline with boilerplate functions."""

import sys
from dataclasses import dataclass
from pathlib import Path

# --- SYSTEM SETUP (Critical for standalone execution) ---
# We add the project root to the system path to fix import errors
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# Use absolute import instead of relative (.config)
from src.config import PipelineConfig, default_config

FEATURES_FILENAME = "encounter_features.parquet"
MODEL_FILENAME = "model.joblib"


def load_feature_table(config):
    """Load the feature table from disk.

    Inputs:
        config: PipelineConfig containing feature_store_dir.
    Outputs:
        features_df: tabular dataset with target and feature columns.
    """
    feature_path = config.feature_store_dir / FEATURES_FILENAME
    
    if not feature_path.exists():
        raise FileNotFoundError(
            f"Features not found at: {feature_path}. "
            "Please run 'src/features.py' first!"
        )
    
    print(f"   Loading features from {feature_path.name}...")
    return pd.read_parquet(feature_path)


def split_train_val(features_df, time_column="encounter_start", val_fraction=0.2):
    """Split features into train/validation sets.

    Inputs:
        features_df: DataFrame from load_feature_table.
        time_column: column name for chronological splitting.
        val_fraction: fraction of newest rows used for validation.
    Outputs:
        train_df, val_df: split datasets.
    """
    # 1. Separate Features (X) and Target (y)
    # We remove metadata columns that shouldn't be used for prediction
    target_column = "encounter_duration_minutes"
    ignore_cols = [target_column, "case:concept:name", "encounter_start", "encounter_end"]
    
    # We create a dictionary to hold X and y for training
    # Note: For this baseline, we use a random split for simplicity/robustness
    X = features_df.drop(columns=[c for c in ignore_cols if c in features_df.columns], errors='ignore')
    y = features_df[target_column]
    
    print(f"   Dataset shape: {X.shape[0]} rows, {X.shape[1]} features.")
    
    # Random split is often more robust for baseline models than strict time splitting
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=val_fraction, random_state=42)
    
    return (X_train, y_train), (X_val, y_val)


def train_baseline_model(train_data, target_column="encounter_duration_minutes"):
    """Fit a baseline model.

    Inputs:
        train_data: tuple (X_train, y_train).
        target_column: target variable name.
    Outputs:
        model: trained model instance.
    """
    X_train, y_train = train_data
    print("   Training Random Forest model (Baseline)...")
    
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    
    return model


def evaluate_model(model, val_data, target_column="encounter_duration_minutes"):
    """Evaluate the model on validation data.

    Inputs:
        model: trained model instance.
        val_data: tuple (X_val, y_val).
        target_column: target variable name.
    Outputs:
        metrics: dict-like structure with evaluation results.
    """
    X_val, y_val = val_data
    predictions = model.predict(X_val)
    
    mae = mean_absolute_error(y_val, predictions)
    r2 = r2_score(y_val, predictions)
    
    return {"MAE": mae, "R2": r2}


def save_artifacts(config, model, metrics):
    """Persist model artifacts and metrics.

    Inputs:
        config: PipelineConfig containing model_dir.
        model: trained model instance.
        metrics: evaluation results.
    Outputs:
        artifact_paths: dict of output locations.
    """
    # Ensure directory exists
    config.model_dir.mkdir(parents=True, exist_ok=True)
    
    model_path = config.model_dir / MODEL_FILENAME
    joblib.dump(model, model_path)
    
    print(f"Model saved at: {model_path}")
    print(f"Final Metrics: {metrics}")
    
    return {"model_path": str(model_path)}


@dataclass
class DefaultModelTrainer:
    """Boilerplate trainer orchestrating the functions above."""

    config: PipelineConfig

    def train_model(self):
        """Orchestrate training pipeline and write artifacts."""
        print("Starting Model Training Pipeline...")
        
        # 1. Load
        df = load_feature_table(self.config)
        
        # 2. Split
        train_data, val_data = split_train_val(df)
        
        # 3. Train
        model = train_baseline_model(train_data)
        
        # 4. Evaluate
        metrics = evaluate_model(model, val_data)
        
        # 5. Save
        save_artifacts(self.config, model, metrics)


# --- EXECUTION BLOCK ---
if __name__ == "__main__":
    # Load default config and run the trainer
    cfg = default_config(PROJECT_ROOT)
    trainer = DefaultModelTrainer(cfg)
    trainer.train_model()