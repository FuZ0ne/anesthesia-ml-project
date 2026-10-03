"""Reusable preprocessing utilities for the anesthesia ML project."""

from __future__ import annotations
from pathlib import Path
import pandas as pd

# Columns excluded from machine-learning features because they are identifiers or contain postoperative / target-related information.
LEAKAGE_COLUMNS = [
    "PatientID",
    "Complications",
    "PostoperativeNotes",
    "PainLevel",
]
# Columns that must exist in the raw dataset before preprocessing.
REQUIRED_COLUMNS = [
    "Age",
    "Gender",
    "BMI",
    "SurgeryType",
    "SurgeryDuration",
    "AnesthesiaType",
    "PreoperativeNotes",
    "Outcome",
]
def load_raw_data(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path)

def prepare_ml_data(df: pd.DataFrame) -> pd.DataFrame:
    missing_columns = [
        column for column in REQUIRED_COLUMNS
        if column not in df.columns
        ]

    if missing_columns:
        raise ValueError(
            "Missing required columns: "
            + ", ".join(missing_columns)
        )

    data = df.copy()
    # Remove identifiers and postoperative / target-related variables.
    data = data.drop(
        columns=LEAKAGE_COLUMNS,
        errors="ignore",
    )
    # Convert values such as "217 min" into numeric minutes.
    duration = pd.to_numeric(
        data["SurgeryDuration"]
        .astype("string")
        .str.extract(r"(\d+(?:\.\d+)?)")[0],
        errors="coerce",
    )
    if duration.isna().any():
        invalid_count = int(duration.isna().sum())

        raise ValueError(
            f"Could not parse SurgeryDuration for "
            f"{invalid_count} row(s)."
        )
    data["SurgeryDuration_min"] = duration.astype(int)

    data = data.drop(
        columns=["SurgeryDuration"]
    )
    # Create a binary indicator for documented hypertension or diabetes in the preoperative notes.
    data["Has_Comorbidities"] = (
        data["PreoperativeNotes"]
        .astype("string")
        .str.contains(
            r"Hypertension|diabetes",
            case=False,
            regex=True,
            na=False,
        )
        .astype(int)
    )
    # The original free-text field is no longer required.
    data = data.drop(
        columns=["PreoperativeNotes"]
    )
    return data

def split_features_target(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    if "Outcome" not in df.columns:
        raise ValueError(
            "The prepared dataset must contain an 'Outcome' column."
        )
    X = df.drop(columns=["Outcome"])
    y = df["Outcome"]

    return X, y