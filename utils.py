from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


AFFAIRS_EXPECTED_COLS = [
    "rate_marriage",
    "age",
    "yrs_married",
    "children",
    "religious",
    "educ",
    "occupation",
    "occupation_husb",
    "affairs",
]


def load_affairs_csv(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    df = pd.read_csv(p)
    df.columns = [c.strip() for c in df.columns]
    missing = [c for c in AFFAIRS_EXPECTED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Kolom affairs_fair_dataset.csv tidak lengkap. Missing: {missing}")
    for c in AFFAIRS_EXPECTED_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["affairs"]).reset_index(drop=True)
    df["had_affair"] = (df["affairs"] > 0).astype(int)
    return df


def clamp_df_numeric(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def apply_filters(
    df: pd.DataFrame,
    age_range: Tuple[float, float] | None = None,
    yrs_range: Tuple[float, float] | None = None,
    children_values: Optional[List[int]] = None,
    religious_range: Tuple[float, float] | None = None,
    educ_range: Tuple[float, float] | None = None,
    occupation_values: Optional[List[int]] = None,
) -> pd.DataFrame:
    out = df.copy()

    if age_range is not None and "age" in out.columns:
        out = out[(out["age"] >= age_range[0]) & (out["age"] <= age_range[1])]
    if yrs_range is not None and "yrs_married" in out.columns:
        out = out[(out["yrs_married"] >= yrs_range[0]) & (out["yrs_married"] <= yrs_range[1])]
    if children_values is not None and "children" in out.columns and len(children_values) > 0:
        out = out[out["children"].round().astype(int).isin(children_values)]
    if religious_range is not None and "religious" in out.columns:
        out = out[(out["religious"] >= religious_range[0]) & (out["religious"] <= religious_range[1])]
    if educ_range is not None and "educ" in out.columns:
        out = out[(out["educ"] >= educ_range[0]) & (out["educ"] <= educ_range[1])]
    if occupation_values is not None and "occupation" in out.columns and len(occupation_values) > 0:
        out = out[out["occupation"].round().astype(int).isin(occupation_values)]

    return out.reset_index(drop=True)


def affair_proportion(df: pd.DataFrame) -> Dict[str, float]:
    n = len(df)
    if n == 0:
        return {"n": 0, "p_had_affair": float("nan")}
    p = float(df["had_affair"].mean())
    return {"n": n, "p_had_affair": p}


def describe_numeric(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    cols2 = [c for c in cols if c in df.columns]
    if not cols2:
        return pd.DataFrame()
    desc = df[cols2].describe().T
    desc = desc.rename(columns={"50%": "median"})
    return desc[["count", "mean", "std", "min", "median", "max"]]


def bucket_series(s: pd.Series, bins: List[float], labels: List[str]) -> pd.Series:
    return pd.cut(s, bins=bins, labels=labels, include_lowest=True, right=True)


def affair_rate_by_bucket(
    df: pd.DataFrame,
    col: str,
    bins: List[float],
    labels: List[str],
) -> pd.DataFrame:
    if col not in df.columns:
        return pd.DataFrame()
    tmp = df.copy()
    tmp["_bucket"] = bucket_series(tmp[col], bins=bins, labels=labels)
    out = (
        tmp.groupby("_bucket", observed=False)["had_affair"]
        .agg(["mean", "count"])
        .reset_index()
        .rename(columns={"_bucket": "bucket", "mean": "rate", "count": "n"})
    )
    out["rate"] = out["rate"].astype(float)
    return out


def mean_diff_rank(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str = "had_affair",
) -> pd.DataFrame:
    cols = [c for c in feature_cols if c in df.columns]
    if len(df) == 0 or len(cols) == 0:
        return pd.DataFrame(columns=["feature", "mean_affair", "mean_no_affair", "diff"])
    g1 = df[df[target_col] == 1][cols].mean(numeric_only=True)
    g0 = df[df[target_col] == 0][cols].mean(numeric_only=True)
    out = pd.DataFrame(
        {
            "feature": cols,
            "mean_affair": [float(g1.get(c, np.nan)) for c in cols],
            "mean_no_affair": [float(g0.get(c, np.nan)) for c in cols],
        }
    )
    out["diff"] = out["mean_affair"] - out["mean_no_affair"]
    out = out.sort_values("diff", ascending=False).reset_index(drop=True)
    return out


@dataclass
class ModelArtifacts:
    pipeline: Pipeline
    feature_cols: List[str]
    auc: float
    fpr: np.ndarray
    tpr: np.ndarray
    threshold: np.ndarray


def train_affair_model(
    df: pd.DataFrame,
    feature_cols: Optional[List[str]] = None,
    test_size: float = 0.25,
    random_state: int = 42,
) -> ModelArtifacts:
    if feature_cols is None:
        feature_cols = [
            "rate_marriage",
            "age",
            "yrs_married",
            "children",
            "religious",
            "educ",
            "occupation",
            "occupation_husb",
        ]
    feature_cols = [c for c in feature_cols if c in df.columns]

    X = df[feature_cols].copy()
    y = df["had_affair"].astype(int).copy()

    X = X.fillna(X.median(numeric_only=True))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y if y.nunique() > 1 else None
    )

    numeric_features = feature_cols
    pre = ColumnTransformer(
        transformers=[("num", StandardScaler(), numeric_features)],
        remainder="drop",
        sparse_threshold=0.0,
    )

    clf = LogisticRegression(max_iter=2000, solver="lbfgs")
    pipe = Pipeline([("pre", pre), ("clf", clf)])

    pipe.fit(X_train, y_train)

    proba = pipe.predict_proba(X_test)[:, 1]
    auc = float(roc_auc_score(y_test, proba)) if y_test.nunique() > 1 else float("nan")
    fpr, tpr, thr = roc_curve(y_test, proba) if y_test.nunique() > 1 else (np.array([]), np.array([]), np.array([]))

    return ModelArtifacts(pipeline=pipe, feature_cols=feature_cols, auc=auc, fpr=fpr, tpr=tpr, threshold=thr)


def predict_probability(pipe: Pipeline, feature_cols: List[str], profile: Dict[str, float]) -> float:
    row = {c: float(profile.get(c, np.nan)) for c in feature_cols}
    X = pd.DataFrame([row])[feature_cols]
    if X.isna().any(axis=None):
        for c in feature_cols:
            if pd.isna(X.loc[0, c]):
                X.loc[0, c] = 0.0
    p = float(pipe.predict_proba(X)[:, 1][0])
    return p


def list_nikah_cerai_files(data_dir: str | Path) -> List[Path]:
    p = Path(data_dir)
    files = sorted(p.glob("nikah_cerai_*.xlsx"))
    return files


def _normalize_cols(cols: List[str]) -> List[str]:
    out = []
    for c in cols:
        s = str(c).strip().lower()
        s = s.replace("\n", " ")
        s = " ".join(s.split())
        out.append(s)
    return out


def load_nikah_cerai_xlsx(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    df = pd.read_excel(p, sheet_name=0)
    df.columns = _normalize_cols(list(df.columns))
    return df


def _infer_year_from_name(path: Path) -> Optional[int]:
    name = path.stem
    digits = "".join([ch for ch in name if ch.isdigit()])
    if len(digits) >= 4:
        try:
            return int(digits[:4])
        except Exception:
            return None
    return None


def _find_numeric_column(df: pd.DataFrame, keywords: List[str]) -> Optional[str]:
    cols = list(df.columns)
    score = []
    for c in cols:
        s = str(c).lower()
        hit = sum(1 for k in keywords if k in s)
        if hit > 0:
            score.append((hit, c))
    if not score:
        return None
    score.sort(reverse=True)
    cand = score[0][1]
    if pd.api.types.is_numeric_dtype(df[cand]) or df[cand].astype(str).str.replace(",", "", regex=False).str.match(r"^-?\d+(\.\d+)?$").mean() > 0.5:
        return cand
    return None


def aggregate_nikah_cerai_trend(data_dir: str | Path) -> pd.DataFrame:
    files = list_nikah_cerai_files(data_dir)
    rows = []
    for f in files:
        year = _infer_year_from_name(f)
        try:
            df = load_nikah_cerai_xlsx(f)
        except Exception:
            continue

        nikah_col = _find_numeric_column(df, ["nikah", "perkawinan", "pernikahan"])
        cerai_col = _find_numeric_column(df, ["cerai", "perceraian", "talak"])

        if nikah_col is None and cerai_col is None:
            continue

        def _to_num(s: pd.Series) -> pd.Series:
            if pd.api.types.is_numeric_dtype(s):
                return s
            return pd.to_numeric(s.astype(str).str.replace(",", "", regex=False), errors="coerce")

        nikah_sum = float(_to_num(df[nikah_col]).sum()) if nikah_col is not None else float("nan")
        cerai_sum = float(_to_num(df[cerai_col]).sum()) if cerai_col is not None else float("nan")

        if year is None:
            year = np.nan

        rows.append({"year": year, "nikah_total": nikah_sum, "cerai_total": cerai_sum, "file": f.name})

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out = out.sort_values("year").reset_index(drop=True)
    return out
