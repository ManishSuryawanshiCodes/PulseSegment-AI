import os
import io
import re
import uuid
import datetime
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
from fastapi import HTTPException, UploadFile
from models import (
    DatasetInfo,
    ColumnProfile,
    DatasetProfile,
    DatasetConfig,
    ConfigResponse,
)


def generate_latent_credit_card_demo() -> pd.DataFrame:
    """
    Generates a realistic synthetic credit card dataset (N=8,950, 18 columns)
    derived from a multi-factor latent behavioral model to produce realistic
    correlations (purchases vs oneoff/installments, cash advance vs cash advance trx).
    """
    np.random.seed(42)
    n = 8950

    # 4 latent behavioral segments with realistic proportions
    # 1: Everyday Spenders (~38%), 2: Revolving Debtors (~26%), 3: Affluent VIPs (~14%), 4: Inactive/Dormant (~22%)
    segment_probs = [0.384, 0.261, 0.142, 0.213]
    segments = np.random.choice([0, 1, 2, 3], size=n, p=segment_probs)

    cust_id = [f"C{10001 + i}" for i in range(n)]

    # Latent factors per segment
    # Segment 0 (Transactor): Moderate balance, high purchase volume, high full payment rate, zero cash advance
    # Segment 1 (Revolver): High revolving balance, low full payment, moderate installment, high cash advance
    # Segment 2 (VIP Spender): Highest credit limit, high oneoff & installments, large payments
    # Segment 3 (Dormant): Very low activity across all transactions, stagnant low balance

    balance = np.zeros(n)
    balance_freq = np.zeros(n)
    purchases = np.zeros(n)
    oneoff = np.zeros(n)
    installments = np.zeros(n)
    cash_adv = np.zeros(n)
    purchases_freq = np.zeros(n)
    oneoff_freq = np.zeros(n)
    installments_freq = np.zeros(n)
    cash_adv_freq = np.zeros(n)
    cash_adv_trx = np.zeros(n, dtype=int)
    purchases_trx = np.zeros(n, dtype=int)
    credit_limit = np.zeros(n)
    payments = np.zeros(n)
    min_payments = np.zeros(n)
    prc_full = np.zeros(n)
    tenure = np.random.choice([6, 7, 8, 9, 10, 11, 12], size=n, p=[0.02, 0.02, 0.02, 0.03, 0.04, 0.07, 0.80])

    for i in range(n):
        s = segments[i]
        if s == 0:  # Transactor
            balance[i] = np.random.lognormal(5.8, 0.6)  # ~350 - 700
            balance_freq[i] = np.clip(np.random.beta(8, 2), 0.5, 1.0)
            purchases[i] = np.random.lognormal(7.2, 0.5)  # ~1300
            oneoff[i] = purchases[i] * np.random.uniform(0.3, 0.6)
            installments[i] = purchases[i] - oneoff[i]
            cash_adv[i] = 0.0 if np.random.rand() > 0.08 else np.random.lognormal(4.5, 0.5)
            purchases_freq[i] = np.clip(np.random.beta(7, 2), 0.6, 1.0)
            oneoff_freq[i] = purchases_freq[i] * np.random.uniform(0.3, 0.5)
            installments_freq[i] = purchases_freq[i] * np.random.uniform(0.6, 0.9)
            cash_adv_freq[i] = 0.0 if cash_adv[i] == 0 else np.random.uniform(0.01, 0.08)
            cash_adv_trx[i] = 0 if cash_adv[i] == 0 else np.random.poisson(1)
            purchases_trx[i] = int(purchases_freq[i] * 28 + np.random.poisson(3))
            credit_limit[i] = np.random.lognormal(8.3, 0.4)  # ~4000
            payments[i] = purchases[i] * np.random.uniform(0.9, 1.3)
            min_payments[i] = balance[i] * 0.05 + np.random.uniform(10, 40)
            prc_full[i] = np.clip(np.random.beta(8, 2), 0.6, 1.0)
        elif s == 1:  # Revolver
            balance[i] = np.random.lognormal(8.2, 0.5)  # ~3600
            balance_freq[i] = np.clip(np.random.beta(9, 1), 0.8, 1.0)
            purchases[i] = np.random.lognormal(5.8, 0.6)  # ~330
            oneoff[i] = purchases[i] * 0.2 if np.random.rand() > 0.4 else 0.0
            installments[i] = purchases[i] - oneoff[i]
            cash_adv[i] = np.random.lognormal(7.3, 0.6)  # ~1500
            purchases_freq[i] = np.clip(np.random.beta(3, 4), 0.1, 0.6)
            oneoff_freq[i] = 0.0 if oneoff[i] == 0 else purchases_freq[i] * 0.2
            installments_freq[i] = purchases_freq[i] * 0.6
            cash_adv_freq[i] = np.clip(np.random.beta(4, 3), 0.2, 0.7)
            cash_adv_trx[i] = max(1, np.random.poisson(5))
            purchases_trx[i] = int(purchases_freq[i] * 10 + np.random.poisson(2))
            credit_limit[i] = np.random.lognormal(8.2, 0.4)  # ~3600
            payments[i] = np.random.lognormal(6.8, 0.5)  # ~900
            min_payments[i] = balance[i] * 0.08 + np.random.uniform(20, 80)
            prc_full[i] = np.clip(np.random.beta(0.5, 9), 0.0, 0.15)
        elif s == 2:  # VIP Spender
            balance[i] = np.random.lognormal(7.6, 0.6)  # ~2000
            balance_freq[i] = np.clip(np.random.beta(9, 1), 0.85, 1.0)
            purchases[i] = np.random.lognormal(8.3, 0.5)  # ~4000
            oneoff[i] = purchases[i] * np.random.uniform(0.6, 0.85)
            installments[i] = purchases[i] - oneoff[i]
            cash_adv[i] = 0.0 if np.random.rand() > 0.35 else np.random.lognormal(6.9, 0.6)
            purchases_freq[i] = np.clip(np.random.beta(8, 2), 0.7, 1.0)
            oneoff_freq[i] = purchases_freq[i] * 0.7
            installments_freq[i] = purchases_freq[i] * 0.5
            cash_adv_freq[i] = 0.0 if cash_adv[i] == 0 else np.random.uniform(0.05, 0.25)
            cash_adv_trx[i] = 0 if cash_adv[i] == 0 else np.random.poisson(2)
            purchases_trx[i] = int(purchases_freq[i] * 40 + np.random.poisson(8))
            credit_limit[i] = np.random.lognormal(9.3, 0.4)  # ~11000
            payments[i] = purchases[i] * np.random.uniform(0.95, 1.4)
            min_payments[i] = balance[i] * 0.05 + np.random.uniform(20, 50)
            prc_full[i] = np.clip(np.random.beta(5, 3), 0.3, 0.9)
        else:  # Dormant / At-Risk
            balance[i] = np.random.lognormal(5.0, 0.7)  # ~150 (verified low!)
            balance_freq[i] = np.clip(np.random.beta(4, 5), 0.2, 0.8)
            purchases[i] = 0.0 if np.random.rand() > 0.2 else np.random.lognormal(3.8, 0.6)
            oneoff[i] = purchases[i] if np.random.rand() > 0.5 else 0.0
            installments[i] = purchases[i] - oneoff[i]
            cash_adv[i] = 0.0 if np.random.rand() > 0.1 else np.random.lognormal(4.0, 0.5)
            purchases_freq[i] = 0.0 if purchases[i] == 0 else np.random.uniform(0.02, 0.15)
            oneoff_freq[i] = purchases_freq[i] * 0.5
            installments_freq[i] = purchases_freq[i] * 0.5
            cash_adv_freq[i] = 0.0 if cash_adv[i] == 0 else np.random.uniform(0.01, 0.05)
            cash_adv_trx[i] = 0 if cash_adv[i] == 0 else 1
            purchases_trx[i] = 0 if purchases[i] == 0 else np.random.poisson(1)
            credit_limit[i] = np.random.lognormal(7.6, 0.4)  # ~2000
            payments[i] = np.random.lognormal(4.8, 0.6)  # ~120
            min_payments[i] = balance[i] * 0.04 + np.random.uniform(5, 20)
            prc_full[i] = np.clip(np.random.beta(1, 6), 0.0, 0.25)

    df = pd.DataFrame(
        {
            "CUST_ID": cust_id,
            "BALANCE": np.round(balance, 2),
            "BALANCE_FREQUENCY": np.round(balance_freq, 4),
            "PURCHASES": np.round(purchases, 2),
            "ONEOFF_PURCHASES": np.round(oneoff, 2),
            "INSTALLMENTS_PURCHASES": np.round(installments, 2),
            "CASH_ADVANCE": np.round(cash_adv, 2),
            "PURCHASES_FREQUENCY": np.round(purchases_freq, 4),
            "ONEOFF_PURCHASES_FREQUENCY": np.round(oneoff_freq, 4),
            "PURCHASES_INSTALLMENTS_FREQUENCY": np.round(installments_freq, 4),
            "CASH_ADVANCE_FREQUENCY": np.round(cash_adv_freq, 4),
            "CASH_ADVANCE_TRX": cash_adv_trx,
            "PURCHASES_TRX": purchases_trx,
            "CREDIT_LIMIT": np.round(credit_limit, 2),
            "PAYMENTS": np.round(payments, 2),
            "MINIMUM_PAYMENTS": np.round(min_payments, 2),
            "PRC_FULL_PAYMENT": np.round(prc_full, 4),
            "TENURE": tenure,
        }
    )

    # Missing value injection for standard benchmark realism
    nan_credit = np.random.choice(n, size=1, replace=False)
    df.loc[nan_credit, "CREDIT_LIMIT"] = np.nan

    nan_minpay = np.random.choice(n, size=313, replace=False)
    df.loc[nan_minpay, "MINIMUM_PAYMENTS"] = np.nan

    return df


class DatasetManager:
    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        self.uploads_dir = os.path.join(data_dir, "uploads")
        os.makedirs(self.uploads_dir, exist_ok=True)

        self._datasets: Dict[str, Dict[str, Any]] = {}
        self._configs: Dict[str, DatasetConfig] = {}
        self._pipeline_instances: Dict[str, Any] = {}

        self._init_builtins()

    def _init_builtins(self):
        """Initialize the built-in Credit Card sample dataset."""
        sample_path = os.path.join(self.data_dir, "cc_general.csv")
        sample_id = "sample_credit_card"

        is_synthetic = False
        if os.path.exists(sample_path):
            try:
                df = pd.read_csv(sample_path)
            except Exception:
                df = generate_latent_credit_card_demo()
                is_synthetic = True
        else:
            df = generate_latent_credit_card_demo()
            is_synthetic = True
            try:
                df.to_csv(sample_path, index=False)
            except Exception:
                pass

        self._datasets[sample_id] = {
            "id": sample_id,
            "name": "Credit Card (Sample)",
            "df": df,
            "path": sample_path,
            "is_synthetic": is_synthetic,
            "created_at": datetime.datetime.now().isoformat(),
        }

        # Default configuration
        self._configs[sample_id] = DatasetConfig(
            id_column="CUST_ID",
            include_columns=None,
            imputation_k=5,
            iqr_multiplier=1.5,
            corr_threshold=0.85,
            mi_k=8,
            pca_variance=0.95,
            n_clusters=4,
            domain_template="credit_card",
        )

        # Also register existing uploads if any
        if os.path.exists(self.uploads_dir):
            for fname in os.listdir(self.uploads_dir):
                fpath = os.path.join(self.uploads_dir, fname)
                if os.path.isfile(fpath) and (fname.endswith(".csv") or fname.endswith(".xlsx") or fname.endswith(".xls")):
                    parts = fname.split("_")
                    if fname.startswith("ds_") and len(parts) >= 3:
                        uid = f"{parts[0]}_{parts[1]}"
                        display_name = "_".join(parts[2:])
                    else:
                        uid = parts[0]
                        display_name = fname
                    try:
                        if fname.endswith(".csv"):
                            udf = pd.read_csv(fpath)
                        else:
                            udf = pd.read_excel(fpath)
                        self._datasets[uid] = {
                            "id": uid,
                            "name": display_name,
                            "df": udf,
                            "path": fpath,
                            "is_synthetic": False,
                            "created_at": datetime.datetime.fromtimestamp(os.path.getctime(fpath)).isoformat(),
                        }
                        # Auto configure
                        numeric_cols = list(udf.select_dtypes(include=[np.number]).columns)
                        id_col = self._auto_detect_id_column(udf)
                        domain = self._auto_detect_domain(udf)
                        self._configs[uid] = DatasetConfig(
                            id_column=id_col,
                            include_columns=numeric_cols,
                            domain_template=domain,
                            mi_k=min(8, max(2, len(numeric_cols) - 1)) if numeric_cols else 4,
                        )
                    except Exception:
                        pass

    def list_datasets(self) -> List[DatasetInfo]:
        """List all available datasets."""
        res = []
        for ds_id, data in self._datasets.items():
            df = data["df"]
            res.append(
                DatasetInfo(
                    id=ds_id,
                    name=data["name"],
                    row_count=int(df.shape[0]),
                    col_count=int(df.shape[1]),
                    is_synthetic=data.get("is_synthetic", False),
                    created_at=data.get("created_at", ""),
                    has_results=ds_id in self._pipeline_instances,
                )
            )
        return res

    def get_dataset(self, dataset_id: str) -> Tuple[pd.DataFrame, bool, str]:
        """Get dataset DataFrame, is_synthetic flag, and display name."""
        if dataset_id not in self._datasets:
            raise HTTPException(status_code=404, detail=f"Dataset with ID '{dataset_id}' not found.")
        d = self._datasets[dataset_id]
        return d["df"], d.get("is_synthetic", False), d["name"]

    def upload_dataset(self, file: UploadFile) -> DatasetInfo:
        """
        Validate, parse, and store an uploaded dataset.
        Validates: max 50MB, >= 100 rows, >= 3 numeric columns, handles encodings & delimiters.
        """
        filename = file.filename or "uploaded_data.csv"
        ext = os.path.splitext(filename)[1].lower()
        if ext not in [".csv", ".xlsx", ".xls", ".txt"]:
            raise HTTPException(
                status_code=422,
                detail=f"Unsupported file format '{ext}'. Please upload a CSV (.csv) or Excel (.xlsx, .xls) file.",
            )

        # Read file bytes into memory
        contents = file.file.read()
        max_bytes = 50 * 1024 * 1024  # 50MB
        if len(contents) > max_bytes:
            raise HTTPException(
                status_code=422,
                detail="File size exceeds maximum allowed limit of 50 MB.",
            )

        # Parse CSV or Excel
        df = None
        if ext in [".xlsx", ".xls"]:
            try:
                df = pd.read_excel(io.BytesIO(contents))
            except Exception as e:
                raise HTTPException(
                    status_code=422,
                    detail=f"Unable to parse Excel workbook: {str(e)}",
                )
        else:
            # Try multiple encodings and separators
            encodings = ["utf-8", "latin1", "iso-8859-1", "cp1252"]
            parsed = False
            last_err = ""
            for enc in encodings:
                try:
                    # Let pandas auto-detect delimiter via python engine
                    df = pd.read_csv(io.BytesIO(contents), encoding=enc, sep=None, engine="python")
                    parsed = True
                    break
                except Exception as e:
                    last_err = str(e)

            if not parsed or df is None:
                raise HTTPException(
                    status_code=422,
                    detail=f"Failed to parse CSV file across standard encodings (utf-8, latin1): {last_err}",
                )

        # Validate rows >= 100
        n_rows = len(df)
        if n_rows < 100:
            raise HTTPException(
                status_code=422,
                detail=f"Dataset must contain at least 100 records for clustering analysis; received {n_rows} rows.",
            )

        # Validate numeric columns >= 3
        numeric_cols = list(df.select_dtypes(include=[np.number]).columns)
        if len(numeric_cols) < 3:
            raise HTTPException(
                status_code=422,
                detail=f"Dataset must contain at least 3 numeric features for multi-dimensional clustering; detected {len(numeric_cols)} numeric columns ({numeric_cols}).",
            )

        # Generate unique dataset ID
        dataset_id = f"ds_{uuid.uuid4().hex[:8]}"
        safe_name = re.sub(r"[^\w\.-]", "_", filename)
        save_path = os.path.join(self.uploads_dir, f"{dataset_id}_{safe_name}")

        try:
            if ext in [".xlsx", ".xls"]:
                df.to_excel(save_path, index=False)
            else:
                df.to_csv(save_path, index=False)
        except Exception:
            pass

        # Register in memory
        self._datasets[dataset_id] = {
            "id": dataset_id,
            "name": filename,
            "df": df,
            "path": save_path,
            "is_synthetic": False,
            "created_at": datetime.datetime.now().isoformat(),
        }

        # Auto-detect sensible defaults
        id_col = self._auto_detect_id_column(df)
        domain = self._auto_detect_domain(df)

        self._configs[dataset_id] = DatasetConfig(
            id_column=id_col,
            include_columns=numeric_cols,
            imputation_k=5,
            iqr_multiplier=1.5,
            corr_threshold=0.85,
            mi_k=min(8, max(2, len(numeric_cols) - 1)),
            pca_variance=0.95,
            n_clusters=4,
            domain_template=domain,
        )

        return DatasetInfo(
            id=dataset_id,
            name=filename,
            row_count=n_rows,
            col_count=int(df.shape[1]),
            is_synthetic=False,
            created_at=self._datasets[dataset_id]["created_at"],
            has_results=False,
        )

    def _auto_detect_id_column(self, df: pd.DataFrame) -> Optional[str]:
        """Heuristic to detect identifier column."""
        for col in df.columns:
            clow = col.lower()
            if any(term in clow for term in ["cust_id", "customer_id", "user_id", "client_id", "uuid"]):
                return col
            if (clow == "id" or clow.endswith("_id")) and df[col].nunique() == len(df):
                return col
        return None

    def _auto_detect_domain(self, df: pd.DataFrame) -> str:
        """Infer business domain based on column signatures."""
        cols_lower = [c.lower() for c in df.columns]
        # Credit Card checks
        if any("balance" in c for c in cols_lower) and any("purchase" in c for c in cols_lower):
            return "credit_card"
        # RFM checks
        if any("recency" in c for c in cols_lower) or any("frequency" in c for c in cols_lower) or any("monetary" in c for c in cols_lower):
            return "ecommerce_rfm"
        # SaaS checks
        if any("mrr" in c for c in cols_lower) or any("seat" in c for c in cols_lower) or any("login" in c for c in cols_lower) or any("churn" in c for c in cols_lower):
            return "saas"
        return "generic"

    def profile_dataset(self, dataset_id: str) -> DatasetProfile:
        """Profile dataset columns, detect dtypes, missing values, unique values, suggested roles, and 10-row preview."""
        df, is_synthetic, name = self.get_dataset(dataset_id)
        n_rows = len(df)

        col_profiles: List[ColumnProfile] = []
        for col in df.columns:
            dtype_str = str(df[col].dtype)
            missing = int(df[col].isna().sum())
            missing_pct = round(float((missing / n_rows) * 100.0), 2)
            uniques = int(df[col].nunique(dropna=True))

            # Sample values (up to 5)
            sample_non_null = df[col].dropna().head(5).tolist()
            sample_vals = [str(x) if not isinstance(x, (int, float, bool)) else x for x in sample_non_null]

            # Suggested role
            clow = col.lower()
            is_numeric = pd.api.types.is_numeric_dtype(df[col])

            if (
                any(term in clow for term in ["cust_id", "customer_id", "user_id", "client_id", "uuid"])
                or (clow == "id" or clow.endswith("_id"))
                or (uniques == n_rows and not is_numeric)
            ):
                role = "id"
            elif uniques <= 1:
                role = "constant"
            elif missing_pct > 80.0:
                role = "exclude"
            elif is_numeric:
                role = "numeric_feature"
            elif uniques < 50:
                role = "categorical"
            else:
                role = "exclude"

            col_profiles.append(
                ColumnProfile(
                    name=col,
                    dtype=dtype_str,
                    missing_count=missing,
                    missing_pct=missing_pct,
                    unique_count=uniques,
                    suggested_role=role,
                    sample_values=sample_vals,
                )
            )

        # 10-row preview
        preview_df = df.head(10).copy()
        # Clean preview for JSON compliance
        preview_records = []
        for _, row in preview_df.iterrows():
            rec = {}
            for k, v in row.items():
                if pd.isna(v):
                    rec[k] = None
                elif isinstance(v, (np.floating, float)):
                    rec[k] = round(float(v), 4) if not np.isinf(v) else None
                elif isinstance(v, (np.integer, int)):
                    rec[k] = int(v)
                else:
                    rec[k] = str(v)
            preview_records.append(rec)

        return DatasetProfile(
            dataset_id=dataset_id,
            name=name,
            row_count=n_rows,
            col_count=int(df.shape[1]),
            is_synthetic=is_synthetic,
            columns=col_profiles,
            preview=preview_records,
        )

    def get_config(self, dataset_id: str) -> DatasetConfig:
        """Get dataset configuration."""
        if dataset_id not in self._configs:
            self._configs[dataset_id] = DatasetConfig()
        return self._configs[dataset_id]

    def configure_dataset(self, dataset_id: str, config: DatasetConfig) -> ConfigResponse:
        """Save configuration and invalidate cached pipeline instances."""
        if dataset_id not in self._datasets:
            raise HTTPException(status_code=404, detail=f"Dataset '{dataset_id}' not found.")

        self._configs[dataset_id] = config

        # Invalidate pipeline instance for this dataset
        if dataset_id in self._pipeline_instances:
            del self._pipeline_instances[dataset_id]

        config_hash = self._compute_config_hash(config)
        return ConfigResponse(
            dataset_id=dataset_id,
            config_hash=config_hash,
            config=config,
            message="Configuration saved successfully. Preprocessing pipeline refreshed.",
        )

    def _compute_config_hash(self, config: DatasetConfig) -> str:
        """Generate a deterministic config hash."""
        import hashlib
        import json
        cfg_dict = config.model_dump()
        raw = json.dumps(cfg_dict, sort_keys=True)
        return hashlib.md5(raw.encode("utf-8")).hexdigest()[:12]

    def get_pipeline(self, dataset_id: str = "sample_credit_card") -> Any:
        """Get or lazily instantiate the PulseSegmentPipeline for a dataset."""
        if dataset_id not in self._datasets:
            if "sample_credit_card" in self._datasets:
                dataset_id = "sample_credit_card"
            else:
                raise HTTPException(status_code=404, detail=f"Dataset '{dataset_id}' not found.")

        cfg = self.get_config(dataset_id)
        cfg_hash = self._compute_config_hash(cfg)

        if dataset_id in self._pipeline_instances:
            pipe, existing_hash = self._pipeline_instances[dataset_id]
            if existing_hash == cfg_hash:
                return pipe

        from pipeline import PulseSegmentPipeline
        df, is_synthetic, _ = self.get_dataset(dataset_id)
        pipe = PulseSegmentPipeline(
            df=df,
            config=cfg,
            dataset_id=dataset_id,
            is_synthetic=is_synthetic,
        )
        self._pipeline_instances[dataset_id] = (pipe, cfg_hash)
        return pipe

