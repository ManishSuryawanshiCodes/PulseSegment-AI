from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field


# --- Dataset Manager Models ---

class DatasetInfo(BaseModel):
    id: str
    name: str
    row_count: int
    col_count: int
    is_synthetic: bool = False
    created_at: str
    has_results: bool = False


class ColumnProfile(BaseModel):
    name: str
    dtype: str
    missing_count: int
    missing_pct: float
    unique_count: int
    suggested_role: str  # 'id' | 'numeric_feature' | 'categorical' | 'constant' | 'exclude'
    sample_values: List[Any] = Field(default_factory=list)


class DatasetProfile(BaseModel):
    dataset_id: str
    name: str
    row_count: int
    col_count: int
    is_synthetic: bool = False
    columns: List[ColumnProfile]
    preview: List[Dict[str, Any]] = Field(default_factory=list)


class DatasetConfig(BaseModel):
    id_column: Optional[str] = None
    include_columns: Optional[List[str]] = None
    imputation_k: int = 5
    iqr_multiplier: float = 1.5
    corr_threshold: float = 0.85
    mi_k: Optional[int] = None
    pca_variance: float = 0.95
    n_clusters: int = 4
    domain_template: str = "credit_card"  # 'credit_card' | 'ecommerce_rfm' | 'saas' | 'generic'


class ConfigResponse(BaseModel):
    dataset_id: str
    config_hash: str
    config: DatasetConfig
    message: str


# --- Background Job Models ---

class JobStatusResponse(BaseModel):
    job_id: str
    dataset_id: str
    status: str  # 'pending' | 'running' | 'completed' | 'failed' | 'cancelled'
    completed: int
    total: int = 16
    current_step: str
    elapsed_ms: float
    error: Optional[str] = None


# --- Pipeline & Benchmark Models ---

class DatasetSummary(BaseModel):
    dataset_id: Optional[str] = None
    row_count: int
    col_count: int
    is_synthetic: bool = False
    columns: List[str]
    missing_values: Dict[str, int]
    descriptive_stats: Dict[str, Dict[str, float]]
    iqr_outlier_counts_before: Dict[str, int]
    iqr_outlier_counts_after: Dict[str, int]
    preprocessing_steps: List[str]


class PrunedFeatureInfo(BaseModel):
    feature: str
    correlated_with: str
    correlation: float


class FeatureEngineeringResult(BaseModel):
    dataset_id: Optional[str] = None
    original_features: List[str]
    ds2_correlation_features: List[str]
    ds2_removed_features: List[str]
    ds2_pruned_details: List[PrunedFeatureInfo] = Field(default_factory=list)
    correlation_matrix: List[List[float]]
    correlation_feature_names: List[str] = Field(default_factory=list)
    ds3_mutual_info_features: List[str]
    ds3_mutual_info_scores: Dict[str, float]
    ds3_method_note: Optional[str] = None
    ds4_pca_n_components: int
    ds4_pca_explained_variance_ratio: List[float]
    ds4_pca_cumulative_variance: List[float]


class ScatterPoint(BaseModel):
    pc1: float
    pc2: float
    cluster_label: int


class ExperimentResult(BaseModel):
    dataset_name: str
    dataset_dim: str
    algorithm: str
    silhouette_score: Optional[float] = None
    davies_bouldin_index: Optional[float] = None
    calinski_harabasz_score: Optional[float] = None
    composite_score: Optional[float] = None
    execution_time_ms: float
    cluster_count: int
    noise_percentage: float
    sampled_silhouette: bool = False
    status: str = "success"
    reason: Optional[str] = None
    scatter_data: List[Dict[str, Any]] = Field(default_factory=list)
    cluster_sizes: Optional[Dict[int, int]] = Field(default_factory=dict)


class KSweepPoint(BaseModel):
    k: int
    inertia: float
    silhouette: Optional[float] = None


class ExperimentsMatrix(BaseModel):
    dataset_id: Optional[str] = None
    config_hash: Optional[str] = None
    experiments: List[ExperimentResult]
    winning_experiment: Optional[ExperimentResult] = None
    selection_formula: Optional[str] = None
    k_sweep: Optional[List[KSweepPoint]] = None
    suggested_k: Optional[int] = None


# --- Growth Personas & Next-Best-Action (NBA) Models ---

class DistinguishingFeature(BaseModel):
    feature: str
    z_score: float
    cluster_mean: float
    global_mean: float


class NBACampaign(BaseModel):
    title: str
    description: str
    expected_roi_lift: str
    channel: str
    estimated_net_lift_dollars: Optional[float] = None
    roi_multiple: Optional[float] = None


class PersonaAssumptions(BaseModel):
    avg_revenue_per_customer: float = 120.0
    expected_uplift_pct: float = 0.15
    campaign_cost_per_customer: float = 3.50


class PersonaProfile(BaseModel):
    id: Optional[int] = None
    name: str
    percentage: float
    count: int
    color: str
    centroid_profile: Dict[str, float]
    top_positive_features: List[DistinguishingFeature] = Field(default_factory=list)
    top_negative_features: List[DistinguishingFeature] = Field(default_factory=list)
    characteristics: List[str]
    assumptions: Optional[PersonaAssumptions] = None
    nba_campaigns: List[NBACampaign] = Field(default_factory=list)


class GrowthPersonas(BaseModel):
    dataset_id: Optional[str] = None
    personas: List[PersonaProfile]
    winning_model: str
    winning_silhouette: Optional[float] = None
    winning_dbi: Optional[float] = None
    domain_template: Optional[str] = None


# --- Operational & Inference Models ---

class UploadResponse(BaseModel):
    dataset_id: str
    filename: str
    row_count: int
    col_count: int
    columns: List[str]
    is_synthetic: bool = False
    message: str


class PredictSegmentRequest(BaseModel):
    features: Dict[str, float]


class PredictSegmentResponse(BaseModel):
    cluster_label: int
    persona_name: str
    confidence: float
    recommended_nba: NBACampaign
    distance_to_centroid: float
    top_traits: List[str]
