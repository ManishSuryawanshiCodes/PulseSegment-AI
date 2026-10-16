from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field


# --- Dataset Manager Models ---

class DatasetInfo(BaseModel):
    id: str
    name: str
    row_count: int
    col_count: int
    is_synthetic: bool = False
    real_file_missing: bool = False
    real_file_path: Optional[str] = None
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
    real_file_missing: bool = False
    real_file_path: Optional[str] = None
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
    enable_pulsecluster: bool = True
    pulsecluster_ratios: bool = True
    pulsecluster_time_budget: float = 20.0


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
    common_space_silhouette: Optional[float] = None
    common_space_dbi: Optional[float] = None
    common_space_chi: Optional[float] = None
    composite_score: Optional[float] = None
    execution_time_ms: float
    cluster_count: int
    noise_percentage: float
    sampled_silhouette: bool = False
    silhouette_sample_size: Optional[int] = None
    status: str = "success"
    reason: Optional[str] = None
    scatter_data: List[Dict[str, Any]] = Field(default_factory=list)
    cluster_sizes: Optional[Dict[int, int]] = Field(default_factory=dict)
    reliability: Optional[float] = None
    stability: Optional[float] = None
    timing_breakdown: Optional[Dict[str, float]] = None


class KSweepPoint(BaseModel):
    k: int
    inertia: float
    silhouette: Optional[float] = None
    composite_rank: Optional[float] = None
    min_cluster_size_pct: Optional[float] = None


class ExperimentsMatrix(BaseModel):
    dataset_id: Optional[str] = None
    config_hash: Optional[str] = None
    experiments: List[ExperimentResult]
    winning_experiment: Optional[ExperimentResult] = None
    selection_formula: Optional[str] = None
    k_sweep: Optional[List[KSweepPoint]] = None
    suggested_k: Optional[int] = None
    suggested_k_reason: Optional[str] = None


# --- Growth Personas & Next-Best-Action (NBA) Models ---

class DistinguishingFeature(BaseModel):
    feature: str
    z_score: float
    cluster_mean: float
    global_mean: float
    interpretation: Optional[str] = None


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
    distinguishing_note: Optional[str] = None
    characteristics: List[str] = Field(default_factory=list)
    assumptions: Optional[PersonaAssumptions] = None
    nba_campaigns: List[NBACampaign] = Field(default_factory=list)
    real_unit_centroid: Optional[Dict[str, float]] = Field(default_factory=dict)
    value_score: Optional[float] = None
    risk_score: Optional[float] = None
    value_metric_name: Optional[str] = None
    risk_formula_explanation: Optional[str] = None


class ClusterBoxPlotStat(BaseModel):
    min: float
    q1: float
    median: float
    q3: float
    max: float


class FeatureBoxPlot(BaseModel):
    feature: str
    clusters: Dict[str, ClusterBoxPlotStat]  # cluster name -> stats


class GrowthPersonas(BaseModel):
    dataset_id: Optional[str] = None
    personas: List[PersonaProfile]
    winning_model: str
    winning_silhouette: Optional[float] = None
    winning_dbi: Optional[float] = None
    domain_template: Optional[str] = None
    box_plots: Optional[List[FeatureBoxPlot]] = Field(default_factory=list)
    direction_map: Optional[Dict[str, str]] = Field(default_factory=dict)
    global_real_means: Optional[Dict[str, float]] = Field(default_factory=dict)



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


class ModelJSONExport(BaseModel):
    dataset_id: str
    exported_at: str
    winning_algorithm: str
    winning_space: str
    feature_names: List[str]
    scaler: Dict[str, Any]
    centroids: Dict[str, Dict[str, float]]
    pca_parameters: Optional[Dict[str, Any]] = None
    personas: List[Dict[str, Any]]


# --- New Advanced Intelligence Models ---

class SegmentScores(BaseModel):
    cluster_label: int
    persona_name: str
    value_score: float  # Share of spend/revenue (0-100%)
    risk_score: float   # Weighted risk score (0-100)
    customer_count: int
    customer_pct: float
    value_metric_name: str
    formula_explanation: str


class CustomerLookupResponse(BaseModel):
    customer_id: str
    dataset_id: str
    cluster_label: int
    persona_name: str
    confidence: float
    distance_to_centroid: float
    distinguishing_traits: List[str]
    customer_features: Dict[str, float]
    cluster_centroid: Dict[str, float]
    differences_vs_cluster: Dict[str, float]


class SummaryReportResponse(BaseModel):
    dataset_id: str
    generated_at: str
    source_type: str  # 'Template' | 'AI-written'
    summary_text: str
    key_findings: List[str]
    aggregate_stats: Dict[str, Any]


class SavedRunInfo(BaseModel):
    run_id: str
    dataset_id: str
    config_hash: str
    winning_algorithm: str
    winning_space: str
    n_clusters: int
    created_at: str
    config: Dict[str, Any]
    metrics: Dict[str, Any]


class CompareRunsResponse(BaseModel):
    run1: SavedRunInfo
    run2: SavedRunInfo
    metrics_comparison: Dict[str, Dict[str, Any]]
    config_diff: Dict[str, Any]


class DriftMigrationItem(BaseModel):
    original_group: str
    new_group: str
    count: int
    percent: float


class DriftCheckResponse(BaseModel):
    original_dataset_id: str
    drift_dataset_name: str
    total_customers: int
    migration_matrix: List[List[int]]
    group_names_original: List[str]
    group_names_new: List[str]
    migration_table: List[DriftMigrationItem]
    stability_index: float  # Percent staying in identical archetype
    drift_alert: str

