const API_BASE = 'http://localhost:8000'

export interface DatasetInfo {
  id: string
  name: string
  row_count: number
  col_count: number
  is_synthetic: boolean
  real_file_missing?: boolean
  real_file_path?: string
  created_at: string
  has_results: boolean
}

export interface ColumnProfile {
  name: string
  dtype: string
  missing_count: number
  missing_pct: number
  unique_count: number
  suggested_role: 'id' | 'numeric_feature' | 'categorical' | 'constant' | 'exclude'
  sample_values: any[]
}

export interface DatasetProfile {
  dataset_id: string
  name: string
  row_count: number
  col_count: number
  is_synthetic: boolean
  real_file_missing?: boolean
  real_file_path?: string
  columns: ColumnProfile[]
  preview: Record<string, any>[]
}

export interface DatasetConfig {
  id_column?: string | null
  include_columns?: string[] | null
  imputation_k: number
  iqr_multiplier: number
  corr_threshold: number
  mi_k?: number | null
  pca_variance: number
  n_clusters: number
  domain_template: 'credit_card' | 'ecommerce_rfm' | 'saas' | 'generic'
}

export interface JobStatus {
  job_id: string
  dataset_id: string
  status: 'pending' | 'running' | 'completed' | 'failed' | 'cancelled'
  completed: number
  total: number
  current_step: string
  elapsed_ms: number
  error?: string | null
}

export interface DatasetSummary {
  dataset_id?: string
  row_count: number
  col_count: number
  columns: string[]
  is_synthetic?: boolean
  missing_values: Record<string, number>
  descriptive_stats: Record<string, {
    mean: number
    std: number
    median: number
    min: number
    max: number
    q1: number
    q3: number
  }>
  iqr_outlier_counts_before: Record<string, number>
  iqr_outlier_counts_after: Record<string, number>
  preprocessing_steps: string[]
}

export interface PrunedFeatureInfo {
  feature: string
  correlated_with: string
  correlation: number
}

export interface FeatureEngineering {
  dataset_id?: string
  original_features: string[]
  ds2_correlation_features: string[]
  ds2_removed_features: string[]
  ds2_pruned_details: PrunedFeatureInfo[]
  correlation_matrix: number[][]
  correlation_feature_names: string[]
  ds3_mutual_info_features: string[]
  ds3_mutual_info_scores: Record<string, number>
  ds3_method_note?: string
  ds4_pca_n_components: number
  ds4_pca_explained_variance_ratio: number[]
  ds4_pca_cumulative_variance: number[]
}

export interface ExperimentResult {
  dataset_name: string
  dataset_dim: string
  algorithm: string
  silhouette_score: number | null
  davies_bouldin_index: number | null
  calinski_harabasz_score: number | null
  composite_score?: number | null
  common_space_silhouette?: number | null
  common_space_dbi?: number | null
  common_space_chi?: number | null
  execution_time_ms: number
  cluster_count: number
  noise_percentage: number
  sampled_silhouette?: boolean
  silhouette_sample_size?: number | null
  status?: string
  reason?: string | null
  scatter_data: Array<{ pc1: number; pc2: number; cluster_label: number }>
  cluster_sizes?: Record<string | number, number>
}

export interface KSweepPoint {
  k: number
  inertia: number
  silhouette: number | null
  ch_score?: number | null
  dbi?: number | null
  composite_rank?: number | null
  min_cluster_size_pct?: number | null
}

export interface ExperimentsMatrix {
  dataset_id?: string
  config_hash?: string
  experiments: ExperimentResult[]
  winning_experiment: ExperimentResult
  selection_formula?: string
  k_sweep: KSweepPoint[]
  suggested_k: number
  suggested_k_reason?: string
}

export interface DistinguishingFeature {
  feature: string
  z_score: number
  cluster_mean: number
  global_mean: number
  interpretation?: string
}

export interface NBACampaign {
  title: string
  description: string
  expected_roi_lift: string
  channel: string
  estimated_net_lift_dollars?: number
  roi_multiple?: number
}

export interface PersonaAssumptions {
  avg_revenue_per_customer: number
  expected_uplift_pct: number
  campaign_cost_per_customer: number
}

export interface PersonaProfile {
  id?: number
  name: string
  percentage: number
  count: number
  color: string
  centroid_profile: Record<string, number>
  top_positive_features?: DistinguishingFeature[]
  top_negative_features?: DistinguishingFeature[]
  characteristics: string[]
  distinguishing_note?: string
  assumptions?: PersonaAssumptions
  nba_campaigns: NBACampaign[]
  real_unit_centroid?: Record<string, number>
  value_score?: number | null
  risk_score?: number | null
  value_metric_name?: string | null
  risk_formula_explanation?: string | null
}

export interface ClusterBoxPlotStat {
  min: number
  q1: number
  median: number
  q3: number
  max: number
}

export interface FeatureBoxPlot {
  feature: string
  clusters: Record<string, ClusterBoxPlotStat>
}

export interface GrowthPersonas {
  dataset_id?: string
  personas: PersonaProfile[]
  winning_model: string
  winning_silhouette?: number | null
  winning_dbi?: number | null
  domain_template?: string
  box_plots?: FeatureBoxPlot[]
  direction_map?: Record<string, string>
  global_real_means?: Record<string, number>
}

export interface PredictResponse {
  cluster_label: number
  persona_name: string
  confidence: number
  recommended_nba: NBACampaign
  distance_to_centroid: number
  top_traits: string[]
}

export interface SegmentScore {
  cluster_label: number
  persona_name: string
  value_score: number
  risk_score: number
  customer_count: number
  customer_pct: number
  value_metric_name: string
  formula_explanation: string
}

export interface CustomerLookupResult {
  customer_id: string
  dataset_id: string
  cluster_label: number
  persona_name: string
  confidence: number
  distance_to_centroid: number
  distinguishing_traits: string[]
  customer_features: Record<string, number>
  cluster_centroid: Record<string, number>
  differences_vs_cluster: Record<string, number>
}

export interface SummaryReport {
  dataset_id: string
  generated_at: string
  source_type: 'Template' | 'AI-written'
  summary_text: string
  key_findings: string[]
  aggregate_stats: Record<string, any>
}

export interface SavedRun {
  run_id: string
  dataset_id: string
  config_hash: string
  winning_algorithm: string
  winning_space: string
  n_clusters: number
  created_at: string
  config: Record<string, any>
  metrics: Record<string, any>
}

export interface DriftMigrationItem {
  original_group: string
  new_group: string
  count: number
  percent: number
}


export interface DriftCheckResult {
  original_dataset_id: string
  drift_dataset_name: string
  total_customers: number
  migration_matrix: number[][]
  group_names_original: string[]
  group_names_new: string[]
  migration_table: {
    original_group: string
    new_group: string
    count: number
    percent: number
  }[]
  stability_index: number
  drift_alert: string
}

async function fetchAPI<T>(endpoint: string, options?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE}${endpoint}`, options)
  if (!res.ok) {
    let errMsg = `API error: ${res.status}`
    try {
      const errObj = await res.json()
      if (errObj && errObj.detail) {
        errMsg = typeof errObj.detail === 'string' ? errObj.detail : JSON.stringify(errObj.detail)
      }
    } catch {}
    throw new Error(errMsg)
  }
  return res.json()
}

export const api = {
  health: () => fetchAPI<{ status: string; datasets_available: number; total_records: number; active_dataset: string; is_synthetic?: boolean }>('/api/health'),

  // Datasets
  listDatasets: () => fetchAPI<DatasetInfo[]>('/api/datasets'),
  getProfile: (datasetId: string) => fetchAPI<DatasetProfile>(`/api/datasets/${datasetId}/profile`),
  getConfig: (datasetId: string) => fetchAPI<DatasetConfig>(`/api/datasets/${datasetId}/config`),
  configureDataset: (datasetId: string, config: DatasetConfig) =>
    fetchAPI<{ dataset_id: string; config_hash: string; config: DatasetConfig; message: string }>(
      `/api/datasets/${datasetId}/configure`,
      {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(config),
      }
    ),
  uploadDataset: async (file: File): Promise<DatasetInfo> => {
    const formData = new FormData()
    formData.append('file', file)
    return fetchAPI<DatasetInfo>('/api/datasets/upload', {
      method: 'POST',
      body: formData,
    })
  },
  deleteDataset: (datasetId: string) =>
    fetchAPI<{ status: string; dataset_id: string }>(`/api/datasets/${datasetId}`, {
      method: 'DELETE',
    }),

  // Jobs
  runBenchmark: (datasetId: string) =>
    fetchAPI<{ job_id: string; dataset_id: string; status: string }>(`/api/datasets/${datasetId}/run`, {
      method: 'POST',
    }),
  getJobStatus: (jobId: string) => fetchAPI<JobStatus>(`/api/jobs/${jobId}`),
  cancelJob: (jobId: string) =>
    fetchAPI<{ job_id: string; cancelled: boolean }>(`/api/jobs/${jobId}/cancel`, {
      method: 'POST',
    }),

  // Dynamic Pipeline Outputs
  datasetSummary: (datasetId: string) => fetchAPI<DatasetSummary>(`/api/datasets/${datasetId}/summary`),
  featureEngineering: (datasetId: string) => fetchAPI<FeatureEngineering>(`/api/datasets/${datasetId}/feature-engineering`),
  experimentsMatrix: (datasetId: string) => fetchAPI<ExperimentsMatrix>(`/api/datasets/${datasetId}/experiments`),
  growthPersonas: (datasetId: string) => fetchAPI<GrowthPersonas>(`/api/datasets/${datasetId}/personas`),

  // Prediction & Operational
  predictSegment: (datasetId: string, features: Record<string, number>) =>
    fetchAPI<PredictResponse>(`/api/datasets/${datasetId}/predict-segment`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ features }),
    }),

  // New Intelligence Features
  getScores: (datasetId: string) => fetchAPI<SegmentScore[]>(`/api/datasets/${datasetId}/scores`),
  lookupCustomer: (datasetId: string, customerId: string) =>
    fetchAPI<CustomerLookupResult>(`/api/datasets/${datasetId}/customer/${customerId}`),
  getSummaryReport: (datasetId: string) => fetchAPI<SummaryReport>(`/api/datasets/${datasetId}/summary-report`),
  listSavedRuns: (datasetId: string) => fetchAPI<SavedRun[]>(`/api/datasets/${datasetId}/runs`),
  compareRuns: (run1Id: string, run2Id: string) =>
    fetchAPI<{ run1: SavedRun; run2: SavedRun; metrics_comparison: Record<string, any>; config_diff: Record<string, any> }>(
      `/api/runs/compare?run1_id=${encodeURIComponent(run1Id)}&run2_id=${encodeURIComponent(run2Id)}`
    ),
  driftCheck: async (datasetId: string, file: File): Promise<DriftCheckResult> => {
    const formData = new FormData()
    formData.append('file', file)
    return fetchAPI<DriftCheckResult>(`/api/datasets/${datasetId}/drift-check`, {
      method: 'POST',
      body: formData,
    })
  },

  exportSegmentsUrl: (datasetId: string) => `${API_BASE}/api/datasets/${datasetId}/export/segments.csv`,
  exportModelUrl: (datasetId: string) => `${API_BASE}/api/datasets/${datasetId}/model.joblib`,
  exportModelJsonUrl: (datasetId: string) => `${API_BASE}/api/datasets/${datasetId}/model.json`,
  reportHtmlUrl: (datasetId: string) => `${API_BASE}/api/datasets/${datasetId}/report.html`,

}
