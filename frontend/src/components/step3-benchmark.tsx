'use client'

import React, { useState } from 'react'
import {
  ExperimentsMatrix,
  ExperimentResult,
  JobStatus,
} from '@/lib/api'
import { formatNumber, formatMs } from '@/lib/utils'
import {
  Trophy,
  Loader2,
  HelpCircle,
  BarChart3,
  CheckCircle2,
  AlertTriangle,
  ArrowRight,
  Sparkles,
  Zap,
  Filter,
  ArrowUpDown,
  Star,
  Info,
  XCircle,
} from 'lucide-react'
import {
  ResponsiveContainer,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
} from 'recharts'

interface Step3BenchmarkProps {
  matrix: ExperimentsMatrix | null
  jobStatus: JobStatus | null
  isRunning: boolean
  onCancelJob: () => void
  onAdvanceToExplore: () => void
}

export function Step3Benchmark({
  matrix,
  jobStatus,
  isRunning,
  onCancelJob,
  onAdvanceToExplore,
}: Step3BenchmarkProps) {
  // Simple vs Expert View Toggle
  const [viewMode, setViewMode] = useState<'simple' | 'expert'>('simple')

  // Grouped Bar Chart metric switcher
  const [activeMetric, setActiveMetric] = useState<'silhouette' | 'dbi' | 'chi' | 'speed'>('silhouette')

  // Table filter & sort state
  const [filterAlg, setFilterAlg] = useState<string>('all')
  const [filterDataset, setFilterDataset] = useState<string>('all')
  const [sortField, setSortField] = useState<keyof ExperimentResult>('composite_score')
  const [sortAsc, setSortAsc] = useState<boolean>(false)

  // 1. Running Progress Panel State
  if (isRunning) {
    const completed = jobStatus?.completed || 0
    const total = jobStatus?.total || 16
    const pct = Math.round((completed / total) * 100)
    const currentStep = jobStatus?.current_step || 'Initializing 16-experiment benchmark matrix...'
    const elapsedMs = jobStatus?.elapsed_ms || 0

    return (
      <div className="min-h-[460px] flex items-center justify-center p-6">
        <div className="w-full max-w-lg bg-[var(--card)] border border-[var(--border)] rounded-2xl p-8 shadow-xl text-center space-y-6">
          <div className="w-14 h-14 rounded-2xl bg-accent-indigo/10 border border-accent-indigo/25 flex items-center justify-center text-accent-indigo mx-auto shadow-inner">
            <Loader2 className="w-7 h-7 animate-spin" />
          </div>

          <div className="space-y-2">
            <h3 className="text-lg font-bold text-[var(--text)]">
              Running 16-Experiment Benchmark
            </h3>
            <p className="text-xs text-[var(--text-muted)] font-mono">
              {currentStep}
            </p>
          </div>

          {/* Progress Bar */}
          <div className="space-y-2">
            <div className="w-full bg-[var(--border)] rounded-full h-3 overflow-hidden p-0.5">
              <div
                className="bg-accent-indigo h-full rounded-full transition-all duration-300"
                style={{ width: `${Math.max(pct, 5)}%` }}
              />
            </div>
            <div className="flex items-center justify-between text-xs text-[var(--text-muted)] font-mono">
              <span>{completed} of {total} completed ({pct}%)</span>
              <span className="metric-value">Elapsed: {formatMs(elapsedMs)}</span>
            </div>
          </div>

          <div className="pt-2">
            <button
              type="button"
              onClick={onCancelJob}
              className="px-4 py-2 rounded-lg border border-rose-500/30 text-rose-400 hover:bg-rose-500/10 text-xs font-medium transition-colors"
            >
              Cancel Benchmark Execution
            </button>
          </div>
        </div>
      </div>
    )
  }

  const experiments = matrix?.experiments || []
  const winner = matrix?.winning_experiment || experiments[0]

  // Metric min/max bounds for table cell styling
  const validSil = experiments.map((e) => e.silhouette_score).filter((v): v is number => v !== null)
  const validDbi = experiments.map((e) => e.davies_bouldin_index).filter((v): v is number => v !== null)
  const validChi = experiments.map((e) => e.calinski_harabasz_score).filter((v): v is number => v !== null)
  const validSpeed = experiments.map((e) => e.execution_time_ms)

  const maxSil = validSil.length ? Math.max(...validSil) : 1
  const minSil = validSil.length ? Math.min(...validSil) : 0
  const minDbi = validDbi.length ? Math.min(...validDbi) : 0 // lower is better
  const maxDbi = validDbi.length ? Math.max(...validDbi) : 1 // worst
  const maxChi = validChi.length ? Math.max(...validChi) : 1
  const minChi = validChi.length ? Math.min(...validChi) : 0
  const minSpeed = Math.min(...validSpeed)
  const maxSpeed = Math.max(...validSpeed)

  // Chart Data: Grouped by Dataset Representation
  const datasetKeys = ['DS1_Original', 'DS2_Correlation', 'DS3_MutualInfo', 'DS4_PCA95']
  const chartData = datasetKeys.map((dKey) => {
    const entry: Record<string, any> = {
      dataset: dKey.replace('DS', '').replace('_', ' '),
    }
    experiments
      .filter((e) => e.dataset_name === dKey)
      .forEach((e) => {
        let val: number | null = 0
        if (activeMetric === 'silhouette') val = e.silhouette_score
        else if (activeMetric === 'dbi') val = e.davies_bouldin_index
        else if (activeMetric === 'chi') val = e.calinski_harabasz_score
        else if (activeMetric === 'speed') val = e.execution_time_ms

        entry[e.algorithm] = val !== null ? val : 0
      })
    return entry
  })

  // Table filtering & sorting
  const filteredExperiments = experiments.filter((e) => {
    if (filterAlg !== 'all' && e.algorithm !== filterAlg) return false
    if (filterDataset !== 'all' && e.dataset_name !== filterDataset) return false
    return true
  })

  const sortedExperiments = [...filteredExperiments].sort((a, b) => {
    const aVal = a[sortField] ?? -9999
    const bVal = b[sortField] ?? -9999
    if (aVal < bVal) return sortAsc ? -1 : 1
    if (aVal > bVal) return sortAsc ? 1 : -1
    return 0
  })

  const handleSort = (field: keyof ExperimentResult) => {
    if (sortField === field) {
      setSortAsc(!sortAsc)
    } else {
      setSortField(field)
      setSortAsc(false)
    }
  }

  // Star rating calculator (1 to 5 stars) from composite score
  const getStarRating = (compositeScore?: number | null) => {
    if (compositeScore === undefined || compositeScore === null) return 3
    const stars = Math.min(5, Math.max(1, Math.round(compositeScore * 5)))
    return stars
  }

  return (
    <div className="space-y-8 animate-fadeIn">
      {/* 1. "Why this model won" Hero Card */}
      {winner && (
        <div className="p-6 rounded-2xl bg-[var(--card)] border border-accent-indigo/40 shadow-sm relative overflow-hidden">
          <div className="absolute top-0 right-0 w-96 h-96 bg-accent-indigo/5 rounded-full blur-3xl pointer-events-none" />

          <div className="flex flex-col sm:flex-row sm:items-start justify-between gap-4">
            <div className="space-y-2 max-w-2xl">
              <div className="flex items-center gap-2">
                <span className="text-xs font-semibold px-2.5 py-1 rounded-full bg-accent-indigo text-white flex items-center gap-1.5 shadow-sm">
                  <Trophy className="w-3.5 h-3.5" />
                  Winning Benchmark Model
                </span>
                <span className="text-xs font-mono text-[var(--text-muted)]">
                  {winner.algorithm} on {winner.dataset_name} ({winner.dataset_dim})
                </span>
              </div>

              <h2 className="text-lg font-bold text-[var(--text)] tracking-tight">
                {winner.dataset_name.includes('PCA')
                  ? 'PCA Orthogonal Representation + K-Means'
                  : `${winner.dataset_name} + ${winner.algorithm}`}{' '}
                discovered the highest-signal customer separation.
              </h2>

              <p className="text-xs text-[var(--text-muted)] leading-relaxed">
                Customers naturally resolved into {winner.cluster_count} tightly-clustered, well-separated groups with{' '}
                <strong className="text-[var(--text)]">{(winner.silhouette_score ?? 0).toFixed(3)} Silhouette</strong> and{' '}
                <strong className="text-[var(--text)]">{(winner.davies_bouldin_index ?? 0).toFixed(2)} DBI</strong>. Execution concluded in just{' '}
                <strong className="text-[var(--text)]">{winner.execution_time_ms.toFixed(1)}ms</strong> with 0% noise distortion.
              </p>
            </div>

            {/* View Mode Switcher Toggle */}
            <div className="flex items-center gap-1 bg-[var(--card-elevated)] p-1 rounded-lg border border-[var(--border)] shrink-0 self-start">
              <button
                type="button"
                onClick={() => setViewMode('simple')}
                className={`px-3 py-1 text-xs rounded font-medium transition-colors ${
                  viewMode === 'simple'
                    ? 'bg-accent-indigo text-white'
                    : 'text-[var(--text-muted)] hover:text-[var(--text)]'
                }`}
              >
                Simple View
              </button>
              <button
                type="button"
                onClick={() => setViewMode('expert')}
                className={`px-3 py-1 text-xs rounded font-medium transition-colors ${
                  viewMode === 'expert'
                    ? 'bg-accent-indigo text-white'
                    : 'text-[var(--text-muted)] hover:text-[var(--text)]'
                }`}
              >
                Expert View
              </button>
            </div>
          </div>
        </div>
      )}

      {/* 2. 4 KPI Cards with Explanatory (?) Tooltips */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
        {/* Silhouette Score */}
        <div className="p-4 rounded-xl bg-[var(--card)] border border-[var(--border)] shadow-sm space-y-1.5 relative group">
          <div className="flex items-center justify-between text-xs text-[var(--text-muted)]">
            <span className="font-medium">Silhouette Score</span>
            <div className="flex items-center gap-1">
              <span className="text-[10px] text-accent-emerald font-semibold">Higher is better</span>
              <HelpCircle className="w-3.5 h-3.5 cursor-help" />
            </div>
          </div>
          <div className="text-2xl font-bold metric-value text-accent-emerald">
            {winner?.silhouette_score !== null ? winner?.silhouette_score?.toFixed(3) : 'N/A'}
          </div>
          <span className="text-[11px] text-[var(--text-muted)] block">
            Cluster separation & distinction
          </span>

          <div className="absolute -top-12 left-1/2 -translate-x-1/2 hidden group-hover:block z-30 pointer-events-none w-56 bg-zinc-950 text-zinc-100 text-[11px] p-2 rounded shadow-xl border border-zinc-800">
            Measures how well each customer fits their own segment compared to other segments (-1 to +1).
          </div>
        </div>

        {/* Davies-Bouldin Index */}
        <div className="p-4 rounded-xl bg-[var(--card)] border border-[var(--border)] shadow-sm space-y-1.5 relative group">
          <div className="flex items-center justify-between text-xs text-[var(--text-muted)]">
            <span className="font-medium">Davies-Bouldin Index</span>
            <div className="flex items-center gap-1">
              <span className="text-[10px] text-accent-emerald font-semibold">Lower is better</span>
              <HelpCircle className="w-3.5 h-3.5 cursor-help" />
            </div>
          </div>
          <div className="text-2xl font-bold metric-value text-[var(--text)]">
            {winner?.davies_bouldin_index !== null ? winner?.davies_bouldin_index?.toFixed(2) : 'N/A'}
          </div>
          <span className="text-[11px] text-[var(--text-muted)] block">
            Within-cluster compactness
          </span>

          <div className="absolute -top-12 left-1/2 -translate-x-1/2 hidden group-hover:block z-30 pointer-events-none w-56 bg-zinc-950 text-zinc-100 text-[11px] p-2 rounded shadow-xl border border-zinc-800">
            Ratio of internal cluster spread to separation distance. Lower numbers indicate tight, concentrated groups.
          </div>
        </div>

        {/* Calinski-Harabasz Index */}
        <div className="p-4 rounded-xl bg-[var(--card)] border border-[var(--border)] shadow-sm space-y-1.5 relative group">
          <div className="flex items-center justify-between text-xs text-[var(--text-muted)]">
            <span className="font-medium">Calinski-Harabasz</span>
            <div className="flex items-center gap-1">
              <span className="text-[10px] text-accent-emerald font-semibold">Higher is better</span>
              <HelpCircle className="w-3.5 h-3.5 cursor-help" />
            </div>
          </div>
          <div className="text-2xl font-bold metric-value text-[var(--text)]">
            {winner?.calinski_harabasz_score !== null ? formatNumber(Math.round(winner?.calinski_harabasz_score ?? 0)) : 'N/A'}
          </div>
          <span className="text-[11px] text-[var(--text-muted)] block">
            Between-group variance ratio
          </span>

          <div className="absolute -top-12 left-1/2 -translate-x-1/2 hidden group-hover:block z-30 pointer-events-none w-56 bg-zinc-950 text-zinc-100 text-[11px] p-2 rounded shadow-xl border border-zinc-800">
            Evaluates how much stronger between-group differences are compared to internal background variance.
          </div>
        </div>

        {/* Execution Speed */}
        <div className="p-4 rounded-xl bg-[var(--card)] border border-[var(--border)] shadow-sm space-y-1.5 relative group">
          <div className="flex items-center justify-between text-xs text-[var(--text-muted)]">
            <span className="font-medium">Execution Speed</span>
            <div className="flex items-center gap-1">
              <span className="text-[10px] text-accent-emerald font-semibold">Lower is better</span>
              <HelpCircle className="w-3.5 h-3.5 cursor-help" />
            </div>
          </div>
          <div className="text-2xl font-bold metric-value text-accent-indigo">
            {winner?.execution_time_ms ? `${winner.execution_time_ms.toFixed(1)}ms` : '0ms'}
          </div>
          <span className="text-[11px] text-[var(--text-muted)] block">
            Wall-clock inference latency
          </span>

          <div className="absolute -top-12 left-1/2 -translate-x-1/2 hidden group-hover:block z-30 pointer-events-none w-56 bg-zinc-950 text-zinc-100 text-[11px] p-2 rounded shadow-xl border border-zinc-800">
            Real time taken in milliseconds to train the model and assign cluster labels to all records.
          </div>
        </div>
      </div>

      {/* 3. Grouped Bar Chart Comparing ALL 4 Algorithms */}
      <div className="bg-[var(--card)] border border-[var(--border)] rounded-xl p-6 shadow-sm space-y-4">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
          <div>
            <h3 className="text-sm font-semibold text-[var(--text)] flex items-center gap-2">
              <BarChart3 className="w-4 h-4 text-accent-indigo" />
              Algorithm Comparison Across Feature Representations
            </h3>
            <p className="text-xs text-[var(--text-muted)] mt-0.5">
              Side-by-side performance across Original (DS1), Correlation (DS2), Mutual Info (DS3), and PCA (DS4).
            </p>
          </div>

          {/* Metric Switcher Tabs */}
          <div className="flex items-center gap-1 bg-[var(--card-elevated)] p-1 rounded-lg border border-[var(--border)] text-xs font-medium">
            <button
              type="button"
              onClick={() => setActiveMetric('silhouette')}
              className={`px-2.5 py-1 rounded transition-colors ${
                activeMetric === 'silhouette'
                  ? 'bg-accent-indigo text-white font-semibold'
                  : 'text-[var(--text-muted)] hover:text-[var(--text)]'
              }`}
            >
              Silhouette
            </button>
            <button
              type="button"
              onClick={() => setActiveMetric('dbi')}
              className={`px-2.5 py-1 rounded transition-colors ${
                activeMetric === 'dbi'
                  ? 'bg-accent-indigo text-white font-semibold'
                  : 'text-[var(--text-muted)] hover:text-[var(--text)]'
              }`}
            >
              DBI
            </button>
            <button
              type="button"
              onClick={() => setActiveMetric('chi')}
              className={`px-2.5 py-1 rounded transition-colors ${
                activeMetric === 'chi'
                  ? 'bg-accent-indigo text-white font-semibold'
                  : 'text-[var(--text-muted)] hover:text-[var(--text)]'
              }`}
            >
              CHI
            </button>
            <button
              type="button"
              onClick={() => setActiveMetric('speed')}
              className={`px-2.5 py-1 rounded transition-colors ${
                activeMetric === 'speed'
                  ? 'bg-accent-indigo text-white font-semibold'
                  : 'text-[var(--text-muted)] hover:text-[var(--text)]'
              }`}
            >
              Speed (ms)
            </button>
          </div>
        </div>

        <div className="h-72 w-full pt-2">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={chartData} margin={{ top: 10, right: 20, left: 0, bottom: 5 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#27272A" opacity={0.5} />
              <XAxis dataKey="dataset" stroke="#71717A" tick={{ fontSize: 11 }} />
              <YAxis stroke="#71717A" tick={{ fontSize: 11 }} />
              <Tooltip
                contentStyle={{
                  backgroundColor: '#121215',
                  borderColor: '#27272A',
                  borderRadius: '8px',
                  fontSize: '12px',
                  color: '#FAFAFA',
                }}
                formatter={(val: any, name: string) => [
                  activeMetric === 'speed' ? `${val} ms` : typeof val === 'number' ? val.toFixed(3) : val,
                  name,
                ]}
              />
              <Legend
                wrapperStyle={{ fontSize: '11px', paddingTop: '10px' }}
                iconType="circle"
              />
              <Bar dataKey="KMeans" fill="#6366F1" radius={[4, 4, 0, 0]} />
              <Bar dataKey="Agglomerative" fill="#10B981" radius={[4, 4, 0, 0]} />
              <Bar dataKey="DBSCAN" fill="#F59E0B" radius={[4, 4, 0, 0]} />
              <Bar dataKey="GaussianMixture" fill="#8B5CF6" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* 4. 16-Row Interactive Matrix Table */}
      <div className="bg-[var(--card)] border border-[var(--border)] rounded-xl p-6 shadow-sm space-y-4">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
          <div>
            <h3 className="text-sm font-semibold text-[var(--text)] flex items-center gap-2">
              <Filter className="w-4 h-4 text-accent-indigo" />
              16-Experiment Benchmark Matrix
            </h3>
            <p className="text-xs text-[var(--text-muted)] mt-0.5">
              Comprehensive performance log across all 16 runs. Sort by clicking column headers.
            </p>
          </div>

          {/* Table Filters */}
          <div className="flex items-center gap-2 text-xs">
            <select
              value={filterAlg}
              onChange={(e) => setFilterAlg(e.target.value)}
              className="bg-[var(--card-elevated)] border border-[var(--border)] rounded px-2.5 py-1 text-[var(--text)]"
            >
              <option value="all">All Algorithms</option>
              <option value="KMeans">K-Means</option>
              <option value="Agglomerative">Agglomerative</option>
              <option value="DBSCAN">DBSCAN</option>
              <option value="GaussianMixture">Gaussian Mixture</option>
            </select>

            <select
              value={filterDataset}
              onChange={(e) => setFilterDataset(e.target.value)}
              className="bg-[var(--card-elevated)] border border-[var(--border)] rounded px-2.5 py-1 text-[var(--text)]"
            >
              <option value="all">All Representations</option>
              <option value="DS1_Original">DS1: Original</option>
              <option value="DS2_Correlation">DS2: Correlation</option>
              <option value="DS3_MutualInfo">DS3: Mutual Info</option>
              <option value="DS4_PCA95">DS4: PCA 95%</option>
            </select>
          </div>
        </div>

        <div className="overflow-x-auto border border-[var(--border)] rounded-lg">
          <table className="w-full text-xs text-left">
            <thead className="bg-[var(--card-elevated)] text-[var(--text-muted)] border-b border-[var(--border)] font-mono uppercase text-[10px] tracking-wider">
              <tr>
                <th
                  onClick={() => handleSort('algorithm')}
                  className="px-3 py-2.5 font-semibold cursor-pointer hover:text-[var(--text)]"
                >
                  <div className="flex items-center gap-1">
                    Algorithm <ArrowUpDown className="w-3 h-3" />
                  </div>
                </th>
                <th
                  onClick={() => handleSort('dataset_name')}
                  className="px-3 py-2.5 font-semibold cursor-pointer hover:text-[var(--text)]"
                >
                  <div className="flex items-center gap-1">
                    Dataset Space <ArrowUpDown className="w-3 h-3" />
                  </div>
                </th>
                <th
                  onClick={() => handleSort('silhouette_score')}
                  className="px-3 py-2.5 font-semibold cursor-pointer hover:text-[var(--text)] text-right"
                >
                  <div className="flex items-center justify-end gap-1">
                    Silhouette <ArrowUpDown className="w-3 h-3" />
                  </div>
                </th>

                {viewMode === 'expert' && (
                  <>
                    <th
                      onClick={() => handleSort('davies_bouldin_index')}
                      className="px-3 py-2.5 font-semibold cursor-pointer hover:text-[var(--text)] text-right"
                    >
                      <div className="flex items-center justify-end gap-1">
                        DBI <ArrowUpDown className="w-3 h-3" />
                      </div>
                    </th>
                    <th
                      onClick={() => handleSort('calinski_harabasz_score')}
                      className="px-3 py-2.5 font-semibold cursor-pointer hover:text-[var(--text)] text-right"
                    >
                      <div className="flex items-center justify-end gap-1">
                        CHI <ArrowUpDown className="w-3 h-3" />
                      </div>
                    </th>
                  </>
                )}

                <th
                  onClick={() => handleSort('execution_time_ms')}
                  className="px-3 py-2.5 font-semibold cursor-pointer hover:text-[var(--text)] text-right"
                >
                  <div className="flex items-center justify-end gap-1">
                    Speed (ms) <ArrowUpDown className="w-3 h-3" />
                  </div>
                </th>

                <th className="px-3 py-2.5 font-semibold text-center">
                  {viewMode === 'simple' ? 'Overall Quality' : 'Clusters'}
                </th>
              </tr>
            </thead>

            <tbody className="divide-y divide-[var(--border)] font-mono text-[11px]">
              {sortedExperiments.map((exp, idx) => {
                const isWinningRow =
                  exp.algorithm === winner?.algorithm && exp.dataset_name === winner?.dataset_name

                const isBestSil = exp.silhouette_score !== null && exp.silhouette_score === maxSil
                const isWorstSil = exp.silhouette_score !== null && exp.silhouette_score === minSil
                const isBestDbi = exp.davies_bouldin_index !== null && exp.davies_bouldin_index === minDbi
                const isWorstDbi = exp.davies_bouldin_index !== null && exp.davies_bouldin_index === maxDbi

                return (
                  <tr
                    key={`${exp.algorithm}-${exp.dataset_name}-${idx}`}
                    className={`transition-colors ${
                      isWinningRow
                        ? 'bg-accent-indigo/10 border-l-2 border-accent-indigo font-medium'
                        : 'hover:bg-[var(--card-elevated)]/50'
                    }`}
                  >
                    {/* Algorithm */}
                    <td className="px-3 py-2.5 whitespace-nowrap text-[var(--text)] font-sans flex items-center gap-2">
                      <span className="font-semibold">{exp.algorithm}</span>
                      {isWinningRow && (
                        <span className="px-1.5 py-0.2 rounded text-[10px] bg-accent-indigo text-white font-bold tracking-wider">
                          WINNER
                        </span>
                      )}
                    </td>

                    {/* Dataset Space */}
                    <td className="px-3 py-2.5 whitespace-nowrap text-[var(--text-muted)]">
                      {exp.dataset_name} ({exp.dataset_dim})
                    </td>

                    {/* Silhouette */}
                    <td className="px-3 py-2.5 text-right whitespace-nowrap">
                      {exp.silhouette_score !== null ? (
                        <span
                          className={
                            isBestSil
                              ? 'text-accent-emerald font-bold'
                              : isWorstSil
                              ? 'text-accent-rose'
                              : 'text-[var(--text)]'
                          }
                        >
                          {exp.silhouette_score.toFixed(3)}
                        </span>
                      ) : (
                        <span
                          className="text-[var(--text-muted)] italic cursor-help"
                          title={exp.reason || 'DBSCAN generated single cluster or excessive noise'}
                        >
                          N/A
                        </span>
                      )}
                    </td>

                    {/* Expert: DBI */}
                    {viewMode === 'expert' && (
                      <td className="px-3 py-2.5 text-right whitespace-nowrap">
                        {exp.davies_bouldin_index !== null ? (
                          <span
                            className={
                              isBestDbi
                                ? 'text-accent-emerald font-bold'
                                : isWorstDbi
                                ? 'text-accent-rose'
                                : 'text-[var(--text)]'
                            }
                          >
                            {exp.davies_bouldin_index.toFixed(2)}
                          </span>
                        ) : (
                          <span className="text-[var(--text-muted)] italic">N/A</span>
                        )}
                      </td>
                    )}

                    {/* Expert: CHI */}
                    {viewMode === 'expert' && (
                      <td className="px-3 py-2.5 text-right whitespace-nowrap">
                        {exp.calinski_harabasz_score !== null ? (
                          <span className="text-[var(--text)]">
                            {formatNumber(Math.round(exp.calinski_harabasz_score))}
                          </span>
                        ) : (
                          <span className="text-[var(--text-muted)] italic">N/A</span>
                        )}
                      </td>
                    )}

                    {/* Speed */}
                    <td className="px-3 py-2.5 text-right whitespace-nowrap text-[var(--text-muted)]">
                      {exp.execution_time_ms.toFixed(1)}ms
                    </td>

                    {/* Simple (Stars) or Expert (Cluster counts) */}
                    <td className="px-3 py-2.5 text-center whitespace-nowrap">
                      {viewMode === 'simple' ? (
                        <div className="flex items-center justify-center gap-0.5 text-accent-amber">
                          {[1, 2, 3, 4, 5].map((star) => (
                            <Star
                              key={star}
                              className={`w-3 h-3 ${
                                star <= getStarRating(exp.composite_score)
                                  ? 'fill-accent-amber text-accent-amber'
                                  : 'text-[var(--border)]'
                              }`}
                            />
                          ))}
                        </div>
                      ) : (
                        <span className="text-[var(--text)]">
                          {exp.cluster_count} {exp.noise_percentage > 0 ? `(${exp.noise_percentage}% noise)` : ''}
                        </span>
                      )}
                    </td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        </div>
      </div>

      {/* Primary Action Button */}
      <div className="flex justify-end pt-2">
        <button
          type="button"
          onClick={onAdvanceToExplore}
          className="px-6 py-2.5 rounded-lg bg-accent-indigo hover:bg-accent-indigo/90 text-white font-medium text-xs shadow-sm transition-all flex items-center gap-2 cursor-pointer"
        >
          Explore customer segments
          <ArrowRight className="w-4 h-4" />
        </button>
      </div>
    </div>
  )
}
