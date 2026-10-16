'use client'

import React, { useState } from 'react'
import {
  DatasetSummary,
  FeatureEngineering,
  DatasetConfig,
} from '@/lib/api'
import { formatNumber } from '@/lib/utils'
import {
  Sliders,
  Filter,
  CheckCircle2,
  TrendingUp,
  Layers,
  ArrowRight,
  Info,
  Sparkles,
  BarChart2,
  AlertTriangle,
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
  AreaChart,
  Area,
} from 'recharts'

interface Step2PrepareProps {
  summary: DatasetSummary | null
  features: FeatureEngineering | null
  config: DatasetConfig | null
  onAdvanceToBenchmark: () => void
}

export function Step2Prepare({
  summary,
  features,
  config,
  onAdvanceToBenchmark,
}: Step2PrepareProps) {
  // Live threshold slider state (0.50 to 0.95, default 0.85)
  const [corrThreshold, setCorrThreshold] = useState<number>(config?.corr_threshold || 0.85)
  const [activeCellTooltip, setActiveCellTooltip] = useState<{
    f1: string
    f2: string
    r: number
    x: number
    y: number
  } | null>(null)

  // Missing values count
  const missingCount = summary?.missing_values
    ? Object.values(summary.missing_values).reduce((a, b) => a + b, 0)
    : 0

  // Total outliers capped
  const totalOutliersBefore = summary?.iqr_outlier_counts_before
    ? Object.values(summary.iqr_outlier_counts_before).reduce((a, b) => a + b, 0)
    : 0

  // IQR Horizontal Bar Chart Data: FULL feature names
  const iqrFeatures = summary?.iqr_outlier_counts_before
    ? Object.keys(summary.iqr_outlier_counts_before)
    : []

  const iqrChartData = iqrFeatures.map((feat) => ({
    feature: feat,
    before: summary?.iqr_outlier_counts_before?.[feat] || 0,
    after: summary?.iqr_outlier_counts_after?.[feat] || 0,
  }))

  // Correlation Matrix Data (Lower Triangle Only)
  const corrMatrix = features?.correlation_matrix || []
  const corrFeatureNames = features?.correlation_feature_names || features?.original_features || []
  const nFeatures = corrFeatureNames.length

  // Live calculation of features pruned at current corrThreshold
  const redundantPairs: Array<{ f1: string; f2: string; r: number }> = []
  const livePrunedFeatures = new Set<string>()

  for (let i = 0; i < nFeatures; i++) {
    for (let j = 0; j < i; j++) {
      const r = corrMatrix[i]?.[j]
      if (r !== undefined && Math.abs(r) > corrThreshold) {
        redundantPairs.push({
          f1: corrFeatureNames[i],
          f2: corrFeatureNames[j],
          r,
        })
        livePrunedFeatures.add(corrFeatureNames[i])
      }
    }
  }

  const liveKeptCount = Math.max(3, nFeatures - livePrunedFeatures.size)

  // Color interpolation for diverging heatmap:
  // -1 = Deep Blue (rgb(59, 130, 246)), 0 = Dark / Neutral (rgb(39, 39, 42)), +1 = Electric Indigo / Red (rgb(239, 68, 68))
  const getCellBg = (r: number) => {
    if (r > 0) {
      // 0 to 1 -> interpolate to red/indigo
      const t = Math.min(r, 1)
      return `rgba(239, 68, 68, ${Math.max(0.08, t * 0.85)})`
    } else {
      // 0 to -1 -> interpolate to blue
      const t = Math.min(Math.abs(r), 1)
      return `rgba(59, 130, 246, ${Math.max(0.08, t * 0.85)})`
    }
  }

  // PCA Cumulative Chart Data
  const pcaCumulative = features?.ds4_pca_cumulative_variance || []
  const pcaExplained = features?.ds4_pca_explained_variance_ratio || []
  const pcaChartData = pcaCumulative.map((cumVar, idx) => ({
    component: `PC${idx + 1}`,
    cumulative: Number((cumVar * 100).toFixed(1)),
    explained: Number(((pcaExplained[idx] || 0) * 100).toFixed(1)),
  }))

  return (
    <div className="space-y-8 animate-fadeIn">
      {/* 1. Three Plain-Language Summary Cards */}
      <div>
        <h2 className="text-xs font-semibold uppercase tracking-wider text-[var(--text-muted)] mb-3 flex items-center gap-1.5">
          <CheckCircle2 className="w-3.5 h-3.5 text-accent-emerald" />
          Data Preparation Overview
        </h2>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div className="p-5 rounded-xl bg-[var(--card)] border border-[var(--border)] shadow-sm space-y-2">
            <div className="flex items-center justify-between text-xs text-[var(--text-muted)]">
              <span className="font-medium">1. Missing Imputation</span>
              <span className="text-[10px] px-2 py-0.5 rounded-full bg-accent-indigo/10 text-accent-indigo border border-accent-indigo/25 font-semibold">
                KNN (k=5)
              </span>
            </div>
            <div className="text-2xl font-bold metric-value text-[var(--text)]">
              Filled {formatNumber(missingCount)} missing values
            </div>
            <p className="text-xs text-[var(--text-muted)] leading-relaxed">
              Multi-variate KNN Imputation analyzed patterns across the nearest 5 customer profiles to fill gaps without skewing natural distribution means.
            </p>
          </div>

          <div className="p-5 rounded-xl bg-[var(--card)] border border-[var(--border)] shadow-sm space-y-2">
            <div className="flex items-center justify-between text-xs text-[var(--text-muted)]">
              <span className="font-medium">2. Distribution Capping</span>
              <span className="text-[10px] px-2 py-0.5 rounded-full bg-accent-amber/10 text-accent-amber border border-accent-amber/25 font-semibold">
                1.5x Tukey IQR
              </span>
            </div>
            <div className="text-2xl font-bold metric-value text-accent-amber">
              Capped {formatNumber(totalOutliersBefore)} extreme values
            </div>
            <p className="text-xs text-[var(--text-muted)] leading-relaxed">
              Tukey IQR Winsorization safely clipped extreme transaction spikes to 1.5x IQR boundaries without discarding valuable real customer records.
            </p>
          </div>

          <div className="p-5 rounded-xl bg-[var(--card)] border border-[var(--border)] shadow-sm space-y-2">
            <div className="flex items-center justify-between text-xs text-[var(--text-muted)]">
              <span className="font-medium">3. Feature Normalization</span>
              <span className="text-[10px] px-2 py-0.5 rounded-full bg-accent-emerald/10 text-accent-emerald border border-accent-emerald/25 font-semibold">
                Z-Score
              </span>
            </div>
            <div className="text-2xl font-bold metric-value text-accent-emerald">
              Standardized all {nFeatures} features
            </div>
            <p className="text-xs text-[var(--text-muted)] leading-relaxed">
              StandardScaler scaled each feature to zero mean and unit variance (Z = (x - &mu;) / &sigma;), ensuring high-dollar metrics do not overpower frequencies.
            </p>
          </div>
        </div>
      </div>

      {/* 2. IQR Outlier Chart: Horizontal bars, FULL feature names */}
      <div className="bg-[var(--card)] border border-[var(--border)] rounded-xl p-6 shadow-sm space-y-4">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2">
          <div>
            <h3 className="text-sm font-semibold text-[var(--text)] flex items-center gap-2">
              <BarChart2 className="w-4 h-4 text-accent-amber" />
              Tukey IQR Outlier Capping (Before vs. After)
            </h3>
            <p className="text-xs text-[var(--text-muted)] mt-0.5">
              Comparison of anomalous distribution tails detected before capping vs zero post-clipping distortion.
            </p>
          </div>
          <div className="flex items-center gap-4 text-xs font-mono">
            <span className="flex items-center gap-1.5 text-[var(--text-muted)]">
              <span className="w-3 h-3 rounded bg-amber-500 inline-block" />
              Before Capping
            </span>
            <span className="flex items-center gap-1.5 text-[var(--text-muted)]">
              <span className="w-3 h-3 rounded bg-emerald-500 inline-block" />
              After Capping (0 remaining)
            </span>
          </div>
        </div>

        {/* Scrollable container for long feature sets */}
        <div className="overflow-y-auto max-h-[420px] pr-2">
          <div style={{ height: Math.max(300, iqrChartData.length * 32) }}>
            <ResponsiveContainer width="100%" height="100%">
              <BarChart
                layout="vertical"
                data={iqrChartData}
                margin={{ top: 10, right: 30, left: 160, bottom: 10 }}
              >
                <CartesianGrid strokeDasharray="3 3" stroke="#27272A" opacity={0.5} horizontal={false} />
                <XAxis type="number" stroke="#71717A" tick={{ fontSize: 11 }} />
                <YAxis
                  type="category"
                  dataKey="feature"
                  stroke="#71717A"
                  tick={{ fontSize: 11 }}
                  width={150}
                />
                <Tooltip
                  contentStyle={{
                    backgroundColor: '#121215',
                    borderColor: '#27272A',
                    borderRadius: '8px',
                    fontSize: '12px',
                    color: '#FAFAFA',
                  }}
                  formatter={(val: any, name: string) => [
                    `${val} outliers`,
                    name === 'before' ? 'Before Winsorization' : 'After Winsorization',
                  ]}
                />
                <Bar dataKey="before" name="before" fill="#F59E0B" radius={[0, 4, 4, 0]} barSize={10} />
                <Bar dataKey="after" name="after" fill="#10B981" radius={[0, 4, 4, 0]} barSize={10} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>

      {/* 3. Lower Triangle Correlation Heatmap with Live Slider */}
      <div className="bg-[var(--card)] border border-[var(--border)] rounded-xl p-6 shadow-sm space-y-5">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 pb-4 border-b border-[var(--border)]">
          <div>
            <h3 className="text-sm font-semibold text-[var(--text)] flex items-center gap-2">
              <Sliders className="w-4 h-4 text-accent-indigo" />
              Correlation Heatmap (Lower Triangle Only)
            </h3>
            <p className="text-xs text-[var(--text-muted)] mt-0.5">
              Identifies multicollinear metrics with Pearson |r| &gt; {corrThreshold.toFixed(2)}. Highlighted cells mark redundant features pruned in DS2.
            </p>
          </div>

          {/* Interactive Threshold Slider */}
          <div className="flex items-center gap-3 bg-[var(--card-elevated)] p-2 rounded-lg border border-[var(--border)]">
            <span className="text-xs text-[var(--text-muted)] whitespace-nowrap">Filter Threshold:</span>
            <input
              type="range"
              min="0.50"
              max="0.95"
              step="0.05"
              value={corrThreshold}
              onChange={(e) => setCorrThreshold(parseFloat(e.target.value))}
              className="w-28 accent-accent-indigo cursor-pointer"
            />
            <span className="text-xs font-mono font-bold text-accent-indigo w-10 text-right">
              |r| &gt; {corrThreshold.toFixed(2)}
            </span>
          </div>
        </div>

        {/* Heatmap Grid (Lower Triangle only: row i > col j) */}
        <div className="overflow-x-auto pb-2">
          <div className="inline-block min-w-full">
            <table className="text-[11px] border-collapse font-mono">
              <thead>
                <tr>
                  <th className="p-1.5 text-left text-[var(--text-muted)] font-normal max-w-[120px]">
                    Features
                  </th>
                  {corrFeatureNames.slice(0, nFeatures - 1).map((fName) => (
                    <th
                      key={fName}
                      className="p-1.5 font-normal text-[var(--text-muted)] text-center max-w-[80px] truncate"
                      title={fName}
                    >
                      {fName.length > 9 ? fName.substring(0, 8) + '..' : fName}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {corrFeatureNames.slice(1).map((rowFeat, rowIdx) => {
                  const actualRowIdx = rowIdx + 1
                  return (
                    <tr key={rowFeat}>
                      <td
                        className="p-1.5 font-medium text-[var(--text)] text-left whitespace-nowrap pr-3 truncate max-w-[140px]"
                        title={rowFeat}
                      >
                        {rowFeat.length > 14 ? rowFeat.substring(0, 13) + '..' : rowFeat}
                      </td>

                      {corrFeatureNames.slice(0, nFeatures - 1).map((colFeat, colIdx) => {
                        if (colIdx >= actualRowIdx) {
                          // Upper triangle or diagonal: blank
                          return <td key={colFeat} className="p-1.5" />
                        }

                        const rVal = corrMatrix[actualRowIdx]?.[colIdx] ?? 0.0
                        const isRedundant = Math.abs(rVal) > corrThreshold

                        return (
                          <td
                            key={colFeat}
                            onMouseEnter={(e) => {
                              const rect = e.currentTarget.getBoundingClientRect()
                              setActiveCellTooltip({
                                f1: rowFeat,
                                f2: colFeat,
                                r: rVal,
                                x: rect.x + rect.width / 2,
                                y: rect.y,
                              })
                            }}
                            onMouseLeave={() => setActiveCellTooltip(null)}
                            style={{ backgroundColor: getCellBg(rVal) }}
                            className={`p-1.5 text-center transition-all cursor-pointer select-none rounded-[2px] ${
                              isRedundant
                                ? 'ring-2 ring-accent-amber font-bold text-white shadow-sm'
                                : 'text-[var(--text)]'
                            }`}
                          >
                            {rVal.toFixed(2)}
                          </td>
                        )
                      })}
                    </tr>
                  )
                })}
              </tbody>
            </table>
          </div>
        </div>

        {/* Floating Tooltip for Heatmap */}
        {activeCellTooltip && (
          <div className="p-2.5 rounded-lg bg-zinc-950 text-zinc-100 text-xs border border-zinc-800 shadow-xl max-w-xs space-y-1">
            <div className="font-semibold text-accent-indigo">
              {activeCellTooltip.f1} &times; {activeCellTooltip.f2}
            </div>
            <div className="font-mono text-sm flex items-center justify-between">
              <span>Pearson Correlation (r):</span>
              <span className={`font-bold ${Math.abs(activeCellTooltip.r) > corrThreshold ? 'text-accent-amber' : 'text-zinc-200'}`}>
                {activeCellTooltip.r > 0 ? `+${activeCellTooltip.r.toFixed(3)}` : activeCellTooltip.r.toFixed(3)}
              </span>
            </div>
            {Math.abs(activeCellTooltip.r) > corrThreshold && (
              <span className="text-[10px] text-accent-amber block font-sans">
                Exceeds threshold |r| &gt; {corrThreshold.toFixed(2)}: marked redundant for DS2 filter.
              </span>
            )}
          </div>
        )}

        {/* Heatmap Diverging Gradient Legend */}
        <div className="flex flex-wrap items-center justify-between gap-3 pt-3 border-t border-[var(--border)] text-xs text-[var(--text-muted)]">
          <div className="flex items-center gap-2">
            <span>-1.00 (Inverse)</span>
            <div className="h-3 w-40 rounded bg-gradient-to-r from-blue-500 via-zinc-800 to-rose-500 border border-[var(--border)]" />
            <span>+1.00 (Collinear)</span>
          </div>

          <div className="font-mono">
            Keeps <span className="font-bold text-accent-emerald">{liveKeptCount}</span> of {nFeatures} features at |r| &le; {corrThreshold.toFixed(2)}
          </div>
        </div>
      </div>

      {/* 4. Feature Reduction Inspector (3 Cards) */}
      <div>
        <h2 className="text-xs font-semibold uppercase tracking-wider text-[var(--text-muted)] mb-3 flex items-center gap-1.5">
          <Layers className="w-3.5 h-3.5 text-accent-indigo" />
          Feature Engineering Spaces (Phases 3–4)
        </h2>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          {/* Card 1: Correlation Filter */}
          <div className="p-5 rounded-xl bg-[var(--card)] border border-[var(--border)] shadow-sm space-y-3 flex flex-col justify-between">
            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <span className="text-xs font-semibold text-[var(--text)]">DS2: Correlation Filter</span>
                <span className="text-[11px] px-2 py-0.5 rounded-full bg-accent-indigo/10 text-accent-indigo font-mono font-medium">
                  Kept {features?.ds2_correlation_features.length || 0} of {nFeatures}
                </span>
              </div>
              <p className="text-xs text-[var(--text-muted)] leading-relaxed">
                Removes collinear metrics that move in lockstep with another feature to eliminate redundant variance.
              </p>

              {features?.ds2_pruned_details && features.ds2_pruned_details.length > 0 ? (
                <div className="mt-3 pt-3 border-t border-[var(--border)] space-y-1.5">
                  <span className="text-[10px] uppercase font-bold text-accent-amber block">
                    Pruned Redundancies:
                  </span>
                  {features.ds2_pruned_details.map((p) => (
                    <div key={p.feature} className="text-xs font-mono bg-[var(--card-elevated)] p-1.5 rounded border border-[var(--border)] flex items-center justify-between">
                      <span className="text-rose-400 truncate">{p.feature}</span>
                      <span className="text-[var(--text-muted)] text-[10px]">
                        partner: {p.correlated_with} (|r|={p.correlation.toFixed(2)})
                      </span>
                    </div>
                  ))}
                </div>
              ) : (
                <div className="text-xs text-[var(--text-muted)] italic pt-2">
                  No features exceeded the collinearity threshold.
                </div>
              )}
            </div>
          </div>

          {/* Card 2: Mutual Information Selection */}
          <div className="p-5 rounded-xl bg-[var(--card)] border border-[var(--border)] shadow-sm space-y-3 flex flex-col justify-between">
            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <span className="text-xs font-semibold text-[var(--text)]">DS3: Mutual Information</span>
                <span className="text-[11px] px-2 py-0.5 rounded-full bg-accent-emerald/10 text-accent-emerald font-mono font-medium">
                  Kept top {features?.ds3_mutual_info_features.length || 8}
                </span>
              </div>
              <p className="text-xs text-[var(--text-muted)] leading-relaxed">
                Ranks non-linear dependency against preliminary pseudo-clusters to preserve high-signal drivers.
              </p>

              <div className="mt-3 pt-3 border-t border-[var(--border)] space-y-1">
                {features?.ds3_mutual_info_scores &&
                  Object.entries(features.ds3_mutual_info_scores)
                    .sort((a, b) => b[1] - a[1])
                    .slice(0, 4)
                    .map(([feat, score]) => (
                      <div key={feat} className="flex justify-between text-xs font-mono">
                        <span className="text-[var(--text)] truncate max-w-[130px]">{feat}</span>
                        <span className="text-accent-emerald">{score.toFixed(3)}</span>
                      </div>
                    ))}
              </div>
            </div>
          </div>

          {/* Card 3: PCA 95% Variance */}
          <div className="p-5 rounded-xl bg-[var(--card)] border border-[var(--border)] shadow-sm space-y-3 flex flex-col justify-between">
            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <span className="text-xs font-semibold text-[var(--text)]">DS4: PCA 95% Variance</span>
                <span className="text-[11px] px-2 py-0.5 rounded-full bg-accent-indigo/10 text-accent-indigo font-mono font-medium">
                  {features?.ds4_pca_n_components || 0} Orthogonal PCs
                </span>
              </div>
              <p className="text-xs text-[var(--text-muted)] leading-relaxed">
                Compresses dimensions into uncorrelated Principal Components retaining 95% of cumulative information.
              </p>

              {pcaChartData.length > 0 && (
                <div className="h-24 w-full pt-1">
                  <ResponsiveContainer width="100%" height="100%">
                    <AreaChart data={pcaChartData} margin={{ top: 5, right: 5, left: -25, bottom: 0 }}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#27272A" opacity={0.5} />
                      <XAxis dataKey="component" stroke="#71717A" tick={{ fontSize: 9 }} />
                      <YAxis stroke="#71717A" tick={{ fontSize: 9 }} domain={[0, 100]} />
                      <Tooltip
                        contentStyle={{ backgroundColor: '#121215', borderColor: '#27272A', fontSize: '11px' }}
                        formatter={(val: any) => [`${val}%`, 'Cumulative']}
                      />
                      <Area type="monotone" dataKey="cumulative" stroke="#6366F1" fill="#6366F1" fillOpacity={0.2} />
                    </AreaChart>
                  </ResponsiveContainer>
                </div>
              )}
            </div>
          </div>
        </div>
      </div>

      {/* Primary Action Button */}
      <div className="flex justify-end pt-2">
        <button
          type="button"
          onClick={onAdvanceToBenchmark}
          className="px-6 py-2.5 rounded-lg bg-accent-indigo hover:bg-accent-indigo/90 text-white font-medium text-xs shadow-sm transition-all flex items-center gap-2 cursor-pointer"
        >
          Run 16-experiment benchmark
          <ArrowRight className="w-4 h-4" />
        </button>
      </div>
    </div>
  )
}
