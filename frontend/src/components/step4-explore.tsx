'use client'

import React, { useState } from 'react'
import {
  ExperimentsMatrix,
  GrowthPersonas,
  DatasetConfig,
  api,
} from '@/lib/api'
import { formatNumber } from '@/lib/utils'
import {
  ScatterChart,
  HelpCircle,
  TrendingUp,
  Layers,
  ArrowRight,
  Info,
  Sparkles,
  BarChart2,
  RefreshCw,
  Sliders,
} from 'lucide-react'
import {
  ResponsiveContainer,
  ScatterChart as RechartsScatter,
  Scatter,
  XAxis,
  YAxis,
  ZAxis,
  CartesianGrid,
  Tooltip as RechartsTooltip,
  Legend,
  LineChart,
  Line,
  ReferenceLine,
  BarChart,
  Bar,
} from 'recharts'

interface Step4ExploreProps {
  matrix: ExperimentsMatrix | null
  personas: GrowthPersonas | null
  config: DatasetConfig | null
  datasetId: string
  onRerunWithK: (k: number) => void
  onAdvanceToAct: () => void
}

const CLUSTER_PALETTE = ['#6366F1', '#10B981', '#F59E0B', '#F43F5E', '#8B5CF6', '#06B6D4', '#EC4899', '#14B8A6']

export function Step4Explore({
  matrix,
  personas,
  config,
  datasetId,
  onRerunWithK,
  onAdvanceToAct,
}: Step4ExploreProps) {
  const [selectedDataset, setSelectedDataset] = useState<string>('DS4_PCA95')
  const [selectedAlgorithm, setSelectedAlgorithm] = useState<string>('KMeans')
  const [kChoice, setKChoice] = useState<number>(config?.n_clusters || 4)
  const [isRerunning, setIsRerunning] = useState<boolean>(false)

  const experiments = matrix?.experiments || []

  // Find matching experiment for scatter projection
  const currentExp =
    experiments.find(
      (e) => e.dataset_name === selectedDataset && e.algorithm === selectedAlgorithm
    ) || matrix?.winning_experiment || experiments[0]

  // Scatter points (500 sampled points from backend)
  const rawScatter = currentExp?.scatter_data || []
  const clusterSizes = currentExp?.cluster_sizes || {}
  const totalRecords = Object.values(clusterSizes).reduce((a, b) => a + b, 0) || 500

  // Group scatter points by cluster label for distinct colors
  const clustersInScatter = Array.from(new Set(rawScatter.map((p) => p.cluster_label))).sort(
    (a, b) => a - b
  )

  // Elbow K-sweep data
  const kSweepData = matrix?.k_sweep || []
  const suggestedK = matrix?.suggested_k || 4

  // Persona profiles for mapping cluster labels to persona names
  const personaMap: Record<number, string> = {}
  personas?.personas.forEach((p, idx) => {
    personaMap[p.id ?? idx] = p.name
  })

  // Feature Spread Data per cluster for the Box Plot representation
  const topFeatures = personas?.personas[0]
    ? Object.keys(personas.personas[0].centroid_profile).slice(0, 5)
    : []

  const featureSpreadData = topFeatures.map((feat) => {
    const entry: Record<string, any> = { feature: feat }
    personas?.personas.forEach((p, idx) => {
      entry[p.name] = p.centroid_profile[feat] ?? 0
    })
    return entry
  })

  const handleKChange = async () => {
    setIsRerunning(true)
    try {
      await onRerunWithK(kChoice)
    } finally {
      setIsRerunning(false)
    }
  }

  // Box Plot Feature Selection
  const boxPlotList = personas?.box_plots || []
  const availableBoxFeatures = boxPlotList.length > 0
    ? boxPlotList.map((b) => b.feature)
    : topFeatures
  const [selectedBoxFeature, setSelectedBoxFeature] = useState<string>(availableBoxFeatures[0] || '')
  const [exploreViewMode, setExploreViewMode] = useState<'box' | 'bar'>('box')

  // Keep selectedBoxFeature in sync
  React.useEffect(() => {
    if (availableBoxFeatures.length > 0 && (!selectedBoxFeature || !availableBoxFeatures.includes(selectedBoxFeature))) {
      setSelectedBoxFeature(availableBoxFeatures[0])
    }
  }, [availableBoxFeatures, selectedBoxFeature])

  const activeBoxStat = boxPlotList.find((b) => b.feature === selectedBoxFeature)

  return (
    <div className="space-y-8 animate-fadeIn">
      {/* 1. 2D PCA Scatter Projection */}
      <div className="bg-[var(--card)] border border-[var(--border)] rounded-xl p-6 shadow-sm space-y-4">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 pb-4 border-b border-[var(--border)]">
          <div>
            <h3 className="text-sm font-semibold text-[var(--text)] flex items-center gap-2">
              <ScatterChart className="w-4 h-4 text-accent-indigo" />
              2D Principal Component Projection (PC1 &times; PC2)
            </h3>
            <p className="text-xs text-[var(--text-muted)] mt-0.5">
              Visualizes non-linear cluster separation in 2D space. Hover over points to inspect segment assignment.
            </p>
          </div>

          {/* Model Pickers for Scatter Plot */}
          <div className="flex flex-wrap items-center gap-2 text-xs">
            <select
              value={selectedDataset}
              onChange={(e) => setSelectedDataset(e.target.value)}
              className="bg-[var(--card-elevated)] border border-[var(--border)] rounded px-2.5 py-1 text-[var(--text)] focus:outline-none"
            >
              <option value="DS4_PCA95">DS4: PCA 95%</option>
              <option value="DS1_Original">DS1: Original</option>
              <option value="DS2_Correlation">DS2: Correlation</option>
              <option value="DS3_MutualInfo">DS3: Mutual Info</option>
            </select>

            <select
              value={selectedAlgorithm}
              onChange={(e) => setSelectedAlgorithm(e.target.value)}
              className="bg-[var(--card-elevated)] border border-[var(--border)] rounded px-2.5 py-1 text-[var(--text)] focus:outline-none"
            >
              <option value="KMeans">K-Means</option>
              <option value="Agglomerative">Agglomerative</option>
              <option value="DBSCAN">DBSCAN</option>
              <option value="GaussianMixture">Gaussian Mixture</option>
            </select>
          </div>
        </div>

        {/* Scatter Chart Area */}
        <div className="h-80 w-full pt-1">
          <ResponsiveContainer width="100%" height="100%">
            <RechartsScatter margin={{ top: 10, right: 20, left: 0, bottom: 10 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#27272A" opacity={0.5} />
              <XAxis
                type="number"
                dataKey="pc1"
                name="PC1"
                stroke="#71717A"
                tick={{ fontSize: 11 }}
                label={{ value: 'Principal Component 1', position: 'bottom', offset: -5, fontSize: 11, fill: '#71717A' }}
              />
              <YAxis
                type="number"
                dataKey="pc2"
                name="PC2"
                stroke="#71717A"
                tick={{ fontSize: 11 }}
                label={{ value: 'Principal Component 2', angle: -90, position: 'left', offset: 15, fontSize: 11, fill: '#71717A' }}
              />
              <ZAxis range={[25, 25]} />
              <RechartsTooltip
                cursor={{ strokeDasharray: '3 3' }}
                content={({ payload }) => {
                  if (!payload || !payload[0]) return null
                  const data = payload[0].payload
                  const pName = personaMap[data.cluster_label] || `Cluster ${data.cluster_label}`
                  return (
                    <div className="p-2.5 rounded-lg bg-zinc-950 text-zinc-100 text-xs border border-zinc-800 shadow-xl space-y-1 font-mono">
                      <div className="font-sans font-bold text-accent-indigo">{pName}</div>
                      <div>PC1: {data.pc1.toFixed(2)}</div>
                      <div>PC2: {data.pc2.toFixed(2)}</div>
                      <div className="text-[10px] text-zinc-400 font-sans">
                        Cluster #{data.cluster_label}
                      </div>
                    </div>
                  )
                }}
              />

              {clustersInScatter.map((cLabel, idx) => {
                const pts = rawScatter.filter((p) => p.cluster_label === cLabel)
                const pColor = CLUSTER_PALETTE[idx % CLUSTER_PALETTE.length]
                const pName = personaMap[cLabel] || `Cluster ${cLabel}`
                const count = clusterSizes[cLabel] || pts.length
                const pct = totalRecords > 0 ? ((count / totalRecords) * 100).toFixed(1) : '0'

                return (
                  <Scatter
                    key={cLabel}
                    name={`${pName} (N=${formatNumber(count)} • ${pct}%)`}
                    data={pts}
                    fill={pColor}
                  />
                )
              })}
              <Legend
                wrapperStyle={{ fontSize: '11px', paddingTop: '15px' }}
                iconType="circle"
              />
            </RechartsScatter>
          </ResponsiveContainer>
        </div>

        {/* Plain Language Caption & Subtitle Note */}
        <div className="p-3 rounded-lg bg-[var(--card-elevated)] border border-[var(--border)] text-xs text-[var(--text-muted)] flex items-start gap-2">
          <Info className="w-4 h-4 text-accent-indigo shrink-0 mt-0.5" />
          <div>
            <span className="font-semibold text-[var(--text)] block">What am I looking at?</span>
            <span>
              Each dot represents an individual customer mapped along their two strongest behavioral dimensions (PC1 and PC2).
              Dots with the same color belong to the same cluster. Notice how the groups form distinct islands without overlapping.
              <span className="text-[11px] block mt-0.5 font-mono text-[var(--text-muted)]">
                * Displaying 500 stratified sampled points for rendering responsiveness; true portfolio sizes shown in legend.
              </span>
            </span>
          </div>
        </div>
      </div>

      {/* 2. Real Elbow & Silhouette K-Sweep (k=2..10) with Re-run Control */}
      <div className="bg-[var(--card)] border border-[var(--border)] rounded-xl p-6 shadow-sm space-y-5">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 pb-4 border-b border-[var(--border)]">
          <div>
            <h3 className="text-sm font-semibold text-[var(--text)] flex items-center gap-2">
              <TrendingUp className="w-4 h-4 text-accent-emerald" />
              Optimal Cluster Search (Elbow & Silhouette Sweep, k=2..10)
            </h3>
            <p className="text-xs text-[var(--text-muted)] mt-0.5">
              Mathematically identifies the optimal segment count where inertia loss levels off and silhouette separation peaks.
            </p>
          </div>

          {/* Re-run with k= control */}
          <div className="flex items-center gap-2 bg-[var(--card-elevated)] p-1.5 rounded-lg border border-[var(--border)] text-xs">
            <span className="text-[var(--text-muted)] whitespace-nowrap">Clusters (k):</span>
            <input
              type="number"
              min={2}
              max={8}
              value={kChoice}
              onChange={(e) => setKChoice(parseInt(e.target.value) || 4)}
              className="w-14 bg-[var(--card)] border border-[var(--border)] rounded px-2 py-1 font-mono font-bold text-center text-[var(--text)]"
            />
            <button
              type="button"
              onClick={handleKChange}
              disabled={isRerunning}
              className="px-3 py-1 rounded bg-accent-indigo hover:bg-accent-indigo/90 text-white font-medium flex items-center gap-1 transition-colors disabled:opacity-50"
            >
              {isRerunning ? (
                <RefreshCw className="w-3.5 h-3.5 animate-spin" />
              ) : (
                'Re-run'
              )}
            </button>
          </div>
        </div>

        {/* Dual-Axis Line Chart: Inertia + Silhouette */}
        <div className="h-64 w-full pt-1">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={kSweepData} margin={{ top: 10, right: 30, left: 10, bottom: 5 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#27272A" opacity={0.5} />
              <XAxis dataKey="k" stroke="#71717A" tick={{ fontSize: 11 }} label={{ value: 'Number of Clusters (k)', position: 'bottom', offset: -5, fontSize: 11 }} />
              <YAxis yAxisId="left" stroke="#71717A" tick={{ fontSize: 11 }} label={{ value: 'Inertia (Loss)', angle: -90, position: 'left', offset: 0, fontSize: 11 }} />
              <YAxis yAxisId="right" orientation="right" stroke="#71717A" tick={{ fontSize: 11 }} label={{ value: 'Silhouette Score', angle: 90, position: 'right', offset: 10, fontSize: 11 }} />
              <RechartsTooltip
                contentStyle={{ backgroundColor: '#121215', borderColor: '#27272A', borderRadius: '8px', fontSize: '12px', color: '#FAFAFA' }}
              />
              <Legend wrapperStyle={{ fontSize: '11px', paddingTop: '10px' }} />
              <ReferenceLine yAxisId="right" x={suggestedK} stroke="#10B981" strokeDasharray="3 3" label={{ value: `Suggested k=${suggestedK}`, position: 'top', fill: '#10B981', fontSize: 11 }} />
              <Line yAxisId="left" type="monotone" dataKey="inertia" name="Inertia (Elbow)" stroke="#6366F1" strokeWidth={2} dot={{ r: 4 }} activeDot={{ r: 6 }} />
              <Line yAxisId="right" type="monotone" dataKey="silhouette" name="Silhouette Score" stroke="#10B981" strokeWidth={2} dot={{ r: 4 }} activeDot={{ r: 6 }} />
            </LineChart>
          </ResponsiveContainer>
        </div>

        <div className="p-3 rounded-lg bg-[var(--card-elevated)] border border-[var(--border)] text-xs text-[var(--text-muted)] flex items-start gap-2">
          <Sparkles className="w-4 h-4 text-accent-emerald shrink-0 mt-0.5" />
          <div>
            <span className="font-semibold text-[var(--text)] block">Optimal Choice Recommendation</span>
            <span>
              The algorithm identified <strong className="text-accent-emerald font-mono">k={suggestedK}</strong> as the mathematically optimal cluster configuration, maximizing silhouette separation while preserving distinct actionable customer cohorts.
            </span>
          </div>
        </div>
      </div>

      {/* 3. Per-Cluster Feature Box Plots & Spread */}
      {availableBoxFeatures.length > 0 && (
        <div className="bg-[var(--card)] border border-[var(--border)] rounded-xl p-6 shadow-sm space-y-5">
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 pb-4 border-b border-[var(--border)]">
            <div>
              <h3 className="text-sm font-semibold text-[var(--text)] flex items-center gap-2">
                <BarChart2 className="w-4 h-4 text-accent-indigo" />
                Per-Cluster Feature Box Plots & Distributions
              </h3>
              <p className="text-xs text-[var(--text-muted)] mt-0.5">
                Inspect quartile spreads (Q1, median, Q3) and min/max whiskers across each customer segment in original feature units.
              </p>
            </div>

            {/* View Mode Toggle: Box Plot vs Bar Chart */}
            <div className="flex items-center gap-1 bg-[var(--card-elevated)] p-1 rounded-lg border border-[var(--border)] text-xs font-medium self-start sm:self-auto">
              <button
                type="button"
                onClick={() => setExploreViewMode('box')}
                className={`px-3 py-1 rounded transition-colors ${
                  exploreViewMode === 'box'
                    ? 'bg-accent-indigo text-white font-semibold'
                    : 'text-[var(--text-muted)] hover:text-[var(--text)]'
                }`}
              >
                Box Plots
              </button>
              <button
                type="button"
                onClick={() => setExploreViewMode('bar')}
                className={`px-3 py-1 rounded transition-colors ${
                  exploreViewMode === 'bar'
                    ? 'bg-accent-indigo text-white font-semibold'
                    : 'text-[var(--text-muted)] hover:text-[var(--text)]'
                }`}
              >
                Averages Bar Chart
              </button>
            </div>
          </div>

          {/* Feature Selector Pills */}
          <div className="flex items-center gap-2 overflow-x-auto pb-1 text-xs">
            <span className="text-[var(--text-muted)] text-[11px] font-medium shrink-0">Select Feature:</span>
            {availableBoxFeatures.map((feat) => (
              <button
                key={feat}
                type="button"
                onClick={() => setSelectedBoxFeature(feat)}
                className={`px-2.5 py-1 rounded-md text-xs font-mono transition-colors shrink-0 ${
                  selectedBoxFeature === feat
                    ? 'bg-accent-indigo text-white font-bold'
                    : 'bg-[var(--card-elevated)] border border-[var(--border)] text-[var(--text-muted)] hover:text-[var(--text)]'
                }`}
              >
                {feat}
              </button>
            ))}
          </div>

          {exploreViewMode === 'box' ? (
            /* Box Plot SVG Visualization */
            <div className="space-y-4">
              <div className="bg-[var(--card-elevated)] border border-[var(--border)] rounded-xl p-4 overflow-x-auto">
                {activeBoxStat ? (
                  (() => {
                    const clusterEntries = Object.entries(activeBoxStat.clusters)
                    const allMins = clusterEntries.map(([_, s]) => s.min)
                    const allMaxs = clusterEntries.map(([_, s]) => s.max)
                    const minAll = Math.min(...allMins)
                    const maxAll = Math.max(...allMaxs)
                    const span = maxAll - minAll || 1
                    const svgWidth = Math.max(600, clusterEntries.length * 140)
                    const svgHeight = 220
                    const padTop = 20
                    const padBottom = 30
                    const padLeft = 60
                    const plotH = svgHeight - padTop - padBottom
                    const slotW = (svgWidth - padLeft) / Math.max(1, clusterEntries.length)

                    const getY = (val: number) => {
                      const ratio = (val - minAll) / span
                      return padTop + plotH - ratio * plotH
                    }

                    return (
                      <div className="min-w-[600px]">
                        <svg
                          width="100%"
                          height={svgHeight}
                          viewBox={`0 0 ${svgWidth} ${svgHeight}`}
                          className="overflow-visible font-mono text-[11px]"
                        >
                          {/* Y-Axis Horizontal Gridlines */}
                          {[0, 0.25, 0.5, 0.75, 1.0].map((t) => {
                            const val = minAll + t * span
                            const y = getY(val)
                            return (
                              <g key={t}>
                                <line
                                  x1={padLeft - 6}
                                  y1={y}
                                  x2={svgWidth}
                                  y2={y}
                                  stroke="#27272A"
                                  strokeDasharray="3 3"
                                  strokeOpacity={0.6}
                                />
                                <text
                                  x={padLeft - 10}
                                  y={y + 3}
                                  textAnchor="end"
                                  fill="#71717A"
                                  fontSize="10"
                                >
                                  {formatNumber(Math.round(val))}
                                </text>
                              </g>
                            )
                          })}

                          {/* Render Box & Whiskers per Cluster */}
                          {clusterEntries.map(([cName, s], idx) => {
                            const cx = padLeft + (idx + 0.5) * slotW
                            const col = CLUSTER_PALETTE[idx % CLUSTER_PALETTE.length]
                            const yMin = getY(s.min)
                            const yQ1 = getY(s.q1)
                            const yMed = getY(s.median)
                            const yQ3 = getY(s.q3)
                            const yMax = getY(s.max)
                            const boxW = Math.min(54, slotW * 0.5)

                            return (
                              <g key={cName}>
                                {/* Whisker vertical line */}
                                <line
                                  x1={cx}
                                  y1={yMin}
                                  x2={cx}
                                  y2={yMax}
                                  stroke={col}
                                  strokeWidth="1.5"
                                  strokeDasharray="2 2"
                                />

                                {/* Upper whisker cap */}
                                <line
                                  x1={cx - boxW * 0.35}
                                  y1={yMax}
                                  x2={cx + boxW * 0.35}
                                  y2={yMax}
                                  stroke={col}
                                  strokeWidth="2"
                                />

                                {/* Lower whisker cap */}
                                <line
                                  x1={cx - boxW * 0.35}
                                  y1={yMin}
                                  x2={cx + boxW * 0.35}
                                  y2={yMin}
                                  stroke={col}
                                  strokeWidth="2"
                                />

                                {/* IQR Box (Q1 to Q3) */}
                                <rect
                                  x={cx - boxW / 2}
                                  y={Math.min(yQ1, yQ3)}
                                  width={boxW}
                                  height={Math.max(3, Math.abs(yQ1 - yQ3))}
                                  fill={col}
                                  fillOpacity="0.25"
                                  stroke={col}
                                  strokeWidth="1.5"
                                  rx="3"
                                />

                                {/* Median horizontal line */}
                                <line
                                  x1={cx - boxW / 2}
                                  y1={yMed}
                                  x2={cx + boxW / 2}
                                  y2={yMed}
                                  stroke="#FFFFFF"
                                  strokeWidth="2.5"
                                />

                                {/* X-Axis Cluster Label */}
                                <text
                                  x={cx}
                                  y={svgHeight - 10}
                                  textAnchor="middle"
                                  fill="#A1A1AA"
                                  fontSize="10"
                                  fontWeight="600"
                                >
                                  {cName.length > 18 ? `${cName.slice(0, 16)}...` : cName}
                                </text>
                              </g>
                            )
                          })}
                        </svg>

                        {/* Statistical Summary Pills per Cluster */}
                        <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-4 gap-2.5 pt-4 border-t border-[var(--border)]">
                          {clusterEntries.map(([cName, s], idx) => {
                            const col = CLUSTER_PALETTE[idx % CLUSTER_PALETTE.length]
                            return (
                              <div
                                key={cName}
                                className="p-2.5 rounded-lg bg-[var(--card)] border border-[var(--border)] space-y-1 text-xs font-mono"
                              >
                                <div className="flex items-center gap-1.5 font-sans font-bold text-[var(--text)] text-[11px] truncate">
                                  <span
                                    className="w-2 h-2 rounded-full shrink-0"
                                    style={{ backgroundColor: col }}
                                  />
                                  <span className="truncate">{cName}</span>
                                </div>
                                <div className="grid grid-cols-2 gap-x-2 gap-y-0.5 text-[10px] text-[var(--text-muted)] pt-1">
                                  <div>Min: <strong className="text-[var(--text)]">{formatNumber(s.min)}</strong></div>
                                  <div>Max: <strong className="text-[var(--text)]">{formatNumber(s.max)}</strong></div>
                                  <div>Q1 (25%): <strong className="text-[var(--text)]">{formatNumber(s.q1)}</strong></div>
                                  <div>Q3 (75%): <strong className="text-[var(--text)]">{formatNumber(s.q3)}</strong></div>
                                  <div className="col-span-2 pt-0.5 text-accent-indigo font-bold">
                                    Median: {formatNumber(s.median)}
                                  </div>
                                </div>
                              </div>
                            )
                          })}
                        </div>
                      </div>
                    )
                  })()
                ) : (
                  <div className="p-4 text-center text-xs text-[var(--text-muted)]">
                    No box plot distribution data available for {selectedBoxFeature}.
                  </div>
                )}
              </div>
            </div>
          ) : (
            /* Grouped Bar Chart View */
            <div className="h-64 w-full pt-1">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={featureSpreadData} margin={{ top: 10, right: 30, left: 10, bottom: 5 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#27272A" opacity={0.5} />
                  <XAxis dataKey="feature" stroke="#71717A" tick={{ fontSize: 11 }} />
                  <YAxis stroke="#71717A" tick={{ fontSize: 11 }} />
                  <RechartsTooltip contentStyle={{ backgroundColor: '#121215', borderColor: '#27272A', fontSize: '12px' }} />
                  <Legend wrapperStyle={{ fontSize: '11px', paddingTop: '10px' }} />
                  {personas?.personas.map((p, idx) => (
                    <Bar
                      key={p.name}
                      dataKey={p.name}
                      fill={CLUSTER_PALETTE[idx % CLUSTER_PALETTE.length]}
                      radius={[4, 4, 0, 0]}
                    />
                  ))}
                </BarChart>
              </ResponsiveContainer>
            </div>
          )}

          <div className="p-3 rounded-lg bg-[var(--card-elevated)] border border-[var(--border)] text-xs text-[var(--text-muted)] flex items-start gap-2">
            <Info className="w-4 h-4 text-accent-indigo shrink-0 mt-0.5" />
            <div>
              <span className="font-semibold text-[var(--text)] block">What am I looking at?</span>
              <span>
                The box plot displays the middle 50% interquartile range (IQR) from Q1 to Q3 for{' '}
                <strong className="text-[var(--text)] font-mono">{selectedBoxFeature}</strong>. The solid white bar inside each box is the median customer value, while the dotted whiskers extend to the minimum and maximum boundaries for each customer persona.
              </span>
            </div>
          </div>
        </div>
      )}

      {/* Primary Action Button */}
      <div className="flex justify-end pt-2">
        <button
          type="button"
          onClick={onAdvanceToAct}
          className="px-6 py-2.5 rounded-lg bg-accent-indigo hover:bg-accent-indigo/90 text-white font-medium text-xs shadow-sm transition-all flex items-center gap-2 cursor-pointer"
        >
          Plan campaigns & actions
          <ArrowRight className="w-4 h-4" />
        </button>
      </div>
    </div>
  )
}
