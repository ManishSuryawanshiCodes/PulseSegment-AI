'use client'

import React, { useState } from 'react'
import {
  ExperimentsMatrix,
  ExperimentResult,
  DatasetSummary,
  FeatureEngineering,
  GrowthPersonas,
} from '@/lib/api'
import { formatNumber, formatMs } from '@/lib/utils'
import {
  Sparkles,
  ArrowLeft,
  HelpCircle,
  Trophy,
  Layers,
  CheckCircle2,
  Sliders,
  TrendingUp,
  BarChart2,
  ScatterChart as ScatterIcon,
  Filter,
  Check,
  Info,
} from 'lucide-react'
import {
  ResponsiveContainer,
  ScatterChart,
  Scatter,
  XAxis,
  YAxis,
  ZAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  LineChart,
  Line,
  ReferenceLine,
  BarChart,
  Bar,
} from 'recharts'

interface MethodologyViewProps {
  matrix: ExperimentsMatrix | null
  summary: DatasetSummary | null
  features: FeatureEngineering | null
  personas: GrowthPersonas | null
  onBack: () => void
}

const PALETTE = ['#6366F1', '#10B981', '#F59E0B', '#F43F5E', '#8B5CF6', '#06B6D4', '#EC4899', '#14B8A6']

// Friendly data version names
function friendlyDataVersion(name: string): string {
  switch (name) {
    case 'DS1_Original':
      return 'All columns'
    case 'DS2_Correlation':
      return 'Without duplicates'
    case 'DS3_MutualInfo':
      return 'Most useful columns'
    case 'DS4_PCA95':
      return 'Simplified data'
    default:
      return name
  }
}

export function MethodologyView({
  matrix,
  summary,
  features,
  personas,
  onBack,
}: MethodologyViewProps) {
  const [tab, setTab] = useState<'simple' | 'advanced'>('simple')
  const [filterAlg, setFilterAlg] = useState<string>('all')

  const winningExp = matrix?.winning_experiment
  const experiments = matrix?.experiments || []

  // Dynamic "What am I looking at?" captions
  const winningModelName = winningExp
    ? `${winningExp.algorithm} (${friendlyDataVersion(winningExp.dataset_name)})`
    : 'Winning Model'

  const rawScatter = winningExp?.scatter_data || []
  const kSweepData = matrix?.k_sweep || []
  const suggestedK = winningExp?.cluster_count || matrix?.suggested_k || 4

  // Top features for spread chart
  const topFeatures = personas?.personas[0]
    ? Object.keys(personas.personas[0].centroid_profile).slice(0, 5)
    : []

  const spreadChartData = topFeatures.map((feat) => {
    const entry: Record<string, any> = { feature: feat.replace(/_/g, ' ') }
    personas?.personas.forEach((p) => {
      entry[p.name] = p.centroid_profile[feat] ?? 0
    })
    return entry
  })

  // Filtered experiments for table
  const filteredExperiments = experiments.filter((e) => {
    if (filterAlg !== 'all' && e.algorithm !== filterAlg) return false
    return true
  })

  return (
    <div className="space-y-6 max-w-[1100px] mx-auto animate-fadeIn pb-12">
      {/* Top Bar: Back Button, Title, and Tab Switcher */}
      <div className="bg-[var(--card)] border border-[var(--border)] rounded-2xl p-6 shadow-sm space-y-4">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
          <div className="flex items-center gap-3">
            <button
              type="button"
              onClick={onBack}
              className="p-2 rounded-xl bg-[var(--card-elevated)] border border-[var(--border)] hover:border-accent-indigo text-[var(--text)] transition-colors"
              title="Back to Customer Groups"
            >
              <ArrowLeft className="w-4 h-4" />
            </button>
            <div>
              <div className="flex items-center gap-2">
                <h2 className="text-xl sm:text-2xl font-bold tracking-tight text-[var(--text)]">
                  How we got these results
                </h2>
                <span className="px-2.5 py-0.5 rounded-full text-xs font-semibold bg-accent-indigo/10 text-accent-indigo border border-accent-indigo/25">
                  Winning Model: {winningModelName}
                </span>
              </div>
              <p className="text-xs text-[var(--text-muted)] mt-0.5">
                Every calculation and group boundary is evaluated dynamically from your active data.
              </p>
            </div>
          </div>

          {/* Simple vs Advanced Tabs */}
          <div className="flex items-center gap-1 p-1 bg-[var(--card-elevated)] border border-[var(--border)] rounded-xl text-xs self-start sm:self-auto">
            <button
              type="button"
              onClick={() => setTab('simple')}
              className={`px-4 py-1.5 rounded-lg font-semibold transition-all ${
                tab === 'simple'
                  ? 'bg-accent-indigo text-white shadow-sm'
                  : 'text-[var(--text-muted)] hover:text-[var(--text)]'
              }`}
            >
              Simple overview
            </button>
            <button
              type="button"
              onClick={() => setTab('advanced')}
              className={`px-4 py-1.5 rounded-lg font-semibold transition-all ${
                tab === 'advanced'
                  ? 'bg-accent-indigo text-white shadow-sm'
                  : 'text-[var(--text-muted)] hover:text-[var(--text)]'
              }`}
            >
              Advanced details
            </button>
          </div>
        </div>
      </div>

      {/* ================= SIMPLE TAB ================= */}
      {tab === 'simple' && (
        <div className="space-y-6 animate-fadeIn">
          {/* 3 Simple Stages */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div className="bg-[var(--card)] border border-[var(--border)] rounded-2xl p-5 space-y-2 shadow-sm">
              <div className="w-8 h-8 rounded-lg bg-accent-indigo/10 text-accent-indigo flex items-center justify-center font-bold text-xs">
                1
              </div>
              <h3 className="font-bold text-sm text-[var(--text)]">Cleaned your data</h3>
              <p className="text-xs text-[var(--text-muted)] leading-relaxed">
                Missing values were filled in based on similar customer patterns, unusual extreme values were capped, and columns were put on the same scale.
              </p>
            </div>

            <div className="bg-[var(--card)] border border-[var(--border)] rounded-2xl p-5 space-y-2 shadow-sm">
              <div className="w-8 h-8 rounded-lg bg-accent-indigo/10 text-accent-indigo flex items-center justify-center font-bold text-xs">
                2
              </div>
              <h3 className="font-bold text-sm text-[var(--text)]">Compared multiple methods</h3>
              <p className="text-xs text-[var(--text-muted)] leading-relaxed">
                We tested different clustering algorithms across 4 versions of your data (all columns, pruned duplicates, high-signal columns, and simplified data).
              </p>
            </div>

            <div className="bg-[var(--card)] border border-[var(--border)] rounded-2xl p-5 space-y-2 shadow-sm">
              <div className="w-8 h-8 rounded-lg bg-accent-emerald/10 text-accent-emerald flex items-center justify-center font-bold text-xs">
                3
              </div>
              <h3 className="font-bold text-sm text-[var(--text)]">Chose the clearest grouping</h3>
              <p className="text-xs text-[var(--text-muted)] leading-relaxed">
                The winning method scored highest for group clarity and boundary separation, ensuring your customer groups reflect genuine business behaviors.
              </p>
            </div>
          </div>

          {/* Winning Model Highlights */}
          {winningExp && (
            <div className="bg-[var(--card)] border border-[var(--border)] rounded-2xl p-6 shadow-sm space-y-4">
              <div className="flex items-center justify-between pb-3 border-b border-[var(--border)]">
                <div className="flex items-center gap-2">
                  <Trophy className="w-5 h-5 text-accent-amber" />
                  <div>
                    <h3 className="font-bold text-sm text-[var(--text)]">
                      Top Performer: {winningModelName}
                    </h3>
                    <span className="text-xs text-[var(--text-muted)]">
                      Discovered {winningExp.cluster_count} distinct customer groups.
                    </span>
                  </div>
                </div>
              </div>

              <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
                <div className="p-3.5 rounded-xl bg-[var(--card-elevated)] border border-[var(--border)]">
                  <div className="flex items-center justify-between">
                    <span className="text-[11px] font-medium text-[var(--text-muted)]">Group clarity</span>
                    <span className="text-[10px] text-accent-emerald font-medium">Higher is better</span>
                  </div>
                  <span className="text-xl font-bold metric-value text-[var(--text)] block mt-1">
                    {(winningExp.common_space_silhouette ?? winningExp.silhouette_score ?? 0).toFixed(3)}
                  </span>
                  <span className="text-[10px] text-[var(--text-muted)] mt-0.5 block">
                    Strong boundary separation
                  </span>
                </div>

                <div className="p-3.5 rounded-xl bg-[var(--card-elevated)] border border-[var(--border)]">
                  <div className="flex items-center justify-between">
                    <span className="text-[11px] font-medium text-[var(--text-muted)]">Group overlap</span>
                    <span className="text-[10px] text-accent-emerald font-medium">Lower is better</span>
                  </div>
                  <span className="text-xl font-bold metric-value text-[var(--text)] block mt-1">
                    {(winningExp.common_space_dbi ?? winningExp.davies_bouldin_index ?? 0).toFixed(2)}
                  </span>
                  <span className="text-[10px] text-[var(--text-muted)] mt-0.5 block">
                    Compact customer clusters
                  </span>
                </div>

                <div className="p-3.5 rounded-xl bg-[var(--card-elevated)] border border-[var(--border)]">
                  <div className="flex items-center justify-between">
                    <span className="text-[11px] font-medium text-[var(--text-muted)]">Separation strength</span>
                    <span className="text-[10px] text-accent-emerald font-medium">Higher is better</span>
                  </div>
                  <span className="text-xl font-bold metric-value text-[var(--text)] block mt-1">
                    {(winningExp.common_space_chi ?? winningExp.calinski_harabasz_score ?? 0).toFixed(0)}
                  </span>
                  <span className="text-[10px] text-[var(--text-muted)] mt-0.5 block">
                    Distinction vs noise
                  </span>
                </div>

                <div className="p-3.5 rounded-xl bg-[var(--card-elevated)] border border-[var(--border)]">
                  <span className="text-[11px] font-medium text-[var(--text-muted)] block">Execution time</span>
                  <span className="text-xl font-bold metric-value text-accent-indigo block mt-1">
                    {formatMs(winningExp.execution_time_ms)}
                  </span>
                  <span className="text-[10px] text-[var(--text-muted)] mt-0.5 block">
                    Wall-clock calculation
                  </span>
                </div>
              </div>
            </div>
          )}

          {/* 4 Data Versions Explained Simply */}
          <div className="bg-[var(--card)] border border-[var(--border)] rounded-2xl p-6 shadow-sm space-y-4">
            <h3 className="font-bold text-sm text-[var(--text)]">The 4 Data Versions We Tested</h3>
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 text-xs">
              <div className="p-3.5 rounded-xl bg-[var(--card-elevated)] border border-[var(--border)] space-y-1">
                <span className="font-semibold text-accent-indigo block">1. All columns</span>
                <p className="text-[var(--text-muted)] leading-relaxed">
                  Includes all numeric metrics from your file, scaled to the same range so larger dollar values do not drown out frequency.
                </p>
              </div>

              <div className="p-3.5 rounded-xl bg-[var(--card-elevated)] border border-[var(--border)] space-y-1">
                <span className="font-semibold text-accent-indigo block">2. Without duplicates</span>
                <p className="text-[var(--text-muted)] leading-relaxed">
                  Removes redundant metrics that track virtually the exact same trend as another column, avoiding double-counting.
                </p>
              </div>

              <div className="p-3.5 rounded-xl bg-[var(--card-elevated)] border border-[var(--border)] space-y-1">
                <span className="font-semibold text-accent-indigo block">3. Most useful columns</span>
                <p className="text-[var(--text-muted)] leading-relaxed">
                  Selects the top independent metrics that carry the strongest signal for separating customer habits.
                </p>
              </div>

              <div className="p-3.5 rounded-xl bg-[var(--card-elevated)] border border-[var(--border)] space-y-1">
                <span className="font-semibold text-accent-indigo block">4. Simplified data</span>
                <p className="text-[var(--text-muted)] leading-relaxed">
                  Condenses dozens of metrics into the few strongest independent signals that still explain 95% of customer variance.
                </p>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* ================= ADVANCED TAB ================= */}
      {tab === 'advanced' && (
        <div className="space-y-6 animate-fadeIn">
          {/* 1. Comparison Matrix Table */}
          <div className="bg-[var(--card)] border border-[var(--border)] rounded-2xl p-6 shadow-sm space-y-4">
            <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
              <div>
                <h3 className="text-sm font-bold text-[var(--text)]">
                  Method Comparison Matrix ({experiments.length} Runs)
                </h3>
                <p className="text-xs text-[var(--text-muted)] mt-0.5">
                  Compares 4 clustering algorithms across 4 data versions. The highlighted row is the winning configuration.
                </p>
              </div>

              {/* Filter */}
              <div className="flex items-center gap-2 text-xs">
                <Filter className="w-3.5 h-3.5 text-[var(--text-muted)]" />
                <select
                  value={filterAlg}
                  onChange={(e) => setFilterAlg(e.target.value)}
                  className="bg-[var(--card-elevated)] border border-[var(--border)] rounded-lg px-2.5 py-1 text-xs text-[var(--text)] focus:outline-none focus:border-accent-indigo"
                >
                  <option value="all">All Algorithms</option>
                  <option value="PulseCluster">PulseCluster</option>
                  <option value="KMeans">KMeans</option>
                  <option value="Agglomerative">Agglomerative</option>
                  <option value="DBSCAN">DBSCAN</option>
                  <option value="GaussianMixture">GaussianMixture</option>
                </select>
              </div>
            </div>

            <div className="overflow-x-auto border border-[var(--border)] rounded-xl max-h-72">
              <table className="w-full text-xs text-left">
                <thead className="bg-[var(--card-elevated)] text-[var(--text-muted)] border-b border-[var(--border)] font-mono text-[11px]">
                  <tr>
                    <th className="px-3.5 py-2 font-medium">Algorithm</th>
                    <th className="px-3.5 py-2 font-medium">Data Version</th>
                    <th className="px-3.5 py-2 font-medium">
                      <span className="flex items-center gap-1" title="Measures how clearly separated customer groups are (higher is better, max 1.0).">
                        Group clarity (?)
                      </span>
                    </th>
                    <th className="px-3.5 py-2 font-medium">
                      <span className="flex items-center gap-1" title="Measures how much customer groups overlap with each other (lower is better).">
                        Group overlap (?)
                      </span>
                    </th>
                    <th className="px-3.5 py-2 font-medium">
                      <span className="flex items-center gap-1" title="Measures overall group distinction compared to random noise (higher is better).">
                        Separation strength (?)
                      </span>
                    </th>
                    <th className="px-3.5 py-2 font-medium">Speed</th>
                    <th className="px-3.5 py-2 font-medium">Groups</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-[var(--border)]">
                  {filteredExperiments.map((exp, idx) => {
                    const isWinner =
                      exp.algorithm === winningExp?.algorithm &&
                      exp.dataset_name === winningExp?.dataset_name
                    const clarity = exp.common_space_silhouette ?? exp.silhouette_score ?? 0
                    const overlap = exp.common_space_dbi ?? exp.davies_bouldin_index ?? 0
                    const separation = exp.common_space_chi ?? exp.calinski_harabasz_score ?? 0

                    return (
                      <tr
                        key={idx}
                        className={`hover:bg-[var(--card-elevated)]/40 transition-colors ${
                          isWinner ? 'bg-accent-indigo/10 font-semibold' : ''
                        }`}
                      >
                        <td className="px-3.5 py-2 whitespace-nowrap text-[var(--text)] flex items-center gap-1.5">
                          {isWinner && <Trophy className="w-3.5 h-3.5 text-accent-amber shrink-0" />}
                          <span>{exp.algorithm}</span>
                        </td>
                        <td className="px-3.5 py-2 whitespace-nowrap text-[var(--text-muted)]">
                          {friendlyDataVersion(exp.dataset_name)}
                        </td>
                        <td className="px-3.5 py-2 whitespace-nowrap font-mono metric-value text-accent-emerald">
                          {clarity ? clarity.toFixed(3) : '-'}
                        </td>
                        <td className="px-3.5 py-2 whitespace-nowrap font-mono metric-value text-accent-amber">
                          {overlap ? overlap.toFixed(2) : '-'}
                        </td>
                        <td className="px-3.5 py-2 whitespace-nowrap font-mono metric-value text-[var(--text)]">
                          {separation ? separation.toFixed(0) : '-'}
                        </td>
                        <td className="px-3.5 py-2 whitespace-nowrap font-mono metric-value text-[var(--text-muted)]">
                          {formatMs(exp.execution_time_ms)}
                        </td>
                        <td className="px-3.5 py-2 whitespace-nowrap font-mono metric-value text-[var(--text)]">
                          {exp.cluster_count}
                        </td>
                      </tr>
                    )
                  })}
                </tbody>
              </table>
            </div>
          </div>

          {/* 2. Charts Grid: 2D Projection Scatter & Optimal Groups Sweep */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* 2D Projection Scatter */}
            <div className="bg-[var(--card)] border border-[var(--border)] rounded-2xl p-6 shadow-sm space-y-3">
              <div>
                <h3 className="text-sm font-bold text-[var(--text)]">2D Customer Projection</h3>
                <p className="text-xs text-[var(--text-muted)] mt-0.5">
                  Showing {rawScatter.length} sampled customers along top independent signals, colored by customer group.
                </p>
              </div>

              <div className="h-64 w-full">
                {rawScatter.length > 0 ? (
                  <ResponsiveContainer width="100%" height="100%">
                    <ScatterChart margin={{ top: 10, right: 10, bottom: 20, left: 0 }}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#27272A" opacity={0.5} />
                      <XAxis type="number" dataKey="pc1" name="PC1" stroke="#71717A" fontSize={10} tickLine={false} />
                      <YAxis type="number" dataKey="pc2" name="PC2" stroke="#71717A" fontSize={10} tickLine={false} />
                      <Tooltip
                        contentStyle={{
                          backgroundColor: '#18181B',
                          border: '1px solid #27272A',
                          borderRadius: '8px',
                          fontSize: '11px',
                        }}
                      />
                      <Scatter data={rawScatter} fill="#6366F1" />
                    </ScatterChart>
                  </ResponsiveContainer>
                ) : (
                  <div className="h-full flex items-center justify-center text-xs text-[var(--text-muted)]">
                    No scatter projection available
                  </div>
                )}
              </div>
            </div>

            {/* Optimal Groups Sweep (Elbow Chart) */}
            <div className="bg-[var(--card)] border border-[var(--border)] rounded-2xl p-6 shadow-sm space-y-3">
              <div>
                <h3 className="text-sm font-bold text-[var(--text)]">Optimal Groups Sweep (Elbow Curve)</h3>
                <p className="text-xs text-[var(--text-muted)] mt-0.5">
                  Measures how group clarity changes across candidate group sizes (3 to 8). Suggested: {suggestedK} groups.
                </p>
              </div>

              <div className="h-64 w-full">
                {kSweepData.length > 0 ? (
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={kSweepData} margin={{ top: 10, right: 10, bottom: 20, left: 10 }}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#27272A" opacity={0.5} />
                      <XAxis dataKey="k" stroke="#71717A" fontSize={10} tickLine={false} />
                      <YAxis stroke="#71717A" fontSize={10} tickLine={false} />
                      <Tooltip
                        contentStyle={{
                          backgroundColor: '#18181B',
                          border: '1px solid #27272A',
                          borderRadius: '8px',
                          fontSize: '11px',
                        }}
                      />
                      <ReferenceLine x={suggestedK} stroke="#10B981" strokeDasharray="3 3" label={{ value: `Optimal k=${suggestedK}`, fill: '#10B981', fontSize: 10 }} />
                      <Line type="monotone" dataKey="inertia" stroke="#6366F1" strokeWidth={2} dot={{ r: 3 }} />
                    </LineChart>
                  </ResponsiveContainer>
                ) : (
                  <div className="h-full flex items-center justify-center text-xs text-[var(--text-muted)]">
                    Sweep data available after benchmark execution
                  </div>
                )}
              </div>
            </div>
          </div>

          {/* 3. Feature Spread Across Groups */}
          {spreadChartData.length > 0 && (
            <div className="bg-[var(--card)] border border-[var(--border)] rounded-2xl p-6 shadow-sm space-y-3">
              <div>
                <h3 className="text-sm font-bold text-[var(--text)]">Key Column Spreads Across Groups</h3>
                <p className="text-xs text-[var(--text-muted)] mt-0.5">
                  Shows how key numerical columns differ between customer groups compared with the overall average.
                </p>
              </div>

              <div className="h-64 w-full">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={spreadChartData} margin={{ top: 10, right: 10, bottom: 20, left: 0 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#27272A" opacity={0.5} />
                    <XAxis dataKey="feature" stroke="#71717A" fontSize={10} tickLine={false} />
                    <YAxis stroke="#71717A" fontSize={10} tickLine={false} />
                    <Tooltip
                      contentStyle={{
                        backgroundColor: '#18181B',
                        border: '1px solid #27272A',
                        borderRadius: '8px',
                        fontSize: '11px',
                      }}
                    />
                    <Legend wrapperStyle={{ fontSize: '11px' }} />
                    {personas?.personas.map((p, pIdx) => (
                      <Bar key={p.id ?? pIdx} dataKey={p.name} fill={PALETTE[pIdx % PALETTE.length]} radius={[4, 4, 0, 0]} />
                    ))}
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  )
}
