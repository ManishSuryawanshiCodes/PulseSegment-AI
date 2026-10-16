'use client'

import React from 'react'
import { X, HelpCircle, BookOpen, Layers, CheckCircle2, FileText, Database, TrendingUp, Sparkles } from 'lucide-react'

interface HelpDrawerProps {
  open: boolean
  onClose: () => void
}

export function HelpDrawer({ open, onClose }: HelpDrawerProps) {
  if (!open) return null

  return (
    <div className="fixed inset-0 z-50 flex justify-end">
      {/* Backdrop */}
      <div
        className="fixed inset-0 bg-black/60 backdrop-blur-sm transition-opacity"
        onClick={onClose}
      />

      {/* Drawer Panel */}
      <div className="relative w-full max-w-md bg-[var(--card)] border-l border-[var(--border)] h-full overflow-y-auto p-6 shadow-2xl flex flex-col justify-between z-10 text-[var(--text)]">
        <div className="space-y-6">
          {/* Header */}
          <div className="flex items-center justify-between pb-4 border-b border-[var(--border)]">
            <div className="flex items-center gap-2">
              <div className="w-8 h-8 rounded-lg bg-accent-indigo/10 border border-accent-indigo/30 flex items-center justify-center text-accent-indigo">
                <HelpCircle className="w-4 h-4" />
              </div>
              <div>
                <h3 className="font-semibold text-sm">PulseSegment Guide</h3>
                <p className="text-xs text-[var(--text-muted)]">Everything you need to know in plain English</p>
              </div>
            </div>
            <button
              onClick={onClose}
              className="p-1.5 rounded-md hover:bg-[var(--card-elevated)] text-[var(--text-muted)] hover:text-[var(--text)] transition-colors"
            >
              <X className="w-4 h-4" />
            </button>
          </div>

          {/* Section 1: How to Use in 5 Steps */}
          <div className="space-y-3">
            <h4 className="text-xs font-semibold uppercase tracking-wider text-[var(--text-muted)] flex items-center gap-1.5">
              <BookOpen className="w-3.5 h-3.5 text-accent-indigo" />
              How to Use in 5 Simple Steps
            </h4>

            <div className="space-y-2.5 text-xs">
              <div className="p-3 rounded-md bg-[var(--card-elevated)] border border-[var(--border)] space-y-1">
                <div className="font-medium text-[var(--text)] flex items-center gap-1.5">
                  <span className="w-4 h-4 rounded-full bg-accent-indigo/20 text-accent-indigo text-[10px] font-bold flex items-center justify-center">1</span>
                  Step 1: Data
                </div>
                <p className="text-[var(--text-muted)] leading-relaxed">
                  Upload your customer CSV or XLSX. We automatically verify customer IDs, check for missing fields, and detect numerical spending patterns.
                </p>
              </div>

              <div className="p-3 rounded-md bg-[var(--card-elevated)] border border-[var(--border)] space-y-1">
                <div className="font-medium text-[var(--text)] flex items-center gap-1.5">
                  <span className="w-4 h-4 rounded-full bg-accent-indigo/20 text-accent-indigo text-[10px] font-bold flex items-center justify-center">2</span>
                  Step 2: Prepare
                </div>
                <p className="text-[var(--text-muted)] leading-relaxed">
                  Gaps are filled using KNN pattern matching, extreme outlier spikes are trimmed, and redundant features are pruned using correlations.
                </p>
              </div>

              <div className="p-3 rounded-md bg-[var(--card-elevated)] border border-[var(--border)] space-y-1">
                <div className="font-medium text-[var(--text)] flex items-center gap-1.5">
                  <span className="w-4 h-4 rounded-full bg-accent-indigo/20 text-accent-indigo text-[10px] font-bold flex items-center justify-center">3</span>
                  Step 3: Benchmark
                </div>
                <p className="text-[var(--text-muted)] leading-relaxed">
                  The system tests 16 clustering configurations across 4 algorithms to objectively pick the model that best groups your customers.
                </p>
              </div>

              <div className="p-3 rounded-md bg-[var(--card-elevated)] border border-[var(--border)] space-y-1">
                <div className="font-medium text-[var(--text)] flex items-center gap-1.5">
                  <span className="w-4 h-4 rounded-full bg-accent-indigo/20 text-accent-indigo text-[10px] font-bold flex items-center justify-center">4</span>
                  Step 4: Explore
                </div>
                <p className="text-[var(--text-muted)] leading-relaxed">
                  Inspect the 2D cluster map, check cluster sizes, and see feature spreads across groups with interactive tooltips.
                </p>
              </div>

              <div className="p-3 rounded-md bg-[var(--card-elevated)] border border-[var(--border)] space-y-1">
                <div className="font-medium text-[var(--text)] flex items-center gap-1.5">
                  <span className="w-4 h-4 rounded-full bg-accent-indigo/20 text-accent-indigo text-[10px] font-bold flex items-center justify-center">5</span>
                  Step 5: Act
                </div>
                <p className="text-[var(--text-muted)] leading-relaxed">
                  View your distinct customer personas, simulate real ROI with custom assumptions, copy marketing campaigns, or score individual leads.
                </p>
              </div>
            </div>
          </div>

          {/* Section 2: Methodology & Scientific Rigor */}
          <div className="space-y-3">
            <h4 className="text-xs font-semibold uppercase tracking-wider text-[var(--text-muted)] flex items-center gap-1.5">
              <Layers className="w-3.5 h-3.5 text-accent-indigo" />
              Methodology & Pipeline
            </h4>

            <div className="space-y-2 text-xs">
              <div className="p-3 rounded-md bg-[var(--card-elevated)] border border-[var(--border)] space-y-1.5">
                <span className="font-semibold text-accent-indigo block">4-Stage Preprocessing Pipeline</span>
                <p className="text-[var(--text-muted)] leading-relaxed">
                  1. <strong>Identifier Exclusion:</strong> Strips account/customer IDs to prevent spurious identity correlation.<br />
                  2. <strong>KNN Imputation (k=5):</strong> Multi-variate imputation using Euclidean nearest neighbors for missing numerical fields.<br />
                  3. <strong>Tukey IQR Winsorization:</strong> Extreme spikes outside [Q1 - 1.5&times;IQR, Q3 + 1.5&times;IQR] are safely clipped to fences.<br />
                  4. <strong>Z-Score Standardization:</strong> Features are scaled to zero mean and unit variance (&mu;=0, &sigma;=1).
                </p>
              </div>

              <div className="p-3 rounded-md bg-[var(--card-elevated)] border border-[var(--border)] space-y-1.5">
                <span className="font-semibold text-accent-indigo block">Feature Spaces & Mutual Information</span>
                <p className="text-[var(--text-muted)] leading-relaxed">
                  Because unsupervised clustering lacks ground-truth labels, <strong>DS3 (Mutual Information)</strong> derives pseudo-labels from an initial K-Means run on DS1 to calculate non-linear feature dependency. DS2 prunes collinear features with |r| &gt; 0.85, and DS4 retains 95% cumulative PCA variance.
                </p>
              </div>

              <div className="p-3 rounded-md bg-[var(--card-elevated)] border border-[var(--border)] space-y-1.5">
                <span className="font-semibold text-accent-indigo block">Objective Winner Selection Formula</span>
                <p className="text-[var(--text-muted)] leading-relaxed">
                  The winning model is selected via a balanced composite index:
                  <code className="block mt-1 p-1 bg-zinc-950/60 rounded text-[10px] text-accent-indigo font-mono">
                    composite_score = (norm_common_sil + (1 - norm_common_dbi) + norm_common_chi) / 3
                  </code>
                  where each metric is evaluated in the common standardized feature space (DS1) and min-max normalized across all 16 runs.
                </p>
              </div>

              <div className="p-3 rounded-md bg-[var(--card-elevated)] border border-[var(--border)] space-y-1.5">
                <span className="font-semibold text-accent-indigo block">Common-Space Clarity & Dimensional Comparability</span>
                <p className="text-[var(--text-muted)] leading-relaxed">
                  Silhouette scores calculated across spaces with different dimensionality (e.g., 17 original features vs 4 PCA components) cannot be mathematically compared directly because Euclidean distance scales with dimension count. PulseSegment AI solves this by measuring each run&apos;s discovered customer cluster boundaries in a unified, standardized original feature space (<code className="text-accent-indigo font-mono">DS1_Original</code>, sampled to 5,000 points if N &gt; 8,000). This provides an objective, apples-to-apples basis for benchmark comparison and winning model selection.
                </p>
              </div>

              <div className="p-3 rounded-md bg-[var(--card-elevated)] border border-[var(--border)] space-y-1.5">
                <span className="font-semibold text-accent-indigo block">Sampling & Execution Speed Note</span>
                <p className="text-[var(--text-muted)] leading-relaxed">
                  For large datasets where <strong>N &gt; 8,000</strong> (<code className="text-accent-indigo font-mono">SILHOUETTE_SAMPLE_THRESHOLD</code>), Silhouette is evaluated on a random subsample of exactly <strong>5,000 points</strong> (<code className="text-accent-indigo font-mono">SILHOUETTE_SAMPLE_SIZE = 5000</code>, returning <code className="text-accent-emerald font-mono">sampled=true, sample_size=5000</code> in the API response) to prevent O(N&sup2;) pairwise latency. Agglomerative clustering similarly samples if N &gt; 3,000 using NearestCentroid projection. All execution runtimes (ms) are machine-dependent and reflect dynamic wall-clock inference speed.
                </p>
              </div>
            </div>
          </div>

          {/* Section 3: Non-Jargon Glossary */}
          <div className="space-y-3">
            <h4 className="text-xs font-semibold uppercase tracking-wider text-[var(--text-muted)] flex items-center gap-1.5">
              <Sparkles className="w-3.5 h-3.5 text-accent-emerald" />
              Metric Glossary (Plain English)
            </h4>

            <div className="space-y-2 text-xs">
              <div className="border border-[var(--border)] rounded-md p-2.5">
                <span className="font-semibold text-accent-indigo">Silhouette Score</span>
                <span className="text-[10px] text-accent-emerald font-medium ml-2">Higher is better (Max: 1.0)</span>
                <p className="text-[var(--text-muted)] mt-0.5">
                  Tells you how well-separated each customer group is from the others. Higher values mean clear, distinct customer boundaries.
                </p>
              </div>

              <div className="border border-[var(--border)] rounded-md p-2.5">
                <span className="font-semibold text-accent-indigo">Davies-Bouldin Index</span>
                <span className="text-[10px] text-accent-emerald font-medium ml-2">Lower is better</span>
                <p className="text-[var(--text-muted)] mt-0.5">
                  Measures how tight each cluster is internally compared to the distance between clusters. Lower values mean compact, tightly-knit segments.
                </p>
              </div>

              <div className="border border-[var(--border)] rounded-md p-2.5">
                <span className="font-semibold text-accent-indigo">Calinski-Harabasz Index</span>
                <span className="text-[10px] text-accent-emerald font-medium ml-2">Higher is better</span>
                <p className="text-[var(--text-muted)] mt-0.5">
                  A variance-ratio score. Higher values confirm that between-group differences far exceed within-group noise.
                </p>
              </div>

              <div className="border border-[var(--border)] rounded-md p-2.5">
                <span className="font-semibold text-accent-indigo">KNN Imputation</span>
                <p className="text-[var(--text-muted)] mt-0.5">
                  Fills in missing fields by finding the 5 most similar customer records and taking their average, avoiding statistical distortion.
                </p>
              </div>

              <div className="border border-[var(--border)] rounded-md p-2.5">
                <span className="font-semibold text-accent-indigo">PCA 95% Variance</span>
                <p className="text-[var(--text-muted)] mt-0.5">
                  Condenses dozens of metrics into the few strongest independent signals that still explain 95% of customer variance.
                </p>
              </div>
            </div>
          </div>

          {/* Section 3: File Format Requirements */}
          <div className="space-y-2 text-xs">
            <h4 className="text-xs font-semibold uppercase tracking-wider text-[var(--text-muted)] flex items-center gap-1.5">
              <FileText className="w-3.5 h-3.5 text-accent-amber" />
              CSV Requirements
            </h4>
            <ul className="list-disc list-inside space-y-1 text-[var(--text-muted)] leading-relaxed">
              <li>Format: Standard CSV or Excel (.xlsx) file, up to 50MB.</li>
              <li>Structure: Exactly one row per customer.</li>
              <li>Minimums: At least 100 rows and 3 numerical columns.</li>
              <li>IDs: An identifier column (e.g., CUST_ID, User_ID) is auto-detected and excluded from clustering.</li>
            </ul>
          </div>
        </div>

        {/* Footer */}
        <div className="pt-4 mt-6 border-t border-[var(--border)] flex justify-end">
          <button
            onClick={onClose}
            className="px-4 py-2 text-xs font-medium bg-accent-indigo text-white rounded-md hover:bg-accent-indigo/90 transition-colors"
          >
            Got it, close
          </button>
        </div>
      </div>
    </div>
  )
}
