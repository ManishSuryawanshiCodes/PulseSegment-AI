'use client'

import React, { useState } from 'react'
import {
  GrowthPersonas,
  PersonaProfile,
  ExperimentsMatrix,
} from '@/lib/api'
import { formatNumber } from '@/lib/utils'
import {
  Sparkles,
  Edit2,
  Check,
  X,
  ArrowRight,
  AlertCircle,
  Layers,
  ChevronRight,
} from 'lucide-react'

interface Step2GroupsProps {
  personasData: GrowthPersonas | null
  matrix: ExperimentsMatrix | null
  onPlanCampaign: (persona: PersonaProfile) => void
  onViewMethodology: () => void
  customNames: Record<number, string>
  onUpdateCustomName: (id: number, name: string) => void
}

const PALETTE = ['#6366F1', '#10B981', '#F59E0B', '#F43F5E', '#8B5CF6', '#06B6D4', '#EC4899', '#14B8A6']

// 5-level indicator levels
function get5LevelIndicator(z: number): {
  level: string
  arrow: string
  badgeColor: string
} {
  if (z >= 1.0) {
    return {
      level: 'Much higher',
      arrow: '↑↑',
      badgeColor: 'text-emerald-500 bg-emerald-500/10 border-emerald-500/30',
    }
  }
  if (z >= 0.35) {
    return {
      level: 'Higher',
      arrow: '↑',
      badgeColor: 'text-emerald-400 bg-emerald-500/10 border-emerald-500/20',
    }
  }
  if (z <= -1.0) {
    return {
      level: 'Much lower',
      arrow: '↓↓',
      badgeColor: 'text-rose-500 bg-rose-500/10 border-rose-500/30',
    }
  }
  if (z <= -0.35) {
    return {
      level: 'Lower',
      arrow: '↓',
      badgeColor: 'text-rose-400 bg-rose-500/10 border-rose-500/20',
    }
  }
  return {
    level: 'Average',
    arrow: '↔',
    badgeColor: 'text-zinc-400 bg-zinc-500/10 border-zinc-500/20',
  }
}

// Convert trait name to friendly phrase
function formatTraitPhrase(featName: string, z: number): string {
  const cleanName = featName
    .replace(/_/g, ' ')
    .toLowerCase()
    .replace(/\b\w/g, (c) => c.toUpperCase())

  if (cleanName.includes('Balance')) {
    if (z > 0.5) return 'Carries higher balances than average'
    if (z < -0.5) return 'Maintains low or zero balances'
  }
  if (cleanName.includes('Purchase')) {
    if (z > 0.5) return 'Spends significantly more than average'
    if (z < -0.5) return 'Rarely makes everyday purchases'
  }
  if (cleanName.includes('Cash Advance')) {
    if (z > 0.5) return 'Frequently uses cash advances'
    if (z < -0.5) return 'Rarely uses cash advances'
  }
  if (cleanName.includes('Payment') || cleanName.includes('Prfull')) {
    if (z > 0.5) return 'Consistently pays off balance in full'
    if (z < -0.5) return 'Makes minimum or partial payments'
  }
  if (cleanName.includes('Limit')) {
    if (z > 0.5) return 'Assigned high spending limit'
    if (z < -0.5) return 'Has modest spending limit'
  }
  if (cleanName.includes('Tenure')) {
    if (z > 0.5) return 'Long-term active customer'
    if (z < -0.5) return 'New or recent customer account'
  }

  if (z >= 1.0) return `Much higher ${cleanName} than average`
  if (z >= 0.35) return `Above-average ${cleanName}`
  if (z <= -1.0) return `Much lower ${cleanName} than average`
  if (z <= -0.35) return `Below-average ${cleanName}`
  return `Average ${cleanName}`
}

export function Step2Groups({
  personasData,
  matrix,
  onPlanCampaign,
  onViewMethodology,
  customNames,
  onUpdateCustomName,
}: Step2GroupsProps) {
  const [editingId, setEditingId] = useState<number | null>(null)
  const [editInput, setEditInput] = useState<string>('')

  const personas = personasData?.personas || []
  const winningExp = matrix?.winning_experiment
  const groupCount = personas.length || winningExp?.cluster_count || 4

  // Group Clarity Score
  const clarityScore =
    winningExp?.common_space_silhouette ?? winningExp?.silhouette_score ?? 0.42

  let qualityLevel: 'Strong' | 'Good' | 'Fair' | 'Weak' = 'Good'
  let qualityBadgeColor = 'text-emerald-500 bg-emerald-500/10 border-emerald-500/30'
  let qualityExplanation = 'Customer groups are distinct and well-separated.'

  if (clarityScore >= 0.5) {
    qualityLevel = 'Strong'
    qualityBadgeColor = 'text-emerald-500 bg-emerald-500/10 border-emerald-500/30'
    qualityExplanation = 'Customer groups are clearly distinct with strong separation.'
  } else if (clarityScore >= 0.35) {
    qualityLevel = 'Good'
    qualityBadgeColor = 'text-emerald-400 bg-emerald-500/10 border-emerald-500/20'
    qualityExplanation = 'Customer groups have clear behavioral differences with mild overlap.'
  } else if (clarityScore >= 0.20) {
    qualityLevel = 'Fair'
    qualityBadgeColor = 'text-amber-500 bg-amber-500/10 border-amber-500/30'
    qualityExplanation = 'Customer groups share some similarities. Adding more behavioral columns like purchase frequency or recency will help sharpen distinctions.'
  } else {
    qualityLevel = 'Weak'
    qualityBadgeColor = 'text-rose-500 bg-rose-500/10 border-rose-500/30'
    qualityExplanation = 'Customer groups overlap significantly. Uploading additional customer attributes or transaction history will improve grouping clarity.'
  }

  const handleStartEdit = (p: PersonaProfile, fallbackIdx: number) => {
    const id = p.id ?? fallbackIdx
    setEditingId(id)
    setEditInput(customNames[id] || p.name)
  }

  const handleSaveEdit = (id: number) => {
    if (editInput.trim()) {
      onUpdateCustomName(id, editInput.trim())
    }
    setEditingId(null)
  }

  return (
    <div className="space-y-8 max-w-[1100px] mx-auto animate-fadeIn">
      {/* 1. Header & Quality Badge */}
      <div className="bg-[var(--card)] border border-[var(--border)] rounded-2xl p-6 sm:p-7 shadow-sm space-y-4">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
          <div>
            <h2 className="text-2xl sm:text-3xl font-bold tracking-tight text-[var(--text)]">
              We found {groupCount} customer groups
            </h2>
            <p className="text-sm text-[var(--text-muted)] mt-1">
              Discovered automatically by comparing grouping methods on your data.
            </p>
          </div>

          <div className="flex items-center gap-2">
            <span
              className={`px-3 py-1.5 rounded-full text-xs font-semibold border flex items-center gap-1.5 ${qualityBadgeColor}`}
            >
              <span className="w-2 h-2 rounded-full bg-current" />
              {qualityLevel} grouping quality
            </span>
          </div>
        </div>

        {/* Quality Explanation Banner */}
        <div className="p-3.5 rounded-xl bg-[var(--card-elevated)] border border-[var(--border)] flex items-start gap-2.5 text-xs text-[var(--text-muted)]">
          {qualityLevel === 'Fair' || qualityLevel === 'Weak' ? (
            <AlertCircle className="w-4 h-4 text-amber-500 shrink-0 mt-0.5" />
          ) : (
            <Sparkles className="w-4 h-4 text-accent-indigo shrink-0 mt-0.5" />
          )}
          <span>{qualityExplanation}</span>
        </div>

        {/* 2. Single Horizontal Bar of Group Sizes */}
        <div className="space-y-2 pt-2">
          <div className="flex items-center justify-between text-xs text-[var(--text-muted)]">
            <span className="font-medium text-[var(--text)]">Customer Group Distribution</span>
            <span>{personas.reduce((a, b) => a + b.count, 0).toLocaleString()} total customers</span>
          </div>

          <div className="w-full h-7 rounded-xl overflow-hidden flex border border-[var(--border)] bg-zinc-900/10 shadow-inner">
            {personas.map((p, idx) => {
              const color = PALETTE[idx % PALETTE.length]
              const name = customNames[p.id ?? idx] || p.name
              const pct = p.percentage
              return (
                <div
                  key={p.id ?? idx}
                  style={{ width: `${Math.max(pct, 5)}%`, backgroundColor: color }}
                  className="h-full relative group transition-all flex items-center justify-center px-1 text-white text-[11px] font-mono font-medium truncate"
                  title={`${name}: ${p.count.toLocaleString()} customers (${pct.toFixed(1)}%)`}
                >
                  <span className="truncate drop-shadow-sm">
                    {pct >= 10 ? `${pct.toFixed(0)}%` : ''}
                  </span>
                </div>
              )
            })}
          </div>

          {/* Group legend pills */}
          <div className="flex flex-wrap items-center gap-3 pt-1">
            {personas.map((p, idx) => {
              const color = PALETTE[idx % PALETTE.length]
              const name = customNames[p.id ?? idx] || p.name
              return (
                <div key={p.id ?? idx} className="flex items-center gap-1.5 text-xs">
                  <span className="w-2.5 h-2.5 rounded-full shrink-0" style={{ backgroundColor: color }} />
                  <span className="text-[var(--text)] font-medium">{name}</span>
                  <span className="text-[var(--text-muted)] font-mono text-[11px]">
                    ({p.percentage.toFixed(1)}%)
                  </span>
                </div>
              )
            })}
          </div>
        </div>
      </div>

      {/* 3. Cards Grid (4 across, then 2, then 1) */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 sm:gap-5">
        {personas.map((p, idx) => {
          const id = p.id ?? idx
          const name = customNames[id] || p.name
          const color = PALETTE[idx % PALETTE.length]

          // Distinguishing Traits (Top 3 traits with 5-level indicators)
          const traitList: Array<{ name: string; z: number }> = []
          if (p.top_positive_features && p.top_positive_features.length > 0) {
            p.top_positive_features.slice(0, 2).forEach((feat) => {
              const z = feat.z_score ?? (p.centroid_profile[feat.feature] ?? 1.2)
              traitList.push({ name: feat.feature, z })
            })
          }
          if (p.top_negative_features && p.top_negative_features.length > 0) {
            p.top_negative_features.slice(0, 1).forEach((feat) => {
              const z = feat.z_score ?? (p.centroid_profile[feat.feature] ?? -1.1)
              traitList.push({ name: feat.feature, z })
            })
          }
          if (traitList.length === 0) {
            Object.entries(p.centroid_profile || {})
              .slice(0, 3)
              .forEach(([feat, z]) => {
                traitList.push({ name: feat, z })
              })
          }

          // Real Units Line
          const realUnits = p.real_unit_centroid
          let realUnitString = ''
          if (realUnits && Object.keys(realUnits).length > 0) {
            const keys = Object.keys(realUnits).slice(0, 3)
            realUnitString = keys
              .map((k) => {
                const val = realUnits[k]
                const cleanKey = k.replace(/_/g, ' ').toLowerCase()
                const formatted = typeof val === 'number' ? Math.round(val).toLocaleString() : val
                return `${formatted} ${cleanKey}`
              })
              .join(' • ')
          }

          // 1-sentence description
          const desc =
            p.characteristics?.[0] ||
            p.distinguishing_note ||
            'Key customer behavioral group with distinct spending and payment traits.'

          return (
            <div
              key={id}
              className="bg-[var(--card)] border border-[var(--border)] rounded-2xl p-5 shadow-sm flex flex-col justify-between gap-4 hover:border-accent-indigo/40 transition-all card-hover"
            >
              <div className="space-y-3.5">
                {/* Color bar top accent */}
                <div className="h-1 rounded-full w-12" style={{ backgroundColor: color }} />

                {/* Group Name (Inline Editable) */}
                <div>
                  {editingId === id ? (
                    <div className="flex items-center gap-1.5">
                      <input
                        type="text"
                        value={editInput}
                        onChange={(e) => setEditInput(e.target.value)}
                        className="w-full text-sm font-semibold bg-[var(--card-elevated)] border border-accent-indigo rounded px-2 py-1 text-[var(--text)] focus:outline-none"
                        autoFocus
                      />
                      <button
                        type="button"
                        onClick={() => handleSaveEdit(id)}
                        className="p-1 rounded bg-accent-indigo text-white hover:bg-accent-indigo/90"
                      >
                        <Check className="w-3.5 h-3.5" />
                      </button>
                      <button
                        type="button"
                        onClick={() => setEditingId(null)}
                        className="p-1 rounded bg-[var(--card-elevated)] text-[var(--text-muted)] hover:text-[var(--text)]"
                      >
                        <X className="w-3.5 h-3.5" />
                      </button>
                    </div>
                  ) : (
                    <div className="flex items-center justify-between group">
                      <h3 className="text-base font-bold text-[var(--text)] truncate">
                        {name}
                      </h3>
                      <button
                        type="button"
                        onClick={() => handleStartEdit(p, idx)}
                        className="p-1 rounded hover:bg-[var(--card-elevated)] text-[var(--text-muted)] hover:text-accent-indigo transition-colors"
                        title="Edit group name"
                      >
                        <Edit2 className="w-3.5 h-3.5" />
                      </button>
                    </div>
                  )}

                  {/* Customer count and percentage */}
                  <div className="flex items-center gap-2 mt-1">
                    <span className="text-xs font-semibold metric-value text-[var(--text)]">
                      {p.count.toLocaleString()} customers
                    </span>
                    <span className="text-xs text-[var(--text-muted)] font-mono">
                      ({p.percentage.toFixed(1)}%)
                    </span>
                  </div>
                </div>

                {/* 1-Sentence Description */}
                <p className="text-xs text-[var(--text-muted)] leading-relaxed min-h-[32px]">
                  {desc}
                </p>

                {/* 3 Plain Sentences about What Makes Them Different with 5-Level Arrow Indicators */}
                <div className="space-y-2 pt-2 border-t border-[var(--border)]">
                  <span className="text-[11px] font-semibold text-[var(--text-muted)] uppercase tracking-wider block">
                    What makes them different
                  </span>

                  <div className="space-y-2">
                    {traitList.map((t, tIdx) => {
                      const indicator = get5LevelIndicator(t.z)
                      const phrase = formatTraitPhrase(t.name, t.z)
                      return (
                        <div key={tIdx} className="text-xs flex items-start gap-2">
                          <span
                            className={`px-1.5 py-0.2 rounded border font-mono text-[10px] font-bold shrink-0 mt-0.5 ${indicator.badgeColor}`}
                            title={`Compared with average: ${indicator.level}`}
                          >
                            {indicator.arrow}
                          </span>
                          <span className="text-[var(--text)] leading-snug">
                            {phrase}
                          </span>
                        </div>
                      )
                    })}
                  </div>
                </div>

                {/* Small "Typical customer" line in real units */}
                {realUnitString && (
                  <div className="p-2.5 rounded-xl bg-[var(--card-elevated)] border border-[var(--border)] text-[11px]">
                    <span className="font-medium text-[var(--text-muted)] block">Typical customer:</span>
                    <span className="text-[var(--text)] font-mono mt-0.5 block truncate" title={realUnitString}>
                      {realUnitString}
                    </span>
                  </div>
                )}
              </div>

              {/* "Plan campaign" Button */}
              <div className="pt-2 border-t border-[var(--border)]">
                <button
                  type="button"
                  onClick={() => onPlanCampaign(p)}
                  className="w-full py-2.5 px-3 rounded-xl bg-[var(--card-elevated)] hover:bg-accent-indigo hover:text-white border border-[var(--border)] hover:border-accent-indigo text-xs font-semibold text-[var(--text)] transition-all flex items-center justify-center gap-1.5 cursor-pointer shadow-sm"
                >
                  <span>Plan campaign</span>
                  <ArrowRight className="w-3.5 h-3.5" />
                </button>
              </div>
            </div>
          )
        })}
      </div>

      {/* Link to Methodology */}
      <div className="p-4 rounded-2xl bg-[var(--card)] border border-[var(--border)] flex flex-col sm:flex-row items-center justify-between gap-3 text-xs text-[var(--text-muted)]">
        <div className="flex items-center gap-2">
          <Layers className="w-4 h-4 text-accent-indigo shrink-0" />
          <span>Want to see how data was cleaned, compared across 16 methods, and verified?</span>
        </div>
        <button
          type="button"
          onClick={onViewMethodology}
          className="text-accent-indigo font-semibold hover:underline flex items-center gap-1 shrink-0"
        >
          <span>View how we got these results</span>
          <ChevronRight className="w-4 h-4" />
        </button>
      </div>
    </div>
  )
}
