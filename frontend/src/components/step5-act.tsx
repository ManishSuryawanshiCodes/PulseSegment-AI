'use client'

import React, { useState } from 'react'
import {
  GrowthPersonas,
  PersonaProfile,
  PredictResponse,
  api,
} from '@/lib/api'
import { formatNumber } from '@/lib/utils'
import {
  Sparkles,
  Edit2,
  Check,
  X,
  Copy,
  Download,
  Target,
  Send,
  Mail,
  MessageSquare,
  Bell,
  TrendingUp,
  DollarSign,
  UserCheck,
  ArrowRight,
  Info,
  Layers,
  RefreshCw,
  Sliders,
} from 'lucide-react'

interface Step5ActProps {
  personasData: GrowthPersonas | null
  datasetId: string
  numericFeatures: string[]
}

const PALETTE = ['#6366F1', '#10B981', '#F59E0B', '#F43F5E', '#8B5CF6', '#06B6D4', '#EC4899', '#14B8A6']

export function Step5Act({
  personasData,
  datasetId,
  numericFeatures,
}: Step5ActProps) {
  // Personas state with inline editable names
  const [customNames, setCustomNames] = useState<Record<number, string>>({})
  const [editingId, setEditingId] = useState<number | null>(null)
  const [editInput, setEditInput] = useState<string>('')

  // Campaign Drawer state
  const [activePersona, setActivePersona] = useState<PersonaProfile | null>(null)
  const [activeTab, setActiveTab] = useState<'strategy' | 'email' | 'sms' | 'push'>('strategy')
  const [copiedKey, setCopiedKey] = useState<string | null>(null)

  // Interactive ROI Simulator State
  const [avgRevenue, setAvgRevenue] = useState<number>(120.0)
  const [expectedUplift, setExpectedUplift] = useState<number>(15.0) // %
  const [campaignCost, setCampaignCost] = useState<number>(3.50) // $ per customer

  // Single Customer Scoring Modal State
  const [scoreModalOpen, setScoreModalOpen] = useState<boolean>(false)
  const [featureInputs, setFeatureInputs] = useState<Record<string, number>>({})
  const [scoringLoading, setScoringLoading] = useState<boolean>(false)
  const [scoreResult, setScoreResult] = useState<PredictResponse | null>(null)

  const personas = personasData?.personas || []

  // Initialize feature inputs with reasonable defaults when opening scoring modal
  const openScoreModal = () => {
    const initial: Record<string, number> = {}
    numericFeatures.forEach((feat) => {
      // Use average from first persona or 100
      const sampleVal = personas[0]?.centroid_profile[feat] ?? 100
      initial[feat] = Math.round(sampleVal)
    })
    setFeatureInputs(initial)
    setScoreResult(null)
    setScoreModalOpen(true)
  }

  const handlePredict = async (e: React.FormEvent) => {
    e.preventDefault()
    setScoringLoading(true)
    try {
      const res = await api.predictSegment(datasetId, featureInputs)
      setScoreResult(res)
    } catch (err: any) {
      alert(`Prediction failed: ${err.message}`)
    } finally {
      setScoringLoading(false)
    }
  }

  const handleSaveName = (id: number) => {
    if (editInput.trim()) {
      setCustomNames((prev) => ({ ...prev, [id]: editInput.trim() }))
    }
    setEditingId(null)
  }

  const handleCopy = (text: string, key: string) => {
    navigator.clipboard.writeText(text)
    setCopiedKey(key)
    setTimeout(() => setCopiedKey(null), 2000)
  }

  // Dynamic ROI Calculations for Active Persona
  const activeCount = activePersona?.count || 1000
  const grossLiftDollars = activeCount * avgRevenue * (expectedUplift / 100)
  const totalCampaignCost = activeCount * campaignCost
  const netLiftDollars = Math.max(0, grossLiftDollars - totalCampaignCost)
  const roiMultiple = totalCampaignCost > 0 ? (grossLiftDollars / totalCampaignCost).toFixed(1) : '1.0'

  return (
    <div className="space-y-8 animate-fadeIn">
      {/* Action Bar (Top) */}
      <div className="bg-[var(--card)] border border-[var(--border)] rounded-xl p-4 shadow-sm flex flex-wrap items-center justify-between gap-3">
        <div>
          <h2 className="text-sm font-semibold text-[var(--text)] flex items-center gap-1.5">
            <Target className="w-4 h-4 text-accent-indigo" />
            Customer Growth Intelligence & Activation
          </h2>
          <div className="flex items-center gap-2 mt-1">
            <span className="text-[11px] font-mono px-2 py-0.5 rounded bg-[var(--card-elevated)] border border-[var(--border)] text-accent-indigo">
              Showing: {personasData?.winning_model || 'Winning Model'}
            </span>
          </div>
          <p className="text-xs text-[var(--text-muted)] mt-1">
            Target high-propensity segments with personalized campaigns and deploy customer scoring.
          </p>
        </div>

        <div className="flex flex-wrap items-center gap-2">
          {/* Score One Customer */}
          <button
            type="button"
            onClick={openScoreModal}
            className="px-3 py-1.5 rounded-lg bg-[var(--card-elevated)] border border-[var(--border)] hover:border-accent-indigo/50 text-xs font-medium text-[var(--text)] transition-colors flex items-center gap-1.5"
          >
            <UserCheck className="w-3.5 h-3.5 text-accent-indigo" />
            Score One Customer
          </button>

          {/* Export Segments CSV */}
          <a
            href={api.exportSegmentsUrl(datasetId)}
            download
            className="px-3 py-1.5 rounded-lg bg-[var(--card-elevated)] border border-[var(--border)] hover:border-accent-emerald/50 text-xs font-medium text-[var(--text)] transition-colors flex items-center gap-1.5"
          >
            <Download className="w-3.5 h-3.5 text-accent-emerald" />
            Export Segments CSV
          </a>

          {/* Download Model Parameters JSON */}
          <a
            href={api.exportModelJsonUrl(datasetId)}
            download
            className="px-3 py-1.5 rounded-lg bg-[var(--card-elevated)] border border-[var(--border)] hover:border-accent-indigo/50 text-xs font-medium text-[var(--text)] transition-colors flex items-center gap-1.5"
            title="Download safe, auditable JSON parameters of centroids, scaler, and PCA components"
          >
            <Download className="w-3.5 h-3.5 text-accent-indigo" />
            Model Parameters (.json)
          </a>

          {/* Download Model Pickle */}
          <a
            href={api.exportModelUrl(datasetId)}
            download
            className="px-3 py-1.5 rounded-lg bg-[var(--card-elevated)] border border-[var(--border)] hover:border-accent-indigo/50 text-xs font-medium text-[var(--text)] transition-colors flex items-center gap-1.5"
          >
            <Download className="w-3.5 h-3.5 text-accent-indigo" />
            Model (.joblib)
          </a>
        </div>
      </div>

      {/* Persona Cards Grid (4-up on desktop, 2-up on tablet, 1-up on mobile) */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        {personas.map((persona, idx) => {
          const personaId = persona.id ?? idx
          const displayName = customNames[personaId] || persona.name
          const color = PALETTE[idx % PALETTE.length]
          const isEditing = editingId === personaId

          // Collect top 6 distinguishing features (positives & negatives) for diverging bar chart
          const posFeatures = persona.top_positive_features || []
          const negFeatures = persona.top_negative_features || []
          const allTraits = [...posFeatures, ...negFeatures].slice(0, 6)

          return (
            <div
              key={personaId}
              className="p-5 rounded-2xl bg-[var(--card)] border border-[var(--border)] shadow-sm hover:border-accent-indigo/50 transition-all flex flex-col justify-between space-y-4"
            >
              <div className="space-y-3">
                {/* Header: Color chip + Editable Title */}
                <div className="flex items-start justify-between gap-2">
                  <div className="flex items-center gap-2 flex-1 min-w-0">
                    <span
                      className="w-3 h-3 rounded-full shrink-0"
                      style={{ backgroundColor: color }}
                    />
                    {isEditing ? (
                      <div className="flex items-center gap-1 flex-1">
                        <input
                          type="text"
                          value={editInput}
                          onChange={(e) => setEditInput(e.target.value)}
                          className="w-full text-xs font-bold bg-[var(--card-elevated)] border border-accent-indigo rounded px-1.5 py-0.5 text-[var(--text)] focus:outline-none"
                          autoFocus
                        />
                        <button
                          type="button"
                          onClick={() => handleSaveName(personaId)}
                          className="text-accent-emerald hover:opacity-80 p-0.5"
                        >
                          <Check className="w-3.5 h-3.5" />
                        </button>
                        <button
                          type="button"
                          onClick={() => setEditingId(null)}
                          className="text-[var(--text-muted)] hover:opacity-80 p-0.5"
                        >
                          <X className="w-3.5 h-3.5" />
                        </button>
                      </div>
                    ) : (
                      <div className="flex items-center gap-1.5 min-w-0 flex-1">
                        <h3 className="font-bold text-sm text-[var(--text)] truncate" title={displayName}>
                          {displayName}
                        </h3>
                        <button
                          type="button"
                          onClick={() => {
                            setEditingId(personaId)
                            setEditInput(displayName)
                          }}
                          className="text-[var(--text-muted)] hover:text-accent-indigo opacity-0 group-hover:opacity-100 transition-opacity p-0.5"
                          title="Rename segment"
                        >
                          <Edit2 className="w-3 h-3" />
                        </button>
                      </div>
                    )}
                  </div>

                  <span className="text-[10px] px-1.5 py-0.5 rounded bg-[var(--card-elevated)] border border-[var(--border)] text-[var(--text-muted)] font-mono shrink-0">
                    #{personaId}
                  </span>
                </div>

                {/* Size & Percentage */}
                <div className="flex items-baseline justify-between border-b border-[var(--border)] pb-2.5">
                  <span className="text-xl font-bold metric-value text-[var(--text)]">
                    {formatNumber(persona.count)}
                  </span>
                  <span className="text-xs font-mono font-semibold text-accent-indigo">
                    {persona.percentage.toFixed(1)}% of base
                  </span>
                </div>

                {/* One-Line Description */}
                <p className="text-xs text-[var(--text-muted)] leading-relaxed min-h-[36px]">
                  {persona.characteristics?.[0] || 'Distinct customer behavior group discovered through clustering boundaries.'}
                </p>

                {/* Diverging Centroid Bar Chart (Z-Scores vs Global Average) */}
                <div className="space-y-1.5 pt-1">
                  <span className="text-[10px] font-semibold uppercase tracking-wider text-[var(--text-muted)] block">
                    Distinguishing Behavior (Z-Scores vs. Avg)
                  </span>

                  <div className="space-y-1 text-[11px] font-mono">
                    {allTraits.length < 2 ? (
                      <div className="py-2.5 px-3 rounded-lg bg-[var(--card-elevated)] border border-[var(--border)] text-[11px] text-[var(--text-muted)] italic font-sans leading-relaxed">
                        {persona.distinguishing_note || 'This group differs only slightly from average.'}
                      </div>
                    ) : (
                      allTraits.map((t) => {
                        const isPos = t.z_score >= 0
                        const barWidth = Math.min(100, Math.abs(t.z_score) * 45)

                        return (
                          <div key={t.feature} className="space-y-0.5">
                            <div className="flex justify-between text-[10px] text-[var(--text-muted)]">
                              <span className="truncate max-w-[120px]">{t.feature}</span>
                              <span className={isPos ? 'text-accent-emerald font-semibold' : 'text-rose-400 font-semibold'}>
                                {t.z_score > 0 ? `+${t.z_score.toFixed(2)}σ` : `${t.z_score.toFixed(2)}σ`}
                              </span>
                            </div>

                            {/* Diverging Bar: Left for negative, Right for positive */}
                            <div className="w-full bg-[var(--border)] h-1.5 rounded-full flex relative overflow-hidden">
                              <div className="w-1/2 flex justify-end">
                                {!isPos && (
                                  <div
                                    className="h-full bg-rose-500 rounded-l-full"
                                    style={{ width: `${barWidth}%` }}
                                  />
                                )}
                              </div>
                              <div className="w-px bg-[var(--text-muted)] h-full z-10" />
                              <div className="w-1/2 flex justify-start">
                                {isPos && (
                                  <div
                                    className="h-full bg-emerald-500 rounded-r-full"
                                    style={{ width: `${barWidth}%` }}
                                  />
                                )}
                              </div>
                            </div>
                          </div>
                        )
                      })
                    )}
                  </div>
                </div>
              </div>

              {/* Primary Action Button */}
              <button
                type="button"
                onClick={() => {
                  setActivePersona(persona)
                  setActiveTab('strategy')
                }}
                className="w-full py-2 text-xs font-semibold rounded-lg bg-accent-indigo text-white hover:bg-accent-indigo/90 transition-colors flex items-center justify-center gap-1.5 shadow-sm cursor-pointer"
              >
                Plan campaign
                <ArrowRight className="w-3.5 h-3.5" />
              </button>
            </div>
          )
        })}
      </div>

      {/* Campaign Drawer / Slide-Over Modal */}
      {activePersona && (
        <div className="fixed inset-0 z-50 flex justify-end">
          <div
            className="fixed inset-0 bg-black/60 backdrop-blur-sm transition-opacity"
            onClick={() => setActivePersona(null)}
          />

          <div className="relative w-full max-w-xl bg-[var(--card)] border-l border-[var(--border)] h-full overflow-y-auto p-6 shadow-2xl flex flex-col justify-between z-10 text-[var(--text)]">
            <div className="space-y-6">
              {/* Drawer Header */}
              <div className="flex items-start justify-between pb-4 border-b border-[var(--border)]">
                <div>
                  <div className="flex items-center gap-2">
                    <span className="w-3 h-3 rounded-full bg-accent-indigo" />
                    <h3 className="font-bold text-base">
                      {customNames[activePersona.id ?? 0] || activePersona.name}
                    </h3>
                  </div>
                  <p className="text-xs text-[var(--text-muted)] mt-1">
                    {formatNumber(activePersona.count)} customers ({activePersona.percentage.toFixed(1)}% of total portfolio)
                  </p>
                </div>
                <button
                  onClick={() => setActivePersona(null)}
                  className="p-1.5 rounded-md hover:bg-[var(--card-elevated)] text-[var(--text-muted)] hover:text-[var(--text)] transition-colors"
                >
                  <X className="w-4 h-4" />
                </button>
              </div>

              {/* Dynamic ROI Simulator Panel */}
              <div className="p-4 rounded-xl bg-[var(--card-elevated)] border border-[var(--border)] space-y-3">
                <div className="flex items-center justify-between">
                  <span className="text-xs font-bold uppercase tracking-wider text-[var(--text)] flex items-center gap-1.5">
                    <TrendingUp className="w-4 h-4 text-accent-emerald" />
                    Interactive Campaign ROI Simulator
                  </span>
                  <span className="text-[11px] font-mono text-accent-emerald font-bold">
                    Projected: {roiMultiple}x ROI
                  </span>
                </div>

                <div className="grid grid-cols-3 gap-3">
                  <div>
                    <label className="text-[10px] text-[var(--text-muted)] block">Avg Revenue ($)</label>
                    <input
                      type="number"
                      value={avgRevenue}
                      onChange={(e) => setAvgRevenue(parseFloat(e.target.value) || 0)}
                      className="w-full text-xs font-mono font-bold bg-[var(--card)] border border-[var(--border)] rounded px-2 py-1 mt-0.5 text-[var(--text)]"
                    />
                  </div>

                  <div>
                    <label className="text-[10px] text-[var(--text-muted)] block">Expected Uplift (%)</label>
                    <input
                      type="number"
                      value={expectedUplift}
                      onChange={(e) => setExpectedUplift(parseFloat(e.target.value) || 0)}
                      className="w-full text-xs font-mono font-bold bg-[var(--card)] border border-[var(--border)] rounded px-2 py-1 mt-0.5 text-[var(--text)]"
                    />
                  </div>

                  <div>
                    <label className="text-[10px] text-[var(--text-muted)] block">Cost / User ($)</label>
                    <input
                      type="number"
                      value={campaignCost}
                      onChange={(e) => setCampaignCost(parseFloat(e.target.value) || 0)}
                      className="w-full text-xs font-mono font-bold bg-[var(--card)] border border-[var(--border)] rounded px-2 py-1 mt-0.5 text-[var(--text)]"
                    />
                  </div>
                </div>

                {/* Net Projection Box */}
                <div className="pt-2 border-t border-[var(--border)] flex items-center justify-between text-xs">
                  <span className="text-[var(--text-muted)]">Estimated Net Incremental Revenue:</span>
                  <span className="font-mono font-bold text-accent-emerald text-sm">
                    +${formatNumber(Math.round(netLiftDollars))}
                  </span>
                </div>
              </div>

              {/* Channel Tabs */}
              <div className="space-y-4">
                <div className="flex items-center gap-1 bg-[var(--card-elevated)] p-1 rounded-lg border border-[var(--border)] text-xs font-medium">
                  <button
                    type="button"
                    onClick={() => setActiveTab('strategy')}
                    className={`flex-1 py-1.5 rounded transition-colors flex items-center justify-center gap-1.5 ${
                      activeTab === 'strategy' ? 'bg-accent-indigo text-white font-semibold' : 'text-[var(--text-muted)] hover:text-[var(--text)]'
                    }`}
                  >
                    <Target className="w-3.5 h-3.5" />
                    Strategy
                  </button>
                  <button
                    type="button"
                    onClick={() => setActiveTab('email')}
                    className={`flex-1 py-1.5 rounded transition-colors flex items-center justify-center gap-1.5 ${
                      activeTab === 'email' ? 'bg-accent-indigo text-white font-semibold' : 'text-[var(--text-muted)] hover:text-[var(--text)]'
                    }`}
                  >
                    <Mail className="w-3.5 h-3.5" />
                    Email
                  </button>
                  <button
                    type="button"
                    onClick={() => setActiveTab('sms')}
                    className={`flex-1 py-1.5 rounded transition-colors flex items-center justify-center gap-1.5 ${
                      activeTab === 'sms' ? 'bg-accent-indigo text-white font-semibold' : 'text-[var(--text-muted)] hover:text-[var(--text)]'
                    }`}
                  >
                    <MessageSquare className="w-3.5 h-3.5" />
                    SMS
                  </button>
                  <button
                    type="button"
                    onClick={() => setActiveTab('push')}
                    className={`flex-1 py-1.5 rounded transition-colors flex items-center justify-center gap-1.5 ${
                      activeTab === 'push' ? 'bg-accent-indigo text-white font-semibold' : 'text-[var(--text-muted)] hover:text-[var(--text)]'
                    }`}
                  >
                    <Bell className="w-3.5 h-3.5" />
                    Push
                  </button>
                </div>

                {/* Tab Content */}
                {activeTab === 'strategy' && (
                  <div className="space-y-3">
                    <div className="p-4 rounded-lg bg-[var(--card-elevated)] border border-[var(--border)] space-y-2">
                      <span className="text-xs font-semibold text-accent-indigo block">
                        Recommended Next-Best-Action (NBA)
                      </span>
                      <h4 className="text-sm font-bold text-[var(--text)]">
                        {activePersona.nba_campaigns?.[0]?.title || 'Segment Acceleration Playbook'}
                      </h4>
                      <p className="text-xs text-[var(--text-muted)] leading-relaxed">
                        {activePersona.nba_campaigns?.[0]?.description ||
                          'Target this cohort with category-specific bonus incentives aligned with their distinguishing behavioral traits.'}
                      </p>
                      <div className="pt-2 flex items-center justify-between text-xs text-[var(--text-muted)] font-mono">
                        <span>Preferred Channel: {activePersona.nba_campaigns?.[0]?.channel || 'Omnichannel'}</span>
                        <span className="text-accent-emerald font-bold">
                          {activePersona.nba_campaigns?.[0]?.expected_roi_lift || '+18% Expected Lift'}
                        </span>
                      </div>
                    </div>
                  </div>
                )}

                {activeTab === 'email' && (
                  <div className="p-4 rounded-lg bg-[var(--card-elevated)] border border-[var(--border)] space-y-3">
                    <div className="flex items-center justify-between">
                      <span className="text-xs font-semibold text-[var(--text)]">Email Campaign Template</span>
                      <button
                        type="button"
                        onClick={() =>
                          handleCopy(
                            `Subject: Exclusive privilege curated for you\n\nDear valued member,\n\nBased on your active activity with us, we are delighted to unlock an exclusive priority privilege...`,
                            'email'
                          )
                        }
                        className="text-xs text-accent-indigo hover:underline flex items-center gap-1 font-medium"
                      >
                        {copiedKey === 'email' ? <Check className="w-3.5 h-3.5 text-accent-emerald" /> : <Copy className="w-3.5 h-3.5" />}
                        {copiedKey === 'email' ? 'Copied!' : 'Copy to clipboard'}
                      </button>
                    </div>
                    <div className="font-mono text-xs text-[var(--text-muted)] bg-[var(--card)] p-3 rounded border border-[var(--border)] leading-relaxed">
                      <strong className="text-[var(--text)] block mb-1">Subject: Exclusive priority invitation</strong>
                      Dear Member,
                      <br /><br />
                      As one of our most valued {customNames[activePersona.id ?? 0] || activePersona.name} accounts, you have been selected for an accelerated 0% fee rate on your next transactions.
                      <br /><br />
                      Claim your personalized privilege within the mobile app before month-end.
                    </div>
                  </div>
                )}

                {activeTab === 'sms' && (
                  <div className="p-4 rounded-lg bg-[var(--card-elevated)] border border-[var(--border)] space-y-3">
                    <div className="flex items-center justify-between">
                      <span className="text-xs font-semibold text-[var(--text)]">SMS Quick Alert Template</span>
                      <button
                        type="button"
                        onClick={() =>
                          handleCopy(
                            `[Priority Alert] Exclusive offer: enjoy special rewards on your account this week. Tap to activate: app.link/activate`,
                            'sms'
                          )
                        }
                        className="text-xs text-accent-indigo hover:underline flex items-center gap-1 font-medium"
                      >
                        {copiedKey === 'sms' ? <Check className="w-3.5 h-3.5 text-accent-emerald" /> : <Copy className="w-3.5 h-3.5" />}
                        {copiedKey === 'sms' ? 'Copied!' : 'Copy to clipboard'}
                      </button>
                    </div>
                    <div className="font-mono text-xs text-[var(--text-muted)] bg-[var(--card)] p-3 rounded border border-[var(--border)] leading-relaxed">
                      [Priority Alert] Exclusive rewards unlocked for your account. Tap to activate before expiry: pulsesegment.io/rewards
                    </div>
                  </div>
                )}

                {activeTab === 'push' && (
                  <div className="p-4 rounded-lg bg-[var(--card-elevated)] border border-[var(--border)] space-y-3">
                    <div className="flex items-center justify-between">
                      <span className="text-xs font-semibold text-[var(--text)]">In-App Push Notification</span>
                      <button
                        type="button"
                        onClick={() =>
                          handleCopy(
                            `Your exclusive rewards tier has been upgraded! Open the app to view your updated perks.`,
                            'push'
                          )
                        }
                        className="text-xs text-accent-indigo hover:underline flex items-center gap-1 font-medium"
                      >
                        {copiedKey === 'push' ? <Check className="w-3.5 h-3.5 text-accent-emerald" /> : <Copy className="w-3.5 h-3.5" />}
                        {copiedKey === 'push' ? 'Copied!' : 'Copy to clipboard'}
                      </button>
                    </div>
                    <div className="font-mono text-xs text-[var(--text-muted)] bg-[var(--card)] p-3 rounded border border-[var(--border)] leading-relaxed">
                      🌟 Special benefit: New tailored rewards are ready in your dashboard. Tap to view your perks.
                    </div>
                  </div>
                )}
              </div>

              {/* AI Campaign Regenerate button */}
              <div className="pt-2">
                <button
                  type="button"
                  onClick={() => alert('LLM generation key not configured. Using enriched deterministic domain templates.')}
                  className="w-full py-2.5 rounded-lg border border-[var(--border)] bg-[var(--card-elevated)] hover:border-accent-indigo text-xs font-medium text-[var(--text)] flex items-center justify-center gap-2 transition-colors cursor-pointer"
                >
                  <Sparkles className="w-4 h-4 text-accent-indigo" />
                  Regenerate with AI (LLM Prompting)
                </button>
              </div>
            </div>

            <div className="pt-4 border-t border-[var(--border)] flex justify-end">
              <button
                onClick={() => setActivePersona(null)}
                className="px-4 py-2 text-xs font-medium bg-accent-indigo text-white rounded-md hover:bg-accent-indigo/90 transition-colors cursor-pointer"
              >
                Done
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Single Customer Scoring Modal */}
      {scoreModalOpen && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4">
          <div
            className="fixed inset-0 bg-black/60 backdrop-blur-sm"
            onClick={() => setScoreModalOpen(false)}
          />

          <div className="relative w-full max-w-lg bg-[var(--card)] border border-[var(--border)] rounded-2xl p-6 shadow-2xl z-10 space-y-5 text-[var(--text)] max-h-[90vh] overflow-y-auto">
            <div className="flex items-center justify-between pb-3 border-b border-[var(--border)]">
              <div className="flex items-center gap-2">
                <UserCheck className="w-5 h-5 text-accent-indigo" />
                <h3 className="text-sm font-bold">Score One Customer</h3>
              </div>
              <button
                onClick={() => setScoreModalOpen(false)}
                className="p-1 rounded hover:bg-[var(--card-elevated)] text-[var(--text-muted)] hover:text-[var(--text)]"
              >
                <X className="w-4 h-4" />
              </button>
            </div>

            <form onSubmit={handlePredict} className="space-y-4">
              <p className="text-xs text-[var(--text-muted)]">
                Input behavioral metrics for an individual prospect or customer to classify them into their predicted persona in real time.
              </p>

              <div className="grid grid-cols-2 gap-3 max-h-56 overflow-y-auto pr-1">
                {numericFeatures.map((feat) => (
                  <div key={feat} className="space-y-1">
                    <label className="text-[11px] text-[var(--text-muted)] font-mono block truncate" title={feat}>
                      {feat}
                    </label>
                    <input
                      type="number"
                      step="any"
                      value={featureInputs[feat] ?? 0}
                      onChange={(e) =>
                        setFeatureInputs({
                          ...featureInputs,
                          [feat]: parseFloat(e.target.value) || 0,
                        })
                      }
                      className="w-full text-xs font-mono bg-[var(--card-elevated)] border border-[var(--border)] rounded px-2.5 py-1.5 text-[var(--text)] focus:outline-none focus:border-accent-indigo"
                    />
                  </div>
                ))}
              </div>

              <button
                type="submit"
                disabled={scoringLoading}
                className="w-full py-2.5 rounded-lg bg-accent-indigo hover:bg-accent-indigo/90 text-white font-semibold text-xs transition-colors flex items-center justify-center gap-2 cursor-pointer disabled:opacity-50"
              >
                {scoringLoading ? (
                  <>
                    <RefreshCw className="w-4 h-4 animate-spin" />
                    Calculating cluster distance...
                  </>
                ) : (
                  <>
                    <Sparkles className="w-4 h-4" />
                    Classify Customer Persona
                  </>
                )}
              </button>
            </form>

            {/* Inference Result Card */}
            {scoreResult && (
              <div className="mt-4 p-4 rounded-xl bg-accent-indigo/10 border border-accent-indigo/40 space-y-2 animate-fadeIn">
                <div className="flex items-center justify-between">
                  <span className="text-xs font-semibold uppercase text-accent-indigo">Classification Output</span>
                  <span className="text-xs font-mono font-bold text-accent-emerald">
                    {(scoreResult.confidence * 100).toFixed(1)}% Confidence
                  </span>
                </div>

                <div className="text-base font-bold text-[var(--text)]">
                  {scoreResult.persona_name} (Cluster #{scoreResult.cluster_label})
                </div>

                <p className="text-xs text-[var(--text-muted)]">
                  Recommended NBA: <strong className="text-[var(--text)]">{scoreResult.recommended_nba.title}</strong>
                </p>

                {scoreResult.top_traits && (
                  <div className="text-[11px] text-[var(--text-muted)] font-mono pt-1">
                    Top Drivers: {scoreResult.top_traits.join(', ')}
                  </div>
                )}
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  )
}
