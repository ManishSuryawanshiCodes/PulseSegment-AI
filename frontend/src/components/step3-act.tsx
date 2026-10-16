'use client'

import React, { useState, useEffect } from 'react'
import {
  GrowthPersonas,
  PersonaProfile,
  PredictResponse,
  SegmentScore,
  CustomerLookupResult,
  api,
} from '@/lib/api'
import { formatNumber } from '@/lib/utils'
import {
  Sparkles,
  Copy,
  Download,
  Mail,
  MessageSquare,
  Bell,
  Search,
  Check,
  ChevronDown,
  ChevronUp,
  FileSpreadsheet,
  FileText,
  Sliders,
  DollarSign,
  TrendingUp,
  AlertTriangle,
  Info,
  Loader2,
  X,
  ArrowRight,
} from 'lucide-react'

interface Step3ActProps {
  personasData: GrowthPersonas | null
  datasetId: string
  numericFeatures: string[]
  selectedGroupIndex?: number
  customNames?: Record<number, string>
  showAdvanced?: boolean
}

const PALETTE = ['#6366F1', '#10B981', '#F59E0B', '#F43F5E', '#8B5CF6', '#06B6D4', '#EC4899', '#14B8A6']

export function Step3Act({
  personasData,
  datasetId,
  numericFeatures,
  selectedGroupIndex = 0,
  customNames = {},
  showAdvanced = false,
}: Step3ActProps) {
  const personas = personasData?.personas || []
  const [activeGroupIdx, setActiveGroupIdx] = useState<number>(selectedGroupIndex)
  const [activeChannelTab, setActiveChannelTab] = useState<'email' | 'sms' | 'push'>('email')
  const [copiedKey, setCopiedKey] = useState<string | null>(null)

  // Sync selectedGroupIndex if changed from parent
  useEffect(() => {
    if (selectedGroupIndex >= 0 && selectedGroupIndex < personas.length) {
      setActiveGroupIdx(selectedGroupIndex)
    }
  }, [selectedGroupIndex, personas.length])

  // Interactive ROI Simulator State
  const [avgRevenue, setAvgRevenue] = useState<number>(120.0) // Average yearly revenue per customer
  const [expectedUplift, setExpectedUplift] = useState<number>(15.0) // Expected improvement (%)
  const [campaignCost, setCampaignCost] = useState<number>(3.50) // Cost per customer ($)

  // Scores State
  const [scores, setScores] = useState<SegmentScore[]>([])
  useEffect(() => {
    if (datasetId) {
      api.getScores(datasetId).then(setScores).catch(() => {})
    }
  }, [datasetId])

  // Customer Lookup Modal State
  const [lookupModalOpen, setLookupModalOpen] = useState<boolean>(false)
  const [lookupIdInput, setLookupIdInput] = useState<string>('0')
  const [lookupLoading, setLookupLoading] = useState<boolean>(false)
  const [lookupResult, setLookupResult] = useState<CustomerLookupResult | null>(null)
  const [lookupError, setLookupError] = useState<string | null>(null)

  // Single Lead Scoring Modal State
  const [scoreModalOpen, setScoreModalOpen] = useState<boolean>(false)
  const [featureInputs, setFeatureInputs] = useState<Record<string, number>>({})
  const [scoringLoading, setScoringLoading] = useState<boolean>(false)
  const [scoreResult, setScoreResult] = useState<PredictResponse | null>(null)

  // Advanced Dropdown / Toggle
  const [advancedActionsOpen, setAdvancedActionsOpen] = useState<boolean>(false)

  const activePersona = personas[activeGroupIdx] || personas[0]
  const activeScore = scores.find((s) => s.cluster_label === (activePersona?.id ?? activeGroupIdx))

  const handleCopy = (text: string, key: string) => {
    navigator.clipboard.writeText(text)
    setCopiedKey(key)
    setTimeout(() => setCopiedKey(null), 2000)
  }

  // Dynamic ROI Calculations
  const customerCount = activePersona?.count || 1000
  const grossLiftDollars = customerCount * avgRevenue * (expectedUplift / 100)
  const totalSpendDollars = customerCount * campaignCost
  const extraRevenue = Math.max(0, grossLiftDollars - totalSpendDollars)
  const returnOnSpend = totalSpendDollars > 0 ? (grossLiftDollars / totalSpendDollars).toFixed(1) : '1.0'

  // Customer Lookup Handler
  const handleLookupCustomer = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!lookupIdInput.trim()) return
    setLookupLoading(true)
    setLookupError(null)
    setLookupResult(null)
    try {
      const res = await api.lookupCustomer(datasetId, lookupIdInput.trim())
      setLookupResult(res)
    } catch (err: any) {
      setLookupError(err.message || 'Customer record not found. Try index 0, 1, 2, etc.')
    } finally {
      setLookupLoading(false)
    }
  }

  // Lead Scoring Handler
  const openScoreModal = () => {
    const initial: Record<string, number> = {}
    numericFeatures.forEach((feat) => {
      const sampleVal = activePersona?.centroid_profile[feat] ?? 100
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

  if (!activePersona) {
    return (
      <div className="p-8 text-center bg-[var(--card)] border border-[var(--border)] rounded-2xl max-w-[1100px] mx-auto">
        <p className="text-sm text-[var(--text-muted)]">No customer groups loaded yet. Please run the grouping pipeline first.</p>
      </div>
    )
  }

  const activeName = customNames[activePersona.id ?? activeGroupIdx] || activePersona.name
  const activeOffer = activePersona.nba_campaigns?.[0] || {
    title: 'Customer Growth Campaign',
    description: 'Target this group with tailored rewards and incentives matching their spending patterns.',
    expected_roi_lift: '+15% Lift',
    channel: 'Email & Push',
  }

  // Dynamic multi-channel copy generated from group name & offer
  const emailSubject = `Special privilege unlocked for our ${activeName} members`
  const emailBody = `Dear Valued Customer,\n\nBecause you are one of our top ${activeName} members, we are excited to offer you an exclusive opportunity:\n\n${activeOffer.description}\n\nTake advantage of this privilege today by signing in to your account.\n\nWarm regards,\nThe Customer Success Team`

  const smsText = `[Exclusive] As a top ${activeName} customer, enjoy special rewards on your account this week: ${activeOffer.title}. Tap here to activate: pulsesegment.io/rewards`

  const pushText = `Special offer for ${activeName}: ${activeOffer.title} is now active! Tap to claim before it expires.`

  return (
    <div className="space-y-8 max-w-[1100px] mx-auto animate-fadeIn">
      {/* 1. Customer Group Switcher Tabs */}
      <div className="flex flex-wrap items-center gap-2 border-b border-[var(--border)] pb-3">
        {personas.map((p, idx) => {
          const isSelected = idx === activeGroupIdx
          const color = PALETTE[idx % PALETTE.length]
          const name = customNames[p.id ?? idx] || p.name
          return (
            <button
              key={p.id ?? idx}
              type="button"
              onClick={() => setActiveGroupIdx(idx)}
              className={`px-4 py-2 rounded-xl text-xs font-semibold transition-all flex items-center gap-2 border ${
                isSelected
                  ? 'bg-accent-indigo text-white border-accent-indigo shadow-sm'
                  : 'bg-[var(--card)] text-[var(--text-muted)] border-[var(--border)] hover:border-accent-indigo/50 hover:text-[var(--text)]'
              }`}
            >
              <span
                className="w-2.5 h-2.5 rounded-full shrink-0"
                style={{ backgroundColor: isSelected ? '#ffffff' : color }}
              />
              <span>{name}</span>
              <span className={`text-[11px] font-mono ${isSelected ? 'text-indigo-200' : 'text-[var(--text-muted)]'}`}>
                ({p.percentage.toFixed(0)}%)
              </span>
            </button>
          )
        })}
      </div>

      {/* 2. Main Active Group Playbook Card */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Left Column: Recommended Offer & Copy Generator (2 cols) */}
        <div className="lg:col-span-2 space-y-6">
          {/* Recommended Offer Card */}
          <div className="bg-[var(--card)] border border-[var(--border)] rounded-2xl p-6 shadow-sm space-y-4">
            <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 pb-3 border-b border-[var(--border)]">
              <div>
                <span className="text-xs font-semibold text-accent-indigo uppercase tracking-wider block">
                  Recommended Offer
                </span>
                <h3 className="text-xl font-bold text-[var(--text)] mt-0.5">
                  {activeOffer.title}
                </h3>
              </div>

              {/* Dynamic Value & Risk badges */}
              <div className="flex items-center gap-2">
                <span
                  className="px-2.5 py-1 rounded-full text-xs font-semibold bg-emerald-500/10 text-emerald-500 border border-emerald-500/30"
                  title={activeScore?.formula_explanation || activePersona.value_metric_name || 'Share of activity'}
                >
                  Value: {(activeScore?.value_score ?? activePersona.value_score ?? activePersona.percentage).toFixed(1)}% share
                </span>
                <span
                  className={`px-2.5 py-1 rounded-full text-xs font-semibold border ${
                    (activeScore?.risk_score ?? activePersona.risk_score ?? 50) > 60
                      ? 'bg-rose-500/10 text-rose-500 border-rose-500/30'
                      : 'bg-amber-500/10 text-amber-500 border-amber-500/30'
                  }`}
                  title={activePersona.risk_formula_explanation || 'Weighted Risk Score'}
                >
                  Risk: {(activeScore?.risk_score ?? activePersona.risk_score ?? 50).toFixed(0)}/100
                </span>
              </div>
            </div>

            <p className="text-sm text-[var(--text)] leading-relaxed">
              {activeOffer.description}
            </p>

            <div className="p-3.5 rounded-xl bg-[var(--card-elevated)] border border-[var(--border)] flex items-start gap-2.5 text-xs text-[var(--text-muted)]">
              <Sparkles className="w-4 h-4 text-accent-indigo shrink-0 mt-0.5" />
              <div>
                <span className="font-semibold text-[var(--text)] block">Expected campaign result</span>
                <span className="mt-0.5 block leading-relaxed">
                  Projected {activeOffer.expected_roi_lift} via {activeOffer.channel} channels tailored for {activeName}.
                </span>
              </div>
            </div>
          </div>

          {/* Ready-to-Copy Multi-Channel Text */}
          <div className="bg-[var(--card)] border border-[var(--border)] rounded-2xl p-6 shadow-sm space-y-4">
            <div className="flex items-center justify-between">
              <div>
                <h4 className="text-sm font-bold text-[var(--text)]">Ready-to-copy campaign text</h4>
                <p className="text-xs text-[var(--text-muted)]">Formatted and customized for this customer group.</p>
              </div>

              {/* Channel Tabs */}
              <div className="flex items-center gap-1 p-1 bg-[var(--card-elevated)] border border-[var(--border)] rounded-xl text-xs">
                <button
                  type="button"
                  onClick={() => setActiveChannelTab('email')}
                  className={`px-3 py-1.5 rounded-lg font-medium transition-colors flex items-center gap-1.5 ${
                    activeChannelTab === 'email'
                      ? 'bg-accent-indigo text-white shadow-sm'
                      : 'text-[var(--text-muted)] hover:text-[var(--text)]'
                  }`}
                >
                  <Mail className="w-3.5 h-3.5" />
                  <span>Email</span>
                </button>

                <button
                  type="button"
                  onClick={() => setActiveChannelTab('sms')}
                  className={`px-3 py-1.5 rounded-lg font-medium transition-colors flex items-center gap-1.5 ${
                    activeChannelTab === 'sms'
                      ? 'bg-accent-indigo text-white shadow-sm'
                      : 'text-[var(--text-muted)] hover:text-[var(--text)]'
                  }`}
                >
                  <MessageSquare className="w-3.5 h-3.5" />
                  <span>SMS</span>
                </button>

                <button
                  type="button"
                  onClick={() => setActiveChannelTab('push')}
                  className={`px-3 py-1.5 rounded-lg font-medium transition-colors flex items-center gap-1.5 ${
                    activeChannelTab === 'push'
                      ? 'bg-accent-indigo text-white shadow-sm'
                      : 'text-[var(--text-muted)] hover:text-[var(--text)]'
                  }`}
                >
                  <Bell className="w-3.5 h-3.5" />
                  <span>Push</span>
                </button>
              </div>
            </div>

            {/* Email Tab */}
            {activeChannelTab === 'email' && (
              <div className="p-4 rounded-xl bg-[var(--card-elevated)] border border-[var(--border)] space-y-3 relative">
                <div>
                  <span className="text-[11px] font-semibold text-[var(--text-muted)] uppercase tracking-wider block">Subject line</span>
                  <p className="text-sm font-semibold text-[var(--text)] mt-0.5">
                    {emailSubject}
                  </p>
                </div>

                <div className="pt-2 border-t border-[var(--border)]">
                  <span className="text-[11px] font-semibold text-[var(--text-muted)] uppercase tracking-wider block">Body copy</span>
                  <p className="text-xs text-[var(--text)] mt-1 whitespace-pre-line leading-relaxed font-sans">
                    {emailBody}
                  </p>
                </div>

                <div className="flex justify-end pt-2">
                  <button
                    type="button"
                    onClick={() =>
                      handleCopy(
                        `Subject: ${emailSubject}\n\n${emailBody}`,
                        'email'
                      )
                    }
                    className="px-3 py-1.5 rounded-lg bg-[var(--card)] border border-[var(--border)] hover:border-accent-indigo/60 text-xs font-medium text-[var(--text)] transition-colors flex items-center gap-1.5 cursor-pointer shadow-sm"
                  >
                    {copiedKey === 'email' ? (
                      <>
                        <Check className="w-3.5 h-3.5 text-accent-emerald" />
                        <span className="text-accent-emerald font-semibold">Copied!</span>
                      </>
                    ) : (
                      <>
                        <Copy className="w-3.5 h-3.5 text-[var(--text-muted)]" />
                        <span>Copy email text</span>
                      </>
                    )}
                  </button>
                </div>
              </div>
            )}

            {/* SMS Tab */}
            {activeChannelTab === 'sms' && (
              <div className="p-4 rounded-xl bg-[var(--card-elevated)] border border-[var(--border)] space-y-3 relative">
                <span className="text-[11px] font-semibold text-[var(--text-muted)] uppercase tracking-wider block">SMS Message</span>
                <p className="text-xs text-[var(--text)] leading-relaxed font-sans">
                  {smsText}
                </p>

                <div className="flex items-center justify-between pt-2 border-t border-[var(--border)] text-[11px] text-[var(--text-muted)]">
                  <span className="font-mono">{smsText.length} characters</span>
                  <button
                    type="button"
                    onClick={() => handleCopy(smsText, 'sms')}
                    className="px-3 py-1.5 rounded-lg bg-[var(--card)] border border-[var(--border)] hover:border-accent-indigo/60 text-xs font-medium text-[var(--text)] transition-colors flex items-center gap-1.5 cursor-pointer shadow-sm"
                  >
                    {copiedKey === 'sms' ? (
                      <>
                        <Check className="w-3.5 h-3.5 text-accent-emerald" />
                        <span className="text-accent-emerald font-semibold">Copied!</span>
                      </>
                    ) : (
                      <>
                        <Copy className="w-3.5 h-3.5 text-[var(--text-muted)]" />
                        <span>Copy SMS</span>
                      </>
                    )}
                  </button>
                </div>
              </div>
            )}

            {/* Push Tab */}
            {activeChannelTab === 'push' && (
              <div className="p-4 rounded-xl bg-[var(--card-elevated)] border border-[var(--border)] space-y-3 relative">
                <span className="text-[11px] font-semibold text-[var(--text-muted)] uppercase tracking-wider block">Push Notification</span>
                <p className="text-xs text-[var(--text)] leading-relaxed font-sans">
                  {pushText}
                </p>

                <div className="flex items-center justify-between pt-2 border-t border-[var(--border)] text-[11px] text-[var(--text-muted)]">
                  <span className="font-mono">{pushText.length} characters</span>
                  <button
                    type="button"
                    onClick={() => handleCopy(pushText, 'push')}
                    className="px-3 py-1.5 rounded-lg bg-[var(--card)] border border-[var(--border)] hover:border-accent-indigo/60 text-xs font-medium text-[var(--text)] transition-colors flex items-center gap-1.5 cursor-pointer shadow-sm"
                  >
                    {copiedKey === 'push' ? (
                      <>
                        <Check className="w-3.5 h-3.5 text-accent-emerald" />
                        <span className="text-accent-emerald font-semibold">Copied!</span>
                      </>
                    ) : (
                      <>
                        <Copy className="w-3.5 h-3.5 text-[var(--text-muted)]" />
                        <span>Copy Push</span>
                      </>
                    )}
                  </button>
                </div>
              </div>
            )}
          </div>
        </div>

        {/* Right Column: ROI Box & Action Buttons (1 col) */}
        <div className="space-y-6">
          {/* ROI Simulator Box */}
          <div className="bg-[var(--card)] border border-[var(--border)] rounded-2xl p-6 shadow-sm space-y-4">
            <div>
              <span className="text-xs font-semibold text-accent-indigo uppercase tracking-wider block">
                Revenue Simulator
              </span>
              <h3 className="text-base font-bold text-[var(--text)] mt-0.5">
                Expected impact for {activeName}
              </h3>
              <p className="text-xs text-[var(--text-muted)] mt-0.5">
                Based on {customerCount.toLocaleString()} customers in this group.
              </p>
            </div>

            {/* 3 User Inputs */}
            <div className="space-y-3.5 pt-1">
              <div>
                <label className="text-xs font-medium text-[var(--text-muted)] block">
                  Average yearly revenue per customer
                </label>
                <div className="relative mt-1">
                  <span className="absolute left-3 top-2 text-xs text-[var(--text-muted)] font-mono">$</span>
                  <input
                    type="number"
                    min="1"
                    step="5"
                    value={avgRevenue}
                    onChange={(e) => setAvgRevenue(Math.max(1, parseFloat(e.target.value) || 0))}
                    className="w-full bg-[var(--card-elevated)] border border-[var(--border)] rounded-xl pl-7 pr-3 py-1.5 text-xs font-mono metric-value text-[var(--text)] focus:outline-none focus:border-accent-indigo"
                  />
                </div>
              </div>

              <div>
                <label className="text-xs font-medium text-[var(--text-muted)] block">
                  Expected improvement (%)
                </label>
                <div className="relative mt-1">
                  <input
                    type="number"
                    min="0.5"
                    max="100"
                    step="0.5"
                    value={expectedUplift}
                    onChange={(e) => setExpectedUplift(Math.max(0.1, parseFloat(e.target.value) || 0))}
                    className="w-full bg-[var(--card-elevated)] border border-[var(--border)] rounded-xl px-3 py-1.5 text-xs font-mono metric-value text-[var(--text)] focus:outline-none focus:border-accent-indigo"
                  />
                  <span className="absolute right-3 top-2 text-xs text-[var(--text-muted)] font-mono">%</span>
                </div>
              </div>

              <div>
                <label className="text-xs font-medium text-[var(--text-muted)] block">
                  Cost per customer
                </label>
                <div className="relative mt-1">
                  <span className="absolute left-3 top-2 text-xs text-[var(--text-muted)] font-mono">$</span>
                  <input
                    type="number"
                    min="0.1"
                    step="0.5"
                    value={campaignCost}
                    onChange={(e) => setCampaignCost(Math.max(0.01, parseFloat(e.target.value) || 0))}
                    className="w-full bg-[var(--card-elevated)] border border-[var(--border)] rounded-xl pl-7 pr-3 py-1.5 text-xs font-mono metric-value text-[var(--text)] focus:outline-none focus:border-accent-indigo"
                  />
                </div>
              </div>
            </div>

            {/* Results: Extra Revenue & Return on Spend */}
            <div className="pt-3 border-t border-[var(--border)] grid grid-cols-2 gap-3">
              <div className="p-3.5 rounded-xl bg-[var(--card-elevated)] border border-[var(--border)]">
                <span className="text-[11px] text-[var(--text-muted)] block font-medium">Extra revenue</span>
                <span className="text-xl font-bold metric-value text-accent-emerald block mt-0.5">
                  ${Math.round(extraRevenue).toLocaleString()}
                </span>
              </div>

              <div className="p-3.5 rounded-xl bg-[var(--card-elevated)] border border-[var(--border)]">
                <span className="text-[11px] text-[var(--text-muted)] block font-medium">Return on spend</span>
                <span className="text-xl font-bold metric-value text-accent-indigo block mt-0.5">
                  {returnOnSpend}x
                </span>
              </div>
            </div>
          </div>

          {/* Action Buttons Card */}
          <div className="bg-[var(--card)] border border-[var(--border)] rounded-2xl p-6 shadow-sm space-y-3">
            <h4 className="text-xs font-semibold text-[var(--text-muted)] uppercase tracking-wider block">
              Customer Actions
            </h4>

            {/* Download customer list with groups (CSV) */}
            <a
              href={api.exportSegmentsUrl(datasetId)}
              download={`customer_groups_${datasetId}.csv`}
              className="w-full py-2.5 px-3 rounded-xl bg-accent-indigo hover:bg-accent-indigo/90 text-white text-xs font-semibold shadow-sm transition-all flex items-center justify-center gap-2 cursor-pointer"
            >
              <Download className="w-4 h-4" />
              <span>Download customer list with groups (CSV)</span>
            </a>

            {/* Check one customer */}
            <button
              type="button"
              onClick={() => {
                setLookupModalOpen(true)
                setLookupResult(null)
                setLookupError(null)
              }}
              className="w-full py-2.5 px-3 rounded-xl bg-[var(--card-elevated)] hover:bg-[var(--border)] border border-[var(--border)] text-xs font-medium text-[var(--text)] transition-colors flex items-center justify-center gap-2 cursor-pointer"
            >
              <Search className="w-4 h-4 text-accent-indigo" />
              <span>Check one customer</span>
            </button>

            {/* Advanced Actions Collapsible / Dropdown (Hides Model Download, Drift check, Executive report) */}
            <div className="pt-2 border-t border-[var(--border)]">
              <button
                type="button"
                onClick={() => setAdvancedActionsOpen(!advancedActionsOpen)}
                className="w-full flex items-center justify-between text-xs font-medium text-[var(--text-muted)] hover:text-[var(--text)] transition-colors py-1"
              >
                <span>Advanced downloads & checks</span>
                {advancedActionsOpen ? (
                  <ChevronUp className="w-3.5 h-3.5" />
                ) : (
                  <ChevronDown className="w-3.5 h-3.5" />
                )}
              </button>

              {advancedActionsOpen && (
                <div className="mt-3 space-y-2 animate-fadeIn text-xs">
                  {/* Download Model */}
                  <a
                    href={api.exportModelUrl(datasetId)}
                    download={`pulsegroups_model_${datasetId}.joblib`}
                    className="w-full py-2 px-3 rounded-lg bg-[var(--card-elevated)] hover:bg-[var(--border)] border border-[var(--border)] text-[var(--text)] transition-colors flex items-center justify-between block"
                  >
                    <span>Download model (.joblib)</span>
                    <Download className="w-3.5 h-3.5 text-[var(--text-muted)]" />
                  </a>

                  {/* Download Model JSON */}
                  <a
                    href={api.exportModelJsonUrl(datasetId)}
                    download={`pulsegroups_model_${datasetId}.json`}
                    className="w-full py-2 px-3 rounded-lg bg-[var(--card-elevated)] hover:bg-[var(--border)] border border-[var(--border)] text-[var(--text)] transition-colors flex items-center justify-between block"
                  >
                    <span>Download model specs (.json)</span>
                    <FileText className="w-3.5 h-3.5 text-[var(--text-muted)]" />
                  </a>

                  {/* Executive Report HTML */}
                  <a
                    href={api.reportHtmlUrl(datasetId)}
                    target="_blank"
                    rel="noreferrer"
                    className="w-full py-2 px-3 rounded-lg bg-[var(--card-elevated)] hover:bg-[var(--border)] border border-[var(--border)] text-[var(--text)] transition-colors flex items-center justify-between block"
                  >
                    <span>One-page summary report (HTML)</span>
                    <FileSpreadsheet className="w-3.5 h-3.5 text-[var(--text-muted)]" />
                  </a>

                  {/* Score New Lead / Customer Button */}
                  <button
                    type="button"
                    onClick={openScoreModal}
                    className="w-full py-2 px-3 rounded-lg bg-[var(--card-elevated)] hover:bg-[var(--border)] border border-[var(--border)] text-[var(--text)] transition-colors flex items-center justify-between"
                  >
                    <span>Classify new customer attributes</span>
                    <Sliders className="w-3.5 h-3.5 text-[var(--text-muted)]" />
                  </button>
                </div>
              )}
            </div>
          </div>
        </div>
      </div>

      {/* Customer Lookup Modal */}
      {lookupModalOpen && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/60 backdrop-blur-sm animate-fadeIn">
          <div className="w-full max-w-md bg-[var(--card)] border border-[var(--border)] rounded-2xl p-6 shadow-2xl space-y-4">
            <div className="flex items-center justify-between pb-3 border-b border-[var(--border)]">
              <div className="flex items-center gap-2">
                <Search className="w-4 h-4 text-accent-indigo" />
                <h3 className="font-bold text-sm text-[var(--text)]">Check One Customer</h3>
              </div>
              <button
                type="button"
                onClick={() => setLookupModalOpen(false)}
                className="p-1 rounded-lg hover:bg-[var(--card-elevated)] text-[var(--text-muted)] hover:text-[var(--text)]"
              >
                <X className="w-4 h-4" />
              </button>
            </div>

            <form onSubmit={handleLookupCustomer} className="space-y-3">
              <div>
                <label className="text-xs font-medium text-[var(--text-muted)] block">
                  Customer ID or Row Index
                </label>
                <div className="flex items-center gap-2 mt-1">
                  <input
                    type="text"
                    value={lookupIdInput}
                    onChange={(e) => setLookupIdInput(e.target.value)}
                    placeholder="e.g. 0, 10, CUST_1001"
                    className="flex-1 bg-[var(--card-elevated)] border border-[var(--border)] rounded-xl px-3 py-2 text-xs font-mono text-[var(--text)] focus:outline-none focus:border-accent-indigo"
                  />
                  <button
                    type="submit"
                    disabled={lookupLoading}
                    className="px-4 py-2 bg-accent-indigo hover:bg-accent-indigo/90 text-white rounded-xl text-xs font-semibold transition-colors disabled:opacity-50"
                  >
                    {lookupLoading ? <Loader2 className="w-4 h-4 animate-spin" /> : 'Search'}
                  </button>
                </div>
              </div>
            </form>

            {lookupError && (
              <div className="p-3 rounded-xl bg-rose-500/10 border border-rose-500/30 text-rose-400 text-xs flex items-center gap-2">
                <AlertTriangle className="w-4 h-4 shrink-0" />
                <span>{lookupError}</span>
              </div>
            )}

            {lookupResult && (
              <div className="p-4 rounded-xl bg-[var(--card-elevated)] border border-[var(--border)] space-y-3 animate-fadeIn text-xs">
                <div className="flex items-center justify-between pb-2 border-b border-[var(--border)]">
                  <div>
                    <span className="text-[11px] text-[var(--text-muted)] block">Assigned Group</span>
                    <span className="text-sm font-bold text-accent-indigo block mt-0.5">
                      {lookupResult.persona_name}
                    </span>
                  </div>
                  <div className="text-right">
                    <span className="text-[11px] text-[var(--text-muted)] block">Confidence</span>
                    <span className="text-sm font-bold metric-value text-accent-emerald block mt-0.5">
                      {lookupResult.confidence.toFixed(1)}%
                    </span>
                  </div>
                </div>

                {lookupResult.distinguishing_traits && lookupResult.distinguishing_traits.length > 0 && (
                  <div>
                    <span className="font-semibold text-[var(--text)] block mb-1">What makes this customer different:</span>
                    <div className="space-y-1">
                      {lookupResult.distinguishing_traits.map((t: string, i: number) => (
                        <div key={i} className="flex items-center gap-1.5 text-zinc-300">
                          <span className="text-accent-indigo">•</span>
                          <span>{t}</span>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            )}
          </div>
        </div>
      )}

      {/* Classify New Customer Lead Modal */}
      {scoreModalOpen && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/60 backdrop-blur-sm animate-fadeIn">
          <div className="w-full max-w-lg bg-[var(--card)] border border-[var(--border)] rounded-2xl p-6 shadow-2xl space-y-4 max-h-[90vh] overflow-y-auto">
            <div className="flex items-center justify-between pb-3 border-b border-[var(--border)]">
              <div className="flex items-center gap-2">
                <Sliders className="w-4 h-4 text-accent-indigo" />
                <h3 className="font-bold text-sm text-[var(--text)]">Classify Customer Attributes</h3>
              </div>
              <button
                type="button"
                onClick={() => setScoreModalOpen(false)}
                className="p-1 rounded-lg hover:bg-[var(--card-elevated)] text-[var(--text-muted)] hover:text-[var(--text)]"
              >
                <X className="w-4 h-4" />
              </button>
            </div>

            <form onSubmit={handlePredict} className="space-y-4">
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 max-h-60 overflow-y-auto pr-1 text-xs">
                {numericFeatures.map((feat) => (
                  <div key={feat}>
                    <label className="text-[11px] font-mono text-[var(--text-muted)] block truncate">
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
                      className="w-full mt-1 bg-[var(--card-elevated)] border border-[var(--border)] rounded-lg px-2.5 py-1.5 font-mono text-xs text-[var(--text)] focus:outline-none focus:border-accent-indigo"
                    />
                  </div>
                ))}
              </div>

              <button
                type="submit"
                disabled={scoringLoading}
                className="w-full py-2.5 bg-accent-indigo hover:bg-accent-indigo/90 text-white rounded-xl text-xs font-semibold transition-colors flex items-center justify-center gap-2 disabled:opacity-50"
              >
                {scoringLoading ? (
                  <>
                    <Loader2 className="w-4 h-4 animate-spin" />
                    <span>Matching customer group...</span>
                  </>
                ) : (
                  <span>Find Customer Group</span>
                )}
              </button>
            </form>

            {scoreResult && (
              <div className="p-4 rounded-xl bg-accent-indigo/10 border border-accent-indigo/30 space-y-2 animate-fadeIn text-xs">
                <div className="flex items-center justify-between">
                  <span className="font-semibold text-accent-indigo">Assigned Customer Group:</span>
                  <span className="font-bold text-sm text-[var(--text)]">{scoreResult.persona_name}</span>
                </div>
                <div className="flex items-center justify-between text-[11px] text-[var(--text-muted)]">
                  <span>Confidence:</span>
                  <span className="font-mono font-medium text-accent-emerald">
                    {scoreResult.confidence.toFixed(1)}%
                  </span>
                </div>
                <p className="text-[11px] text-[var(--text-muted)] pt-1 border-t border-accent-indigo/20">
                  Recommended Offer: <strong className="text-[var(--text)]">{scoreResult.recommended_nba?.title}</strong>
                </p>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  )
}
