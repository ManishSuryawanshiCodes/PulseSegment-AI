'use client'

import React, { useState, useEffect, useRef, Suspense } from 'react'
import {
  api,
  DatasetInfo,
  DatasetProfile,
  DatasetConfig,
  DatasetSummary,
  FeatureEngineering,
  ExperimentsMatrix,
  GrowthPersonas,
  PersonaProfile,
  JobStatus,
} from '@/lib/api'
import { ThemeToggle } from '@/components/theme-toggle'
import { StepStepper, StepId } from '@/components/step-stepper'
import { Step1Data } from '@/components/step1-data'
import { Step2Groups } from '@/components/step2-groups'
import { Step3Act } from '@/components/step3-act'
import { MethodologyView } from '@/components/methodology-view'
import { HelpDrawer } from '@/components/help-drawer'
import { CoachMarks } from '@/components/coach-marks'
import {
  HelpCircle,
  Plus,
  Check,
  ChevronDown,
  Sparkles,
  AlertCircle,
  Sliders,
} from 'lucide-react'

export default function Home() {
  return (
    <Suspense fallback={<div className="min-h-screen bg-[var(--bg)]" />}>
      <PulseSegmentApp />
    </Suspense>
  )
}

function PulseSegmentApp() {
  // 3-Screen Flow + Methodology view
  const [currentStep, setCurrentStep] = useState<StepId>(() => {
    if (typeof window !== 'undefined') {
      const sp = new URLSearchParams(window.location.search).get('step') as StepId
      if (sp && ['data', 'groups', 'act', 'methodology'].includes(sp)) return sp
    }
    return 'data'
  })

  // Advanced details toggle (off by default, remembered in localStorage)
  const [showAdvanced, setShowAdvanced] = useState<boolean>(false)

  useEffect(() => {
    const saved = localStorage.getItem('pulsesegment_advanced_mode')
    if (saved === 'true') {
      setShowAdvanced(true)
    }
  }, [])

  const toggleAdvanced = () => {
    const nextVal = !showAdvanced
    setShowAdvanced(nextVal)
    localStorage.setItem('pulsesegment_advanced_mode', nextVal ? 'true' : 'false')
  }

  // Datasets state
  const [datasets, setDatasets] = useState<DatasetInfo[]>([])
  const [activeDatasetId, setActiveDatasetId] = useState<string>(() => {
    if (typeof window !== 'undefined') {
      const ds = new URLSearchParams(window.location.search).get('dataset')
      if (ds) return ds
    }
    return 'sample_credit_card'
  })
  const [datasetDropdownOpen, setDatasetDropdownOpen] = useState(false)
  const [isUploading, setIsUploading] = useState(false)

  // Step data states
  const [profile, setProfile] = useState<DatasetProfile | null>(null)
  const [config, setConfig] = useState<DatasetConfig | null>(null)
  const [summary, setSummary] = useState<DatasetSummary | null>(null)
  const [features, setFeatures] = useState<FeatureEngineering | null>(null)
  const [matrix, setMatrix] = useState<ExperimentsMatrix | null>(null)
  const [personas, setPersonas] = useState<GrowthPersonas | null>(null)

  // Custom group names persisted in session
  const [customNames, setCustomNames] = useState<Record<number, string>>({})
  const handleUpdateCustomName = (id: number, name: string) => {
    setCustomNames((prev) => ({ ...prev, [id]: name }))
  }

  // Active group selected for campaign
  const [selectedGroupIdx, setSelectedGroupIdx] = useState<number>(0)

  // Background Job & Loading states
  const [isFindingGroups, setIsFindingGroups] = useState<boolean>(false)
  const [progressMessage, setProgressMessage] = useState<string>(
    'Cleaning your data... Trying different methods... Choosing the best'
  )
  const [pageLoading, setPageLoading] = useState<boolean>(true)
  const [errorNotice, setErrorNotice] = useState<string | null>(null)
  const [toast, setToast] = useState<{ message: string; type?: 'info' | 'warning' | 'error' } | null>(null)

  // Help drawer state
  const [helpOpen, setHelpOpen] = useState(false)

  const hiddenUploadRef = useRef<HTMLInputElement>(null)
  const pollIntervalRef = useRef<NodeJS.Timeout | null>(null)

  // Auto-dismiss toast
  useEffect(() => {
    if (toast) {
      const timer = setTimeout(() => setToast(null), 4000)
      return () => clearTimeout(timer)
    }
  }, [toast])

  // Synchronize step and dataset with URL search param, enforcing locked step protection
  useEffect(() => {
    if (typeof window !== 'undefined') {
      const params = new URLSearchParams(window.location.search)
      const stepParam = params.get('step') as StepId
      const dsParam = params.get('dataset')
      if (dsParam && dsParam !== activeDatasetId) {
        setMatrix(null)
        setPersonas(null)
        setActiveDatasetId(dsParam)
        return
      }
      if (stepParam && ['data', 'groups', 'act', 'methodology'].includes(stepParam)) {
        if (!pageLoading) {
          const hasBenchmarkResults = Boolean(
            matrix &&
            matrix.dataset_id === activeDatasetId &&
            matrix.experiments &&
            matrix.experiments.length > 0
          )
          if ((stepParam === 'groups' || stepParam === 'act') && !hasBenchmarkResults) {
            setCurrentStep('data')
            setToast({
              message: 'Locked step: Find customer groups first before exploring segments or campaigns.',
              type: 'warning',
            })
            const url = new URL(window.location.href)
            url.searchParams.set('step', 'data')
            window.history.replaceState({}, '', url.toString())
            return
          }
        }
        setCurrentStep(stepParam)
      }
    }
  }, [matrix, pageLoading, activeDatasetId])

  const matrixRef = useRef<ExperimentsMatrix | null>(null)

  const setStep = (step: StepId) => {
    const activeMatrix = matrixRef.current || matrix
    const hasBenchmarkResults = Boolean(
      activeMatrix &&
      activeMatrix.dataset_id === activeDatasetId &&
      activeMatrix.experiments &&
      activeMatrix.experiments.length > 0
    )
    if ((step === 'groups' || step === 'act') && !hasBenchmarkResults) {
      setToast({
        message: 'Locked step: Click "Find my customer groups" first.',
        type: 'warning',
      })
      setCurrentStep('data')
      if (typeof window !== 'undefined') {
        const url = new URL(window.location.href)
        url.searchParams.set('step', 'data')
        window.history.pushState({}, '', url.toString())
      }
      return
    }

    setCurrentStep(step)
    if (typeof window !== 'undefined') {
      const url = new URL(window.location.href)
      url.searchParams.set('step', step)
      window.history.pushState({}, '', url.toString())
    }
  }

  const selectDataset = (dsId: string) => {
    setActiveDatasetId(dsId)
    setMatrix(null)
    setPersonas(null)
    setCustomNames({})
    if (typeof window !== 'undefined') {
      const url = new URL(window.location.href)
      url.searchParams.set('dataset', dsId)
      window.history.pushState({}, '', url.toString())
    }
  }

  // Load available datasets on mount
  useEffect(() => {
    loadDatasetsList()
  }, [])

  // Whenever activeDatasetId changes, load its profile and config
  useEffect(() => {
    if (activeDatasetId) {
      setMatrix(null)
      setPersonas(null)
      loadDatasetData(activeDatasetId)
    }
  }, [activeDatasetId])

  const loadDatasetsList = async () => {
    try {
      const list = await api.listDatasets()
      setDatasets(list)
      const urlDs = typeof window !== 'undefined' ? new URLSearchParams(window.location.search).get('dataset') : null
      if (urlDs && list.some((d) => d.id === urlDs)) {
        setActiveDatasetId(urlDs)
      } else if (list.length > 0 && !activeDatasetId) {
        setActiveDatasetId(list[0].id)
      }
    } catch (err: any) {
      console.warn('Dataset list fetch notice:', err.message)
    }
  }

  const loadDatasetData = async (dsId: string) => {
    setPageLoading(true)
    setErrorNotice(null)
    try {
      const dsMeta = datasets.find((d) => d.id === dsId)
      const shouldFetchExperiments = dsMeta ? Boolean(dsMeta.has_results) : true

      const [profData, cfgData, expData] = await Promise.all([
        api.getProfile(dsId).catch(() => null),
        api.getConfig(dsId).catch(() => null),
        shouldFetchExperiments ? api.experimentsMatrix(dsId).catch(() => null) : Promise.resolve(null),
      ])

      if (profData) setProfile(profData)
      if (cfgData) setConfig(cfgData)

      if (expData && expData.experiments && expData.experiments.length > 0) {
        matrixRef.current = expData
        setMatrix(expData)
        const persData = await api.growthPersonas(dsId).catch(() => null)
        if (persData) setPersonas(persData)
      } else {
        matrixRef.current = null
        setMatrix(null)
        setPersonas(null)
      }

      const [sumData, featData] = await Promise.all([
        api.datasetSummary(dsId).catch(() => null),
        api.featureEngineering(dsId).catch(() => null),
      ])
      if (sumData) setSummary(sumData)
      if (featData) setFeatures(featData)
    } catch (err: any) {
      setErrorNotice(err.message || 'Error loading dataset')
    } finally {
      setPageLoading(false)
    }
  }

  // File upload handler
  const handleFileUpload = async (file: File) => {
    setIsUploading(true)
    try {
      const newDataset = await api.uploadDataset(file)
      await loadDatasetsList()
      setActiveDatasetId(newDataset.id)
      setStep('data')
    } finally {
      setIsUploading(false)
    }
  }

  // Automated "Find my customer groups" orchestrator
  const handleFindCustomerGroups = async () => {
    if (isFindingGroups || !activeDatasetId) return
    setIsFindingGroups(true)
    setErrorNotice(null)
    setProgressMessage('Cleaning your data... Trying different methods... Choosing the best')

    try {
      // 1. Ensure latest configuration is saved
      if (config) {
        await api.configureDataset(activeDatasetId, config)
      }

      // 2. Trigger the multi-algorithm benchmark job
      const jobInfo = await api.runBenchmark(activeDatasetId)

      // 3. Poll job status until completed
      if (pollIntervalRef.current) clearInterval(pollIntervalRef.current)
      pollIntervalRef.current = setInterval(async () => {
        try {
          const status = await api.getJobStatus(jobInfo.job_id)

          if (status.status === 'completed') {
            if (pollIntervalRef.current) clearInterval(pollIntervalRef.current)
            setIsFindingGroups(false)

            // Fetch completed matrix & personas
            const [matrixRes, personasRes, sumRes, featRes] = await Promise.all([
              api.experimentsMatrix(activeDatasetId),
              api.growthPersonas(activeDatasetId),
              api.datasetSummary(activeDatasetId).catch(() => null),
              api.featureEngineering(activeDatasetId).catch(() => null),
            ])
            matrixRef.current = matrixRes
            setMatrix(matrixRes)
            setPersonas(personasRes)
            if (sumRes) setSummary(sumRes)
            if (featRes) setFeatures(featRes)

            // Transition smoothly to Screen 2: Your customer groups
            setCurrentStep('groups')
            if (typeof window !== 'undefined') {
              const url = new URL(window.location.href)
              url.searchParams.set('step', 'groups')
              window.history.pushState({}, '', url.toString())
            }
          } else if (status.status === 'failed' || status.status === 'cancelled') {
            if (pollIntervalRef.current) clearInterval(pollIntervalRef.current)
            setIsFindingGroups(false)
            if (status.status === 'failed') {
              setErrorNotice(status.error || 'Grouping failed. Please check customer data columns.')
            }
          }
        } catch {
          // keep polling
        }
      }, 500)
    } catch (err: any) {
      setIsFindingGroups(false)
      setErrorNotice(err.message || 'Failed to start customer grouping')
    }
  }

  // Plan campaign for a specific customer group
  const handlePlanCampaign = (persona: PersonaProfile) => {
    const idx = personas?.personas.findIndex((p) => (p.id ?? p.name) === (persona.id ?? persona.name)) ?? 0
    setSelectedGroupIdx(Math.max(0, idx))
    setStep('act')
  }

  const activeDataset = datasets.find((d) => d.id === activeDatasetId) || datasets[0]
  const numericFeaturesList =
    profile?.columns
      .filter(
        (c) =>
          c.suggested_role === 'numeric_feature' &&
          (!config?.include_columns || config.include_columns.includes(c.name))
      )
      .map((c) => c.name) || []

  return (
    <div className="min-h-screen bg-[var(--bg)] text-[var(--text)] flex flex-col font-sans transition-colors selection:bg-accent-indigo selection:text-white">
      {/* Hidden file input */}
      <input
        ref={hiddenUploadRef}
        type="file"
        accept=".csv,.xlsx,.xls"
        className="hidden"
        onChange={(e) => {
          if (e.target.files && e.target.files[0]) {
            handleFileUpload(e.target.files[0])
          }
        }}
      />

      {/* Top Header (Sticky, Frosted) */}
      <header className="sticky top-0 z-40 glass-header bg-[#FAFAFA]/85 dark:bg-[#09090B]/85 border-b border-[var(--border)] transition-colors">
        <div className="max-w-[1100px] mx-auto px-6 h-14 flex items-center justify-between">
          {/* Left: Brand Logo + Dataset Dropdown */}
          <div className="flex items-center gap-4">
            <span
              className="text-sm font-bold tracking-tight flex items-center gap-1.5 cursor-pointer"
              onClick={() => setStep('data')}
            >
              <span className="text-accent-indigo text-base">◆</span>
              PulseSegment AI
            </span>

            {/* Dataset Selector Dropdown */}
            <div className="relative">
              <button
                type="button"
                onClick={() => setDatasetDropdownOpen(!datasetDropdownOpen)}
                className="flex items-center gap-2 px-3 py-1.5 rounded-xl bg-[var(--card)] border border-[var(--border)] hover:border-accent-indigo/60 text-xs transition-colors shadow-sm"
              >
                <span className="w-2 h-2 rounded-full bg-accent-emerald pulse-dot shrink-0" />
                <span className="font-semibold text-[var(--text)] max-w-[160px] truncate">
                  {activeDataset?.name || 'Select Dataset'}
                </span>
                <span className="text-[var(--text-muted)] font-mono text-[11px]">
                  {activeDataset?.row_count ? `${activeDataset.row_count.toLocaleString()} customers` : ''}
                </span>

                {/* Persistent small "Demo data" badge when is_synthetic is true */}
                {activeDataset?.is_synthetic && (
                  <span className="px-2 py-0.5 rounded-full text-[10px] font-semibold bg-amber-500/15 text-amber-500 dark:text-amber-400 border border-amber-500/30 flex items-center gap-1 shadow-sm">
                    <span className="w-1.5 h-1.5 rounded-full bg-amber-400 animate-pulse" />
                    Demo data
                  </span>
                )}

                <ChevronDown className="w-3.5 h-3.5 text-[var(--text-muted)] shrink-0 ml-0.5" />
              </button>

              {/* Dropdown Menu */}
              {datasetDropdownOpen && (
                <div className="absolute top-full left-0 mt-1.5 w-72 bg-[var(--card)] border border-[var(--border)] rounded-2xl shadow-xl p-1.5 z-50 text-xs animate-fadeIn">
                  <div className="px-2.5 py-1 text-[10px] uppercase font-bold text-[var(--text-muted)]">
                    Available Datasets
                  </div>

                  <div className="max-h-56 overflow-y-auto space-y-1">
                    {datasets.map((d) => (
                      <button
                        key={d.id}
                        type="button"
                        onClick={() => {
                          selectDataset(d.id)
                          setDatasetDropdownOpen(false)
                        }}
                        className={`w-full text-left px-2.5 py-2 rounded-xl flex items-center justify-between transition-colors ${
                          d.id === activeDatasetId
                            ? 'bg-accent-indigo/10 text-accent-indigo font-semibold'
                            : 'hover:bg-[var(--card-elevated)] text-[var(--text)]'
                        }`}
                      >
                        <div className="min-w-0 pr-2">
                          <span className="block truncate font-medium">{d.name}</span>
                          <span className="block text-[10px] text-[var(--text-muted)] font-mono">
                            {d.row_count.toLocaleString()} rows • {d.col_count} columns
                          </span>
                        </div>

                        {d.id === activeDatasetId && (
                          <Check className="w-4 h-4 text-accent-indigo shrink-0" />
                        )}
                      </button>
                    ))}
                  </div>

                  <div className="pt-1.5 mt-1.5 border-t border-[var(--border)]">
                    <button
                      type="button"
                      onClick={() => {
                        setDatasetDropdownOpen(false)
                        hiddenUploadRef.current?.click()
                      }}
                      className="w-full text-left px-2.5 py-1.5 rounded-xl hover:bg-[var(--card-elevated)] text-accent-indigo font-medium flex items-center gap-1.5 transition-colors"
                    >
                      <Plus className="w-3.5 h-3.5" />
                      + Add new dataset (CSV or XLSX)
                    </button>
                  </div>
                </div>
              )}
            </div>
          </div>

          {/* Right: Advanced Details Toggle + Help Button + Theme Toggle */}
          <div className="flex items-center gap-2.5">
            {/* Advanced details toggle */}
            <button
              type="button"
              onClick={toggleAdvanced}
              className={`px-2.5 py-1.5 rounded-xl border text-xs font-medium flex items-center gap-1.5 transition-colors ${
                showAdvanced
                  ? 'bg-accent-indigo/15 border-accent-indigo text-accent-indigo'
                  : 'bg-[var(--card)] border-[var(--border)] text-[var(--text-muted)] hover:text-[var(--text)]'
              }`}
              title="Show advanced technical settings and metrics"
            >
              <Sliders className="w-3.5 h-3.5" />
              <span className="hidden sm:inline">Advanced details</span>
            </button>

            {/* Help Button */}
            <button
              type="button"
              onClick={() => setHelpOpen(true)}
              className="p-2 rounded-xl hover:bg-[var(--card)] text-[var(--text-muted)] hover:text-[var(--text)] transition-colors border border-transparent hover:border-[var(--border)] text-xs flex items-center gap-1"
              title="Help & Guide"
            >
              <HelpCircle className="w-4 h-4" />
              <span className="hidden sm:inline font-medium">Help</span>
            </button>

            <ThemeToggle />
          </div>
        </div>
      </header>

      {/* Main Content Area: max-w-[1100px] with 24/32px spacing */}
      <main className="flex-1 max-w-[1100px] w-full mx-auto px-6 py-8 space-y-8">
        {/* First-Run Coach Marks */}
        <CoachMarks />

        {/* 3-Step Progress Stepper + 'How we got these results' link */}
        <StepStepper
          currentStep={currentStep}
          onSelectStep={setStep}
          hasData={!!activeDataset}
          hasBenchmark={!!matrix && matrix.experiments && matrix.experiments.length > 0}
        />

        {/* Error Notice Banner */}
        {errorNotice && (
          <div className="p-4 rounded-xl bg-rose-500/10 border border-rose-500/30 text-rose-400 text-xs flex items-start justify-between gap-3 animate-fadeIn">
            <div className="flex items-start gap-2">
              <AlertCircle className="w-4 h-4 shrink-0 mt-0.5 text-rose-500" />
              <span>{errorNotice}</span>
            </div>
            <button
              onClick={() => setErrorNotice(null)}
              className="text-rose-400 hover:text-rose-200"
            >
              &times;
            </button>
          </div>
        )}

        {/* SCREEN 1: Add data */}
        {currentStep === 'data' && (
          <Step1Data
            currentDataset={activeDataset || null}
            profile={profile}
            config={config}
            onDatasetChange={selectDataset}
            onConfigChange={setConfig}
            onFindCustomerGroups={handleFindCustomerGroups}
            isFindingGroups={isFindingGroups}
            progressMessage={progressMessage}
            isUploading={isUploading}
            onUploadFile={handleFileUpload}
            showAdvanced={showAdvanced}
          />
        )}

        {/* SCREEN 2: Your customer groups */}
        {currentStep === 'groups' && (
          <Step2Groups
            personasData={personas}
            matrix={matrix}
            onPlanCampaign={handlePlanCampaign}
            onViewMethodology={() => setStep('methodology')}
            customNames={customNames}
            onUpdateCustomName={handleUpdateCustomName}
          />
        )}

        {/* SCREEN 3: Take action */}
        {currentStep === 'act' && (
          <Step3Act
            personasData={personas}
            datasetId={activeDatasetId}
            numericFeatures={numericFeaturesList}
            selectedGroupIndex={selectedGroupIdx}
            customNames={customNames}
            showAdvanced={showAdvanced}
          />
        )}

        {/* METHODOLOGY: How we got these results */}
        {currentStep === 'methodology' && (
          <MethodologyView
            matrix={matrix}
            summary={summary}
            features={features}
            personas={personas}
            onBack={() => setStep(matrix?.experiments?.length ? 'groups' : 'data')}
          />
        )}
      </main>

      {/* Help Slide-Over Drawer */}
      <HelpDrawer open={helpOpen} onClose={() => setHelpOpen(false)} />

      {/* Toast Notification */}
      {toast && (
        <div
          role="alert"
          className="fixed bottom-6 right-6 z-50 max-w-sm p-4 rounded-2xl bg-zinc-950/95 text-zinc-100 border border-amber-500/50 shadow-2xl backdrop-blur-md flex items-start gap-3 animate-fadeIn"
        >
          <AlertCircle className="w-5 h-5 text-amber-400 shrink-0 mt-0.5" />
          <div className="flex-1 text-xs">
            <div className="font-semibold text-amber-300">Step Locked</div>
            <div className="text-zinc-300 mt-0.5 leading-relaxed">{toast.message}</div>
          </div>
          <button
            type="button"
            onClick={() => setToast(null)}
            className="text-zinc-400 hover:text-white text-base leading-none shrink-0"
            aria-label="Dismiss toast"
          >
            &times;
          </button>
        </div>
      )}
    </div>
  )
}
