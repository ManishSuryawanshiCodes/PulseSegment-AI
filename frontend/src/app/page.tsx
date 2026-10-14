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
  JobStatus,
} from '@/lib/api'
import { ThemeToggle } from '@/components/theme-toggle'
import { StepStepper, StepId } from '@/components/step-stepper'
import { Step1Data } from '@/components/step1-data'
import { Step2Prepare } from '@/components/step2-prepare'
import { Step3Benchmark } from '@/components/step3-benchmark'
import { Step4Explore } from '@/components/step4-explore'
import { Step5Act } from '@/components/step5-act'
import { HelpDrawer } from '@/components/help-drawer'
import { CoachMarks } from '@/components/coach-marks'
import {
  HelpCircle,
  Plus,
  Check,
  ChevronDown,
  Database,
  Loader2,
  AlertCircle,
  Sparkles,
} from 'lucide-react'

export default function Home() {
  return (
    <Suspense fallback={<div className="min-h-screen bg-[var(--bg)]" />}>
      <PulseSegmentApp />
    </Suspense>
  )
}

function PulseSegmentApp() {
  // Stepper state with URL search param sync
  const [currentStep, setCurrentStep] = useState<StepId>('data')

  // Datasets state
  const [datasets, setDatasets] = useState<DatasetInfo[]>([])
  const [activeDatasetId, setActiveDatasetId] = useState<string>('sample_credit_card')
  const [datasetDropdownOpen, setDatasetDropdownOpen] = useState(false)
  const [isUploading, setIsUploading] = useState(false)

  // Step data states
  const [profile, setProfile] = useState<DatasetProfile | null>(null)
  const [config, setConfig] = useState<DatasetConfig | null>(null)
  const [summary, setSummary] = useState<DatasetSummary | null>(null)
  const [features, setFeatures] = useState<FeatureEngineering | null>(null)
  const [matrix, setMatrix] = useState<ExperimentsMatrix | null>(null)
  const [personas, setPersonas] = useState<GrowthPersonas | null>(null)

  // Background Job & Loading states
  const [activeJobId, setActiveJobId] = useState<string | null>(null)
  const [jobStatus, setJobStatus] = useState<JobStatus | null>(null)
  const [isBenchmarkRunning, setIsBenchmarkRunning] = useState<boolean>(false)
  const [pageLoading, setPageLoading] = useState<boolean>(true)
  const [errorNotice, setErrorNotice] = useState<string | null>(null)

  // Help drawer state
  const [helpOpen, setHelpOpen] = useState(false)

  const hiddenUploadRef = useRef<HTMLInputElement>(null)
  const pollIntervalRef = useRef<NodeJS.Timeout | null>(null)

  // Synchronize step and dataset with URL search param
  useEffect(() => {
    if (typeof window !== 'undefined') {
      const params = new URLSearchParams(window.location.search)
      const stepParam = params.get('step') as StepId
      if (stepParam && ['data', 'prepare', 'benchmark', 'explore', 'act'].includes(stepParam)) {
        setCurrentStep(stepParam)
      }
      const dsParam = params.get('dataset')
      if (dsParam) {
        setActiveDatasetId(dsParam)
      }
    }
  }, [])

  const setStep = (step: StepId) => {
    setCurrentStep(step)
    if (typeof window !== 'undefined') {
      const url = new URL(window.location.href)
      url.searchParams.set('step', step)
      window.history.pushState({}, '', url.toString())
    }
  }

  const selectDataset = (dsId: string) => {
    setActiveDatasetId(dsId)
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
      const [profData, cfgData] = await Promise.all([
        api.getProfile(dsId).catch(() => null),
        api.getConfig(dsId).catch(() => null),
      ])
      if (profData) setProfile(profData)
      if (cfgData) setConfig(cfgData)

      // Try loading existing summary & feature engineering
      const sumData = await api.datasetSummary(dsId).catch(() => null)
      if (sumData) setSummary(sumData)

      const featData = await api.featureEngineering(dsId).catch(() => null)
      if (featData) setFeatures(featData)

      // Check if benchmark was already run
      const expData = await api.experimentsMatrix(dsId).catch(() => null)
      if (expData && expData.experiments && expData.experiments.length > 0) {
        setMatrix(expData)
        const persData = await api.growthPersonas(dsId).catch(() => null)
        if (persData) setPersonas(persData)
      } else {
        setMatrix(null)
        setPersonas(null)
      }
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

  // Start 16-experiment benchmark background job
  const handleStartBenchmark = async () => {
    if (isBenchmarkRunning) return
    setIsBenchmarkRunning(true)
    setStep('benchmark')

    try {
      const jobInfo = await api.runBenchmark(activeDatasetId)
      setActiveJobId(jobInfo.job_id)

      // Poll job status every 500ms
      if (pollIntervalRef.current) clearInterval(pollIntervalRef.current)
      pollIntervalRef.current = setInterval(async () => {
        try {
          const status = await api.getJobStatus(jobInfo.job_id)
          setJobStatus(status)

          if (status.status === 'completed') {
            if (pollIntervalRef.current) clearInterval(pollIntervalRef.current)
            setIsBenchmarkRunning(false)
            // Fetch completed results
            const [matrixRes, personasRes] = await Promise.all([
              api.experimentsMatrix(activeDatasetId),
              api.growthPersonas(activeDatasetId),
            ])
            setMatrix(matrixRes)
            setPersonas(personasRes)
          } else if (status.status === 'failed' || status.status === 'cancelled') {
            if (pollIntervalRef.current) clearInterval(pollIntervalRef.current)
            setIsBenchmarkRunning(false)
            if (status.status === 'failed') {
              setErrorNotice(status.error || 'Benchmark run failed.')
            }
          }
        } catch {
          // keep polling
        }
      }, 500)
    } catch (err: any) {
      setIsBenchmarkRunning(false)
      setErrorNotice(err.message || 'Failed to trigger benchmark')
    }
  }

  const handleCancelBenchmark = async () => {
    if (activeJobId) {
      await api.cancelJob(activeJobId).catch(() => {})
      if (pollIntervalRef.current) clearInterval(pollIntervalRef.current)
      setIsBenchmarkRunning(false)
    }
  }

  // Re-run benchmark with new k
  const handleRerunWithK = async (k: number) => {
    if (!config) return
    const updated = { ...config, n_clusters: k }
    setConfig(updated)
    await api.configureDataset(activeDatasetId, updated)
    await handleStartBenchmark()
  }

  const activeDataset = datasets.find((d) => d.id === activeDatasetId) || datasets[0]
  const numericFeaturesList = profile?.columns
    .filter((c) => c.suggested_role === 'numeric_feature' && (!config?.include_columns || config.include_columns.includes(c.name)))
    .map((c) => c.name) || []

  return (
    <div className="min-h-screen bg-[var(--bg)] text-[var(--text)] flex flex-col font-sans transition-colors selection:bg-accent-indigo selection:text-white">
      {/* Hidden file input for "+ Add dataset" */}
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

      {/* Top Bar (Frosted, Sticky) */}
      <header className="sticky top-0 z-40 glass-header bg-[#FAFAFA]/85 dark:bg-[#09090B]/85 border-b border-[var(--border)] transition-colors">
        <div className="max-w-[1200px] mx-auto px-6 h-14 flex items-center justify-between">
          {/* Left: Brand Logo + Dataset Selector */}
          <div className="flex items-center gap-4">
            <span className="text-sm font-bold tracking-tight flex items-center gap-1.5 cursor-pointer" onClick={() => setStep('data')}>
              <span className="text-accent-indigo text-base">◆</span>
              PulseSegment AI
            </span>

            {/* Dataset Selector Dropdown */}
            <div className="relative">
              <button
                type="button"
                onClick={() => setDatasetDropdownOpen(!datasetDropdownOpen)}
                className="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-[var(--card)] border border-[var(--border)] hover:border-accent-indigo/60 text-xs transition-colors shadow-sm"
              >
                <span className="w-2 h-2 rounded-full bg-accent-emerald pulse-dot shrink-0" />
                <span className="font-semibold text-[var(--text)] max-w-[160px] truncate">
                  {activeDataset?.name || 'Select Dataset'}
                </span>
                <span className="text-[var(--text-muted)] font-mono text-[11px]">
                  {activeDataset?.row_count ? `${activeDataset.row_count.toLocaleString()} × ${activeDataset.col_count}` : ''}
                </span>

                {activeDataset?.is_synthetic && (
                  <span className="px-1.5 py-0.2 rounded text-[10px] font-semibold bg-amber-500/15 text-amber-400 border border-amber-500/30">
                    SYNTHETIC
                  </span>
                )}

                <ChevronDown className="w-3.5 h-3.5 text-[var(--text-muted)] shrink-0 ml-0.5" />
              </button>

              {/* Dropdown Menu */}
              {datasetDropdownOpen && (
                <div className="absolute top-full left-0 mt-1.5 w-72 bg-[var(--card)] border border-[var(--border)] rounded-xl shadow-xl p-1.5 z-50 text-xs animate-fadeIn">
                  <div className="px-2 py-1 text-[10px] uppercase font-bold text-[var(--text-muted)]">
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
                        className={`w-full text-left px-2.5 py-2 rounded-lg flex items-center justify-between transition-colors ${
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
                      className="w-full text-left px-2.5 py-1.5 rounded-lg hover:bg-[var(--card-elevated)] text-accent-indigo font-medium flex items-center gap-1.5 transition-colors"
                    >
                      <Plus className="w-3.5 h-3.5" />
                      + Add new dataset (CSV or XLSX)
                    </button>
                  </div>
                </div>
              )}
            </div>
          </div>

          {/* Right: Help Button + Theme Toggle */}
          <div className="flex items-center gap-2.5">
            <button
              type="button"
              onClick={() => setHelpOpen(true)}
              className="p-2 rounded-lg hover:bg-[var(--card)] text-[var(--text-muted)] hover:text-[var(--text)] transition-colors border border-transparent hover:border-[var(--border)] text-xs flex items-center gap-1"
              title="Help & Guide"
            >
              <HelpCircle className="w-4 h-4" />
              <span className="hidden sm:inline font-medium">Help</span>
            </button>

            <ThemeToggle />
          </div>
        </div>
      </header>

      {/* Main Container - Centered max-w-[1200px] with 24/32px spacing */}
      <main className="flex-1 max-w-[1200px] w-full mx-auto px-6 py-8 space-y-8">
        {/* First-Run Coach Marks */}
        <CoachMarks />

        {/* 5-Step Progress Stepper */}
        <StepStepper
          currentStep={currentStep}
          onSelectStep={setStep}
          hasData={!!activeDataset}
          hasPrepared={!!summary && !!features}
          hasBenchmark={!!matrix && matrix.experiments.length > 0}
        />

        {/* Inline Global Error Notice if any */}
        {errorNotice && (
          <div className="p-3.5 rounded-xl bg-rose-500/10 border border-rose-500/30 text-rose-400 text-xs flex items-start justify-between gap-2.5 animate-fadeIn">
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

        {/* Active Step Content */}
        {currentStep === 'data' && (
          <Step1Data
            currentDataset={activeDataset || null}
            profile={profile}
            config={config}
            onDatasetChange={setActiveDatasetId}
            onConfigChange={setConfig}
            onAdvanceToPrepare={() => {
              loadDatasetData(activeDatasetId)
              setStep('prepare')
            }}
            isUploading={isUploading}
            onUploadFile={handleFileUpload}
          />
        )}

        {currentStep === 'prepare' && (
          <Step2Prepare
            summary={summary}
            features={features}
            config={config}
            onAdvanceToBenchmark={handleStartBenchmark}
          />
        )}

        {currentStep === 'benchmark' && (
          <Step3Benchmark
            matrix={matrix}
            jobStatus={jobStatus}
            isRunning={isBenchmarkRunning}
            onCancelJob={handleCancelBenchmark}
            onAdvanceToExplore={() => setStep('explore')}
          />
        )}

        {currentStep === 'explore' && (
          <Step4Explore
            matrix={matrix}
            personas={personas}
            config={config}
            datasetId={activeDatasetId}
            onRerunWithK={handleRerunWithK}
            onAdvanceToAct={() => setStep('act')}
          />
        )}

        {currentStep === 'act' && (
          <Step5Act
            personasData={personas}
            datasetId={activeDatasetId}
            numericFeatures={numericFeaturesList}
          />
        )}
      </main>

      {/* Help Slide-Over Drawer */}
      <HelpDrawer open={helpOpen} onClose={() => setHelpOpen(false)} />
    </div>
  )
}
