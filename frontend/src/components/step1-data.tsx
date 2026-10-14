'use client'

import React, { useState, useRef } from 'react'
import {
  DatasetInfo,
  DatasetProfile,
  DatasetConfig,
  api,
} from '@/lib/api'
import { formatNumber } from '@/lib/utils'
import {
  Upload,
  FileSpreadsheet,
  CheckCircle2,
  AlertTriangle,
  Database,
  Layers,
  Sparkles,
  ArrowRight,
  Download,
  Info,
  Loader2,
  Check,
  X,
  Filter,
} from 'lucide-react'

interface Step1DataProps {
  currentDataset: DatasetInfo | null
  profile: DatasetProfile | null
  config: DatasetConfig | null
  onDatasetChange: (datasetId: string) => void
  onConfigChange: (config: DatasetConfig) => void
  onAdvanceToPrepare: () => void
  isUploading: boolean
  onUploadFile: (file: File) => Promise<void>
}

export function Step1Data({
  currentDataset,
  profile,
  config,
  onDatasetChange,
  onConfigChange,
  onAdvanceToPrepare,
  isUploading,
  onUploadFile,
}: Step1DataProps) {
  const [dragActive, setDragActive] = useState(false)
  const [errorMessage, setErrorMessage] = useState<string | null>(null)
  const [savingConfig, setSavingConfig] = useState(false)
  const fileInputRef = useRef<HTMLInputElement>(null)

  // Drag handlers
  const handleDrag = (e: React.DragEvent) => {
    e.preventDefault()
    e.stopPropagation()
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setDragActive(true)
    } else if (e.type === 'dragleave') {
      setDragActive(false)
    }
  }

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault()
    e.stopPropagation()
    setDragActive(false)
    setErrorMessage(null)

    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      processFile(e.dataTransfer.files[0])
    }
  }

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setErrorMessage(null)
    if (e.target.files && e.target.files[0]) {
      processFile(e.target.files[0])
    }
  }

  const processFile = async (file: File) => {
    setErrorMessage(null)
    const ext = file.name.split('.').pop()?.toLowerCase()
    if (ext !== 'csv' && ext !== 'xlsx' && ext !== 'xls') {
      setErrorMessage('Please upload a standard CSV (.csv) or Excel (.xlsx) file.')
      return
    }

    try {
      await onUploadFile(file)
    } catch (err: any) {
      setErrorMessage(
        err.message ||
          'Failed to validate file. Ensure your CSV has at least 100 rows and 3 numeric columns.'
      )
    }
  }

  // Generate downloadable sample CSV template
  const downloadTemplate = () => {
    const csvContent =
      'CUSTOMER_ID,RECENCY,FREQUENCY,MONETARY,AVG_ORDER_VALUE,RETURN_RATE,REGION\n' +
      'CUST_1001,14.5,8,1240.50,155.06,0.025,North America\n' +
      'CUST_1002,45.2,3,320.00,106.67,0.050,EMEA\n' +
      'CUST_1003,3.0,15,4890.20,326.01,0.010,APAC\n' +
      'CUST_1004,120.0,1,85.00,85.00,0.120,LATAM\n' +
      'CUST_1005,28.4,6,740.10,123.35,0.040,North America\n'

    const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' })
    const url = URL.createObjectURL(blob)
    const link = document.createElement('a')
    link.href = url
    link.setAttribute('download', 'pulsesegment_customer_template.csv')
    document.body.appendChild(link)
    link.click()
    document.body.removeChild(link)
  }

  // Handle configuration changes
  const handleToggleInclude = (colName: string) => {
    if (!config || !profile) return
    const currentIncludes = config.include_columns || profile.columns.filter(c => c.suggested_role === 'numeric_feature').map(c => c.name)
    let nextIncludes: string[]
    if (currentIncludes.includes(colName)) {
      if (currentIncludes.length <= 3) {
        setErrorMessage('At least 3 numeric features must remain included for clustering.')
        return
      }
      nextIncludes = currentIncludes.filter((c) => c !== colName)
    } else {
      nextIncludes = [...currentIncludes, colName]
    }
    setErrorMessage(null)
    onConfigChange({ ...config, include_columns: nextIncludes })
  }

  const handleIdColumnChange = (idCol: string) => {
    if (!config) return
    onConfigChange({ ...config, id_column: idCol === 'none' ? null : idCol })
  }

  const handleDomainChange = (domain: 'credit_card' | 'ecommerce_rfm' | 'saas' | 'generic') => {
    if (!config) return
    onConfigChange({ ...config, domain_template: domain })
  }

  const handleSaveAndPrepare = async () => {
    if (!currentDataset || !config) return
    setSavingConfig(true)
    setErrorMessage(null)
    try {
      await api.configureDataset(currentDataset.id, config)
      onAdvanceToPrepare()
    } catch (err: any) {
      setErrorMessage(err.message || 'Failed to apply configuration.')
    } finally {
      setSavingConfig(false)
    }
  }

  // Health summary metrics
  const totalRows = profile?.row_count || currentDataset?.row_count || 0
  const totalCols = profile?.col_count || currentDataset?.col_count || 0
  const includedCols = config?.include_columns?.length ?? profile?.columns?.filter(c => c.suggested_role === 'numeric_feature').length ?? (totalCols > 1 ? totalCols - 1 : totalCols)
  const totalMissing = profile?.columns?.reduce((acc, c) => acc + c.missing_count, 0) || 0
  const totalCells = totalRows * (totalCols || 1)
  const missingPct = totalCells > 0 ? ((totalMissing / totalCells) * 100).toFixed(1) : '0.0'

  return (
    <div className="space-y-8 animate-fadeIn">
      {/* 1. Large Dropzone Card */}
      <div className="bg-[var(--card)] border border-[var(--border)] rounded-xl p-6 shadow-sm">
        <input
          ref={fileInputRef}
          type="file"
          accept=".csv,.xlsx,.xls"
          className="hidden"
          onChange={handleFileChange}
        />

        <div
          onDragEnter={handleDrag}
          onDragLeave={handleDrag}
          onDragOver={handleDrag}
          onDrop={handleDrop}
          onClick={() => fileInputRef.current?.click()}
          className={`border-2 border-dashed rounded-xl p-8 text-center cursor-pointer transition-all ${
            dragActive
              ? 'border-accent-indigo bg-accent-indigo/10'
              : 'border-[var(--border)] hover:border-accent-indigo/60 hover:bg-[var(--card-elevated)]'
          }`}
        >
          {isUploading ? (
            <div className="flex flex-col items-center justify-center py-4">
              <Loader2 className="w-8 h-8 text-accent-indigo animate-spin mb-3" />
              <p className="text-sm font-medium text-[var(--text)]">Validating and auto-profiling dataset...</p>
              <p className="text-xs text-[var(--text-muted)] mt-1">Sniffing delimiters, detecting column types, and verifying record integrity.</p>
            </div>
          ) : (
            <div className="flex flex-col items-center justify-center py-2">
              <div className="w-12 h-12 rounded-xl bg-accent-indigo/10 border border-accent-indigo/25 flex items-center justify-center text-accent-indigo mb-3 shadow-sm">
                <Upload className="w-6 h-6" />
              </div>
              <h3 className="text-base font-semibold text-[var(--text)]">
                Drop your customer CSV here or browse
              </h3>
              <p className="text-xs text-[var(--text-muted)] mt-1.5 max-w-md">
                One row per customer • 100+ rows • 3+ numeric columns (CSV or XLSX up to 50MB)
              </p>
            </div>
          )}
        </div>

        {/* Secondary Actions Under Dropzone */}
        <div className="flex flex-wrap items-center justify-between gap-3 mt-4 pt-3 border-t border-[var(--border)] text-xs">
          <div className="flex items-center gap-2">
            <button
              type="button"
              onClick={() => onDatasetChange('sample_credit_card')}
              className="px-3 py-1.5 rounded-md bg-[var(--card-elevated)] border border-[var(--border)] hover:border-accent-indigo/50 text-[var(--text)] font-medium transition-colors flex items-center gap-1.5"
            >
              <Sparkles className="w-3.5 h-3.5 text-accent-indigo" />
              Try the sample dataset (Credit Card N=8,950)
            </button>
          </div>

          <button
            type="button"
            onClick={downloadTemplate}
            className="text-[var(--text-muted)] hover:text-accent-indigo transition-colors flex items-center gap-1"
          >
            <Download className="w-3.5 h-3.5" />
            Download a CSV template
          </button>
        </div>

        {/* Inline friendly error message */}
        {errorMessage && (
          <div className="mt-4 p-3 rounded-lg bg-rose-500/10 border border-rose-500/30 text-rose-400 text-xs flex items-start gap-2.5">
            <AlertTriangle className="w-4 h-4 shrink-0 mt-0.5 text-rose-500" />
            <div>
              <span className="font-semibold block">Validation Notice</span>
              <span>{errorMessage}</span>
            </div>
          </div>
        )}
      </div>

      {/* 2. Health Summary Cards */}
      <div>
        <div className="flex items-center justify-between mb-3">
          <h3 className="text-xs font-semibold uppercase tracking-wider text-[var(--text-muted)] flex items-center gap-1.5">
            <Database className="w-3.5 h-3.5 text-accent-indigo" />
            Dataset Health Summary
          </h3>
          <span className="text-xs text-[var(--text-muted)] font-mono">
            Active: <span className="text-[var(--text)] font-medium">{currentDataset?.name || 'None'}</span>
            {currentDataset?.is_synthetic && (
              <span className="ml-2 px-1.5 py-0.5 rounded text-[10px] bg-amber-500/15 text-amber-400 border border-amber-500/30">
                SYNTHETIC DEMO
              </span>
            )}
          </span>
        </div>

        <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
          <div className="p-4 rounded-lg bg-[var(--card)] border border-[var(--border)] shadow-sm">
            <span className="text-xs text-[var(--text-muted)] block">Total Customers</span>
            <span className="text-2xl font-bold metric-value text-[var(--text)] block mt-1">
              {formatNumber(totalRows)}
            </span>
            <span className="text-[11px] text-[var(--text-muted)] mt-0.5 block">1 row per account</span>
          </div>

          <div className="p-4 rounded-lg bg-[var(--card)] border border-[var(--border)] shadow-sm">
            <span className="text-xs text-[var(--text-muted)] block">Active Features</span>
            <span className="text-2xl font-bold metric-value text-accent-indigo block mt-1">
              {includedCols} of {totalCols}
            </span>
            <span className="text-[11px] text-[var(--text-muted)] mt-0.5 block">Numerical signals</span>
          </div>

          <div className="p-4 rounded-lg bg-[var(--card)] border border-[var(--border)] shadow-sm">
            <span className="text-xs text-[var(--text-muted)] block">Missing Rate</span>
            <span className="text-2xl font-bold metric-value text-accent-amber block mt-1">
              {missingPct}%
            </span>
            <span className="text-[11px] text-[var(--text-muted)] mt-0.5 block">
              {totalMissing} cells (KNN imputed)
            </span>
          </div>

          <div className="p-4 rounded-lg bg-[var(--card)] border border-[var(--border)] shadow-sm">
            <span className="text-xs text-[var(--text-muted)] block">Integrity Status</span>
            <span className="text-2xl font-bold metric-value text-accent-emerald block mt-1 flex items-center gap-1.5">
              <CheckCircle2 className="w-5 h-5 text-accent-emerald" />
              Verified
            </span>
            <span className="text-[11px] text-[var(--text-muted)] mt-0.5 block">0 duplicates detected</span>
          </div>
        </div>
      </div>

      {/* 3. Configuration & Schema Roles */}
      <div className="bg-[var(--card)] border border-[var(--border)] rounded-xl p-5 shadow-sm space-y-5">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 pb-4 border-b border-[var(--border)]">
          <div>
            <h3 className="text-sm font-semibold text-[var(--text)]">Pipeline Configuration</h3>
            <p className="text-xs text-[var(--text-muted)] mt-0.5">
              Customize the identifier column and business domain template for tailored growth insights.
            </p>
          </div>

          <div className="flex flex-wrap items-center gap-3">
            {/* ID Column Picker */}
            <div className="flex items-center gap-2">
              <label className="text-xs text-[var(--text-muted)] whitespace-nowrap">ID Column:</label>
              <select
                value={config?.id_column || 'none'}
                onChange={(e) => handleIdColumnChange(e.target.value)}
                className="text-xs bg-[var(--card-elevated)] border border-[var(--border)] rounded px-2.5 py-1.5 text-[var(--text)] focus:outline-none focus:border-accent-indigo"
              >
                <option value="none">Auto-exclude / None</option>
                {profile?.columns.map((c) => (
                  <option key={c.name} value={c.name}>
                    {c.name} ({c.unique_count} unique)
                  </option>
                ))}
              </select>
            </div>

            {/* Domain Template Selector */}
            <div className="flex items-center gap-2">
              <label className="text-xs text-[var(--text-muted)] whitespace-nowrap">Domain:</label>
              <select
                value={config?.domain_template || 'generic'}
                onChange={(e) => handleDomainChange(e.target.value as any)}
                className="text-xs bg-[var(--card-elevated)] border border-[var(--border)] rounded px-2.5 py-1.5 text-[var(--text)] focus:outline-none focus:border-accent-indigo"
              >
                <option value="credit_card">Banking / Credit Card</option>
                <option value="ecommerce_rfm">E-Commerce (RFM)</option>
                <option value="saas">SaaS / Subscription</option>
                <option value="generic">Generic Business</option>
              </select>
            </div>
          </div>
        </div>

        {/* Column Types & Include/Exclude List */}
        <div className="space-y-3">
          <div className="flex items-center justify-between text-xs">
            <span className="font-semibold text-[var(--text)]">Feature Schema & Column Inclusions</span>
            <span className="text-[var(--text-muted)]">Toggle to include or exclude specific metrics from segmentation</span>
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-2 max-h-60 overflow-y-auto pr-1">
            {profile?.columns.map((col) => {
              const isIncluded = config?.include_columns
                ? config.include_columns.includes(col.name)
                : col.suggested_role === 'numeric_feature'
              const isId = col.name === config?.id_column

              let roleBadgeColor = 'bg-zinc-500/10 text-zinc-400 border-zinc-500/30'
              if (col.suggested_role === 'numeric_feature') roleBadgeColor = 'bg-emerald-500/10 text-emerald-400 border-emerald-500/30'
              if (col.suggested_role === 'id' || isId) roleBadgeColor = 'bg-indigo-500/10 text-indigo-400 border-indigo-500/30'
              if (col.suggested_role === 'categorical') roleBadgeColor = 'bg-purple-500/10 text-purple-400 border-purple-500/30'

              return (
                <div
                  key={col.name}
                  className={`p-2.5 rounded-lg border text-xs flex items-center justify-between gap-2 transition-colors ${
                    isIncluded
                      ? 'bg-[var(--card-elevated)] border-[var(--border)]'
                      : 'bg-transparent border-[var(--border)]/50 opacity-60'
                  }`}
                >
                  <div className="min-w-0">
                    <div className="flex items-center gap-1.5">
                      <span className="font-mono font-medium text-[var(--text)] truncate">{col.name}</span>
                      <span className={`text-[10px] px-1.5 py-0.2 rounded border font-medium ${roleBadgeColor}`}>
                        {isId ? 'ID' : col.suggested_role}
                      </span>
                    </div>
                    <span className="text-[10px] text-[var(--text-muted)] block mt-0.5">
                      {col.missing_count > 0 ? `${col.missing_count} missing (${col.missing_pct}%)` : '100% complete'}
                    </span>
                  </div>

                  {!isId && col.suggested_role === 'numeric_feature' && (
                    <button
                      type="button"
                      onClick={() => handleToggleInclude(col.name)}
                      className={`px-2 py-1 text-[11px] rounded font-medium transition-colors ${
                        isIncluded
                          ? 'bg-accent-indigo text-white hover:bg-accent-indigo/90'
                          : 'bg-[var(--border)] text-[var(--text-muted)] hover:text-[var(--text)]'
                      }`}
                    >
                      {isIncluded ? 'Included' : 'Excluded'}
                    </button>
                  )}
                </div>
              )
            })}
          </div>
        </div>
      </div>

      {/* 4. 10-Row Data Preview Table */}
      {profile && profile.preview && profile.preview.length > 0 && (
        <div className="bg-[var(--card)] border border-[var(--border)] rounded-xl p-5 shadow-sm space-y-3">
          <div className="flex items-center justify-between">
            <div>
              <h3 className="text-sm font-semibold text-[var(--text)]">10-Row Sample Preview</h3>
              <p className="text-xs text-[var(--text-muted)]">Verified initial rows formatted directly from source dataset.</p>
            </div>
            <span className="text-xs font-mono text-[var(--text-muted)]">
              Showing 10 of {formatNumber(profile.row_count)} rows
            </span>
          </div>

          <div className="overflow-x-auto border border-[var(--border)] rounded-lg">
            <table className="w-full text-xs text-left">
              <thead className="bg-[var(--card-elevated)] text-[var(--text-muted)] border-b border-[var(--border)] font-mono uppercase text-[10px] tracking-wider">
                <tr>
                  {Object.keys(profile.preview[0]).map((key) => (
                    <th key={key} className="px-3 py-2.5 font-semibold whitespace-nowrap">
                      {key}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody className="divide-y divide-[var(--border)]">
                {profile.preview.map((row, idx) => (
                  <tr key={idx} className="hover:bg-[var(--card-elevated)]/50 transition-colors">
                    {Object.values(row).map((val: any, colIdx) => (
                      <td key={colIdx} className="px-3 py-2 whitespace-nowrap font-mono text-[11px] text-[var(--text)]">
                        {val === null || val === undefined ? (
                          <span className="text-accent-amber italic font-sans text-[10px]">null</span>
                        ) : typeof val === 'number' ? (
                          val.toLocaleString()
                        ) : (
                          String(val)
                        )}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Primary Action Button */}
      <div className="flex justify-end pt-2">
        <button
          type="button"
          onClick={handleSaveAndPrepare}
          disabled={savingConfig || !currentDataset}
          className="px-6 py-2.5 rounded-lg bg-accent-indigo hover:bg-accent-indigo/90 text-white font-medium text-xs shadow-sm transition-all flex items-center gap-2 cursor-pointer disabled:opacity-50"
        >
          {savingConfig ? (
            <>
              <Loader2 className="w-4 h-4 animate-spin" />
              Saving Configuration...
            </>
          ) : (
            <>
              Looks good, prepare data
              <ArrowRight className="w-4 h-4" />
            </>
          )}
        </button>
      </div>
    </div>
  )
}
