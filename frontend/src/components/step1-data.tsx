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
  CreditCard,
  ShoppingBag,
  Repeat,
  Building2,
  Sparkles,
  ArrowRight,
  Download,
  AlertTriangle,
  CheckCircle2,
  ChevronDown,
  ChevronUp,
  Loader2,
  Sliders,
  Filter,
} from 'lucide-react'

interface Step1DataProps {
  currentDataset: DatasetInfo | null
  profile: DatasetProfile | null
  config: DatasetConfig | null
  onDatasetChange: (datasetId: string) => void
  onConfigChange: (config: DatasetConfig) => void
  onFindCustomerGroups: () => void
  isFindingGroups: boolean
  progressMessage?: string
  isUploading: boolean
  onUploadFile: (file: File) => Promise<void>
  showAdvanced?: boolean
}

export function Step1Data({
  currentDataset,
  profile,
  config,
  onDatasetChange,
  onConfigChange,
  onFindCustomerGroups,
  isFindingGroups,
  progressMessage = 'Cleaning your data... Trying different methods... Choosing the best',
  isUploading,
  onUploadFile,
  showAdvanced = false,
}: Step1DataProps) {
  const [dragActive, setDragActive] = useState(false)
  const [errorMessage, setErrorMessage] = useState<string | null>(null)
  const [advancedOpen, setAdvancedOpen] = useState(false)
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
      setErrorMessage("We couldn't read this file. Make sure it's a standard CSV or Excel file with one row per customer.")
      return
    }

    try {
      await onUploadFile(file)
    } catch (err: any) {
      setErrorMessage(
        err.message ||
          "We couldn't read this file. Make sure it has one row per customer and at least 3 numerical columns."
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
    link.setAttribute('download', 'customer_data_template.csv')
    document.body.appendChild(link)
    link.click()
    document.body.removeChild(link)
  }

  // Handle configuration changes
  const handleToggleInclude = (colName: string) => {
    if (!config || !profile) return
    const currentIncludes =
      config.include_columns ||
      profile.columns
        .filter((c) => c.suggested_role === 'numeric_feature')
        .map((c) => c.name)
    let nextIncludes: string[]
    if (currentIncludes.includes(colName)) {
      if (currentIncludes.length <= 3) {
        setErrorMessage('Keep at least 3 numeric columns for accurate customer grouping.')
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

  // Key stats
  const totalRows = profile?.row_count || currentDataset?.row_count || 0
  const totalCols = profile?.col_count || currentDataset?.col_count || 0
  const totalMissing = profile?.columns?.reduce((acc, c) => acc + c.missing_count, 0) || 0
  const totalCells = totalRows * (totalCols || 1)
  const missingPctNum = totalCells > 0 ? (totalMissing / totalCells) * 100 : 0
  const missingPct = missingPctNum.toFixed(1)

  // 5-row preview slice
  const previewRows = profile?.preview ? profile.preview.slice(0, 5) : []

  return (
    <div className="space-y-6 max-w-[1100px] mx-auto animate-fadeIn">
      {/* Real Kaggle File Missing Notice */}
      {(currentDataset?.real_file_missing || profile?.real_file_missing) && (
        <div className="p-4 rounded-xl bg-amber-500/10 border border-amber-500/30 text-amber-300 text-sm flex items-start gap-3 shadow-sm">
          <AlertTriangle className="w-5 h-5 text-amber-400 shrink-0 mt-0.5" />
          <div className="space-y-1">
            <span className="font-semibold text-amber-200 block">
              Using demo data. Add your real customer file for personalized results
            </span>
            <p className="text-zinc-300 text-xs leading-relaxed">
              Default location: <code className="px-1.5 py-0.5 rounded bg-zinc-900 border border-zinc-800 text-amber-300 font-mono text-xs select-all">{currentDataset?.real_file_path || profile?.real_file_path || 'backend/data/cc_general.csv'}</code>
            </p>
          </div>
        </div>
      )}

      {/* 1. Big Dropzone Card */}
      <div className="bg-[var(--card)] border border-[var(--border)] rounded-2xl p-6 sm:p-8 shadow-sm">
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
          className={`border-2 border-dashed rounded-xl p-8 sm:p-10 text-center cursor-pointer transition-all ${
            dragActive
              ? 'border-accent-indigo bg-accent-indigo/10'
              : 'border-[var(--border)] hover:border-accent-indigo/60 hover:bg-[var(--card-elevated)]'
          }`}
        >
          {isUploading ? (
            <div className="flex flex-col items-center justify-center py-6 space-y-3">
              <Loader2 className="w-9 h-9 text-accent-indigo animate-spin" />
              <p className="text-base font-semibold text-[var(--text)]">Reading customer file...</p>
              <p className="text-xs text-[var(--text-muted)]">Verifying customer rows and column types.</p>
            </div>
          ) : (
            <div className="flex flex-col items-center justify-center py-4 space-y-3">
              <div className="w-14 h-14 rounded-2xl bg-accent-indigo/10 border border-accent-indigo/25 flex items-center justify-center text-accent-indigo shadow-sm">
                <Upload className="w-7 h-7" />
              </div>
              <div>
                <h3 className="text-base sm:text-lg font-semibold text-[var(--text)]">
                  Drop your customer CSV here or browse
                </h3>
                <p className="text-xs text-[var(--text-muted)] mt-1 max-w-md mx-auto">
                  One row per customer • 100+ rows • 3+ numeric columns (CSV or XLSX up to 50MB)
                </p>
              </div>
            </div>
          )}
        </div>

        {/* Secondary Actions Under Dropzone */}
        <div className="flex flex-wrap items-center justify-between gap-3 mt-5 pt-4 border-t border-[var(--border)] text-xs">
          <button
            type="button"
            onClick={() => onDatasetChange('sample_credit_card')}
            className="px-3.5 py-2 rounded-xl bg-[var(--card-elevated)] border border-[var(--border)] hover:border-accent-indigo/50 text-[var(--text)] font-medium transition-colors flex items-center gap-2"
          >
            <Sparkles className="w-4 h-4 text-accent-indigo" />
            <span>Try demo data</span>
          </button>

          <button
            type="button"
            onClick={downloadTemplate}
            className="text-[var(--text-muted)] hover:text-accent-indigo transition-colors flex items-center gap-1.5 font-medium py-1"
          >
            <Download className="w-4 h-4" />
            <span>Download template</span>
          </button>
        </div>

        {/* Inline friendly error message */}
        {errorMessage && (
          <div className="mt-4 p-3.5 rounded-xl bg-rose-500/10 border border-rose-500/30 text-rose-400 text-xs flex items-start gap-2.5">
            <AlertTriangle className="w-4 h-4 shrink-0 mt-0.5 text-rose-500" />
            <div>
              <span className="font-semibold block text-sm">Notice</span>
              <span className="text-xs">{errorMessage}</span>
            </div>
          </div>
        )}
      </div>

      {/* 2. After Upload Summary: File Name, Customers, Columns, Warning */}
      {currentDataset && (
        <div className="bg-[var(--card)] border border-[var(--border)] rounded-2xl p-6 shadow-sm space-y-6">
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
            <div className="p-4 rounded-xl bg-[var(--card-elevated)] border border-[var(--border)]">
              <span className="text-xs text-[var(--text-muted)] block font-medium">Customer file</span>
              <div className="flex items-center gap-2 mt-1">
                <span className="text-base font-semibold text-[var(--text)] truncate">
                  {currentDataset.name}
                </span>
                {currentDataset.is_synthetic && (
                  <span className="px-2 py-0.5 rounded-full text-[10px] font-semibold bg-amber-500/15 text-amber-500 dark:text-amber-400 border border-amber-500/30 shrink-0">
                    Demo data
                  </span>
                )}
              </div>
            </div>

            <div className="p-4 rounded-xl bg-[var(--card-elevated)] border border-[var(--border)]">
              <span className="text-xs text-[var(--text-muted)] block font-medium">Number of customers</span>
              <span className="text-2xl font-bold metric-value text-[var(--text)] block mt-0.5">
                {formatNumber(totalRows)}
              </span>
            </div>

            <div className="p-4 rounded-xl bg-[var(--card-elevated)] border border-[var(--border)]">
              <span className="text-xs text-[var(--text-muted)] block font-medium">Columns found</span>
              <span className="text-2xl font-bold metric-value text-accent-indigo block mt-0.5">
                {totalCols}
              </span>
            </div>
          </div>

          {/* Plain English Missing Values Notice */}
          <div className="p-3.5 rounded-xl border flex items-center gap-3 text-xs">
            {totalMissing > 0 ? (
              <div className="flex items-center gap-2.5 text-amber-500 dark:text-amber-400">
                <AlertTriangle className="w-4 h-4 shrink-0" />
                <span>
                  <strong>{missingPct}% of values were missing</strong> across {formatNumber(totalMissing)} cells. We will fill them in automatically based on similar customers.
                </span>
              </div>
            ) : (
              <div className="flex items-center gap-2.5 text-accent-emerald">
                <CheckCircle2 className="w-4 h-4 shrink-0" />
                <span>
                  <strong>All values are complete!</strong> No missing customer information was found.
                </span>
              </div>
            )}
          </div>

          {/* "What kind of business is this?" with icons */}
          <div className="space-y-3 pt-2">
            <div>
              <h4 className="text-sm font-semibold text-[var(--text)]">What kind of business is this?</h4>
              <p className="text-xs text-[var(--text-muted)] mt-0.5">
                This helps us recommend the most relevant campaign ideas for each customer group.
              </p>
            </div>

            <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
              {[
                {
                  id: 'credit_card',
                  label: 'Banking & Credit Card',
                  icon: CreditCard,
                  desc: 'Balances, limits, and payments',
                },
                {
                  id: 'ecommerce_rfm',
                  label: 'Online Store & Retail',
                  icon: ShoppingBag,
                  desc: 'Orders, recency, and spend',
                },
                {
                  id: 'saas',
                  label: 'Subscription & SaaS',
                  icon: Repeat,
                  desc: 'Monthly plans and usage',
                },
                {
                  id: 'generic',
                  label: 'General Business',
                  icon: Building2,
                  desc: 'Flexible custom metrics',
                },
              ].map((item) => {
                const isSelected = (config?.domain_template || 'generic') === item.id
                const Icon = item.icon
                return (
                  <button
                    key={item.id}
                    type="button"
                    onClick={() => handleDomainChange(item.id as any)}
                    className={`p-3.5 rounded-xl border text-left transition-all flex flex-col justify-between gap-2.5 ${
                      isSelected
                        ? 'bg-accent-indigo/10 border-accent-indigo text-accent-indigo shadow-sm'
                        : 'bg-[var(--card-elevated)] border-[var(--border)] text-[var(--text)] hover:border-accent-indigo/40'
                    }`}
                  >
                    <div className="flex items-center justify-between">
                      <div
                        className={`w-8 h-8 rounded-lg flex items-center justify-center ${
                          isSelected
                            ? 'bg-accent-indigo text-white'
                            : 'bg-[var(--card)] border border-[var(--border)] text-[var(--text-muted)]'
                        }`}
                      >
                        <Icon className="w-4 h-4" />
                      </div>
                      {isSelected && (
                        <span className="w-2 h-2 rounded-full bg-accent-indigo" />
                      )}
                    </div>
                    <div>
                      <span className="block text-xs font-semibold">{item.label}</span>
                      <span className="block text-[11px] text-[var(--text-muted)] mt-0.5">
                        {item.desc}
                      </span>
                    </div>
                  </button>
                )
              })}
            </div>
          </div>

          {/* 5-Row Data Preview */}
          {previewRows.length > 0 && (
            <div className="space-y-2.5 pt-2">
              <div className="flex items-center justify-between">
                <h4 className="text-xs font-semibold uppercase tracking-wider text-[var(--text-muted)]">
                  Preview (First 5 Customers)
                </h4>
                <span className="text-xs text-[var(--text-muted)] font-mono">
                  Showing 5 of {formatNumber(totalRows)}
                </span>
              </div>

              <div className="overflow-x-auto border border-[var(--border)] rounded-xl max-h-56">
                <table className="w-full text-xs text-left">
                  <thead className="bg-[var(--card-elevated)] text-[var(--text-muted)] border-b border-[var(--border)] font-mono text-[11px]">
                    <tr>
                      {Object.keys(previewRows[0]).map((key) => (
                        <th key={key} className="px-3.5 py-2 font-medium whitespace-nowrap">
                          {key}
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-[var(--border)]">
                    {previewRows.map((row, idx) => (
                      <tr key={idx} className="hover:bg-[var(--card-elevated)]/40 transition-colors">
                        {Object.values(row).map((val: any, colIdx) => (
                          <td
                            key={colIdx}
                            className="px-3.5 py-2 whitespace-nowrap font-mono text-[11px] text-[var(--text)]"
                          >
                            {val === null || val === undefined ? (
                              <span className="text-amber-500 italic font-sans text-[11px]">missing</span>
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

          {/* Advanced Settings Collapsible (ID column & Column include/exclude) */}
          <div className="pt-2 border-t border-[var(--border)]">
            <button
              type="button"
              onClick={() => setAdvancedOpen(!advancedOpen)}
              className="flex items-center gap-2 text-xs font-semibold text-[var(--text-muted)] hover:text-[var(--text)] transition-colors py-1"
            >
              <Sliders className="w-3.5 h-3.5" />
              <span>Advanced settings</span>
              {advancedOpen ? (
                <ChevronUp className="w-3.5 h-3.5" />
              ) : (
                <ChevronDown className="w-3.5 h-3.5" />
              )}
            </button>

            {advancedOpen && (
              <div className="mt-4 p-4 rounded-xl bg-[var(--card-elevated)] border border-[var(--border)] space-y-4 animate-fadeIn text-xs">
                {/* ID Column Picker */}
                <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 pb-3 border-b border-[var(--border)]">
                  <div>
                    <span className="font-semibold text-[var(--text)] block">Customer identifier column</span>
                    <span className="text-[var(--text-muted)] text-[11px] block">
                      Excluded from grouping calculations to prevent identity bias.
                    </span>
                  </div>
                  <select
                    value={config?.id_column || 'none'}
                    onChange={(e) => handleIdColumnChange(e.target.value)}
                    className="bg-[var(--card)] border border-[var(--border)] rounded-lg px-3 py-1.5 text-xs text-[var(--text)] focus:outline-none focus:border-accent-indigo"
                  >
                    <option value="none">Auto-exclude / None</option>
                    {profile?.columns.map((c) => (
                      <option key={c.name} value={c.name}>
                        {c.name}
                      </option>
                    ))}
                  </select>
                </div>

                {/* Column Include/Exclude */}
                <div className="space-y-2">
                  <span className="font-semibold text-[var(--text)] block">Included numeric columns</span>
                  <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-2 max-h-48 overflow-y-auto pr-1">
                    {profile?.columns.map((col) => {
                      const isIncluded = config?.include_columns
                        ? config.include_columns.includes(col.name)
                        : col.suggested_role === 'numeric_feature'
                      const isId = col.name === config?.id_column

                      return (
                        <div
                          key={col.name}
                          className={`p-2 rounded-lg border flex items-center justify-between gap-2 ${
                            isIncluded
                              ? 'bg-[var(--card)] border-[var(--border)]'
                              : 'bg-transparent border-[var(--border)]/50 opacity-60'
                          }`}
                        >
                          <div className="min-w-0">
                            <span className="font-mono text-xs text-[var(--text)] truncate block">
                              {col.name}
                            </span>
                          </div>

                          {!isId && col.suggested_role === 'numeric_feature' && (
                            <button
                              type="button"
                              onClick={() => handleToggleInclude(col.name)}
                              className={`px-2 py-0.5 rounded text-[11px] font-medium transition-colors ${
                                isIncluded
                                  ? 'bg-accent-indigo text-white'
                                  : 'bg-[var(--border)] text-[var(--text-muted)]'
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
            )}
          </div>

          {/* Primary Action Button: "Find my customer groups" */}
          <div className="pt-2 flex flex-col sm:flex-row items-center justify-between gap-4 border-t border-[var(--border)]">
            <div className="text-xs text-[var(--text-muted)]">
              {isFindingGroups ? (
                <div className="flex items-center gap-2 text-accent-indigo font-medium">
                  <Loader2 className="w-4 h-4 animate-spin shrink-0" />
                  <span>{progressMessage}</span>
                </div>
              ) : (
                <span>Ready to clean, compare methods, and discover your customer groups.</span>
              )}
            </div>

            <button
              type="button"
              onClick={onFindCustomerGroups}
              disabled={isFindingGroups || !currentDataset}
              className="w-full sm:w-auto px-7 py-3 rounded-xl bg-accent-indigo hover:bg-accent-indigo/90 text-white font-semibold text-sm shadow-sm transition-all flex items-center justify-center gap-2 cursor-pointer disabled:opacity-50"
            >
              {isFindingGroups ? (
                <>
                  <Loader2 className="w-4 h-4 animate-spin" />
                  <span>Finding groups...</span>
                </>
              ) : (
                <>
                  <Sparkles className="w-4 h-4" />
                  <span>Find my customer groups</span>
                  <ArrowRight className="w-4 h-4" />
                </>
              )}
            </button>
          </div>
        </div>
      )}
    </div>
  )
}
