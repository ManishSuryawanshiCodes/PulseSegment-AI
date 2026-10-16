'use client'

import React from 'react'
import { X, HelpCircle, BookOpen, Layers, CheckCircle2, FileText, Sparkles } from 'lucide-react'

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
            <div className="flex items-center gap-2.5">
              <div className="w-8 h-8 rounded-xl bg-accent-indigo/10 border border-accent-indigo/30 flex items-center justify-center text-accent-indigo">
                <HelpCircle className="w-4 h-4" />
              </div>
              <div>
                <h3 className="font-bold text-sm">Help & Guide</h3>
                <p className="text-xs text-[var(--text-muted)]">Everything you need to know in plain English</p>
              </div>
            </div>
            <button
              onClick={onClose}
              className="p-1.5 rounded-lg hover:bg-[var(--card-elevated)] text-[var(--text-muted)] hover:text-[var(--text)] transition-colors"
            >
              <X className="w-4 h-4" />
            </button>
          </div>

          {/* Section 1: How It Works in 3 Steps */}
          <div className="space-y-3">
            <h4 className="text-xs font-semibold uppercase tracking-wider text-[var(--text-muted)] flex items-center gap-1.5">
              <BookOpen className="w-3.5 h-3.5 text-accent-indigo" />
              How It Works in 3 Steps
            </h4>

            <div className="space-y-2.5 text-xs">
              <div className="p-3.5 rounded-xl bg-[var(--card-elevated)] border border-[var(--border)] space-y-1">
                <div className="font-semibold text-[var(--text)] flex items-center gap-2">
                  <span className="w-5 h-5 rounded-full bg-accent-indigo/20 text-accent-indigo text-[11px] font-bold flex items-center justify-center">
                    1
                  </span>
                  Step 1: Add data
                </div>
                <p className="text-[var(--text-muted)] leading-relaxed">
                  Upload your customer CSV or click "Try demo data". The system automatically verifies customer IDs, checks for missing entries, and previews your columns.
                </p>
              </div>

              <div className="p-3.5 rounded-xl bg-[var(--card-elevated)] border border-[var(--border)] space-y-1">
                <div className="font-semibold text-[var(--text)] flex items-center gap-2">
                  <span className="w-5 h-5 rounded-full bg-accent-indigo/20 text-accent-indigo text-[11px] font-bold flex items-center justify-center">
                    2
                  </span>
                  Step 2: Your customer groups
                </div>
                <p className="text-[var(--text-muted)] leading-relaxed">
                  Click "Find my customer groups". We automatically fill missing values, test multiple grouping algorithms, and pick the one with the highest clarity. You get distinct customer cards with real-unit typical metrics and key differences.
                </p>
              </div>

              <div className="p-3.5 rounded-xl bg-[var(--card-elevated)] border border-[var(--border)] space-y-1">
                <div className="font-semibold text-[var(--text)] flex items-center gap-2">
                  <span className="w-5 h-5 rounded-full bg-accent-indigo/20 text-accent-indigo text-[11px] font-bold flex items-center justify-center">
                    3
                  </span>
                  Step 3: Take action
                </div>
                <p className="text-[var(--text-muted)] leading-relaxed">
                  Put your segments to work. Review recommended offers, copy ready-to-send marketing messages (Email, SMS, Push), simulate ROI with your revenue estimates, and download the full customer list with assigned groups.
                </p>
              </div>
            </div>
          </div>

          {/* Section 2: 6-Term Non-Jargon Glossary */}
          <div className="space-y-3">
            <h4 className="text-xs font-semibold uppercase tracking-wider text-[var(--text-muted)] flex items-center gap-1.5">
              <Sparkles className="w-3.5 h-3.5 text-accent-emerald" />
              6-Term Plain English Glossary
            </h4>

            <div className="space-y-2.5 text-xs">
              <div className="border border-[var(--border)] rounded-xl p-3 bg-[var(--card-elevated)]">
                <span className="font-semibold text-[var(--text)]">1. Customer group</span>
                <p className="text-[var(--text-muted)] mt-1 leading-relaxed">
                  A distinct cluster of customers who share similar shopping, payment, or usage behaviors.
                </p>
              </div>

              <div className="border border-[var(--border)] rounded-xl p-3 bg-[var(--card-elevated)]">
                <span className="font-semibold text-[var(--text)]">2. Typical customer</span>
                <p className="text-[var(--text-muted)] mt-1 leading-relaxed">
                  The average profile of people inside a group, expressed in real numbers (like average balance or total purchases).
                </p>
              </div>

              <div className="border border-[var(--border)] rounded-xl p-3 bg-[var(--card-elevated)]">
                <span className="font-semibold text-[var(--text)]">3. Recommended offer</span>
                <p className="text-[var(--text-muted)] mt-1 leading-relaxed">
                  The specific marketing strategy, campaign, or discount most likely to motivate that particular group.
                </p>
              </div>

              <div className="border border-[var(--border)] rounded-xl p-3 bg-[var(--card-elevated)]">
                <div className="flex items-center justify-between">
                  <span className="font-semibold text-[var(--text)]">4. Group clarity</span>
                  <span className="text-[10px] text-accent-emerald font-semibold">Higher is better (Max: 1.0)</span>
                </div>
                <p className="text-[var(--text-muted)] mt-1 leading-relaxed">
                  How well-separated the customer groups are from one another. Higher clarity means clean, easy-to-target groups.
                </p>
              </div>

              <div className="border border-[var(--border)] rounded-xl p-3 bg-[var(--card-elevated)]">
                <div className="flex items-center justify-between">
                  <span className="font-semibold text-[var(--text)]">5. Group overlap</span>
                  <span className="text-[10px] text-accent-emerald font-semibold">Lower is better</span>
                </div>
                <p className="text-[var(--text-muted)] mt-1 leading-relaxed">
                  How much different customer groups blend together. Lower overlap means groups have unique boundaries.
                </p>
              </div>

              <div className="border border-[var(--border)] rounded-xl p-3 bg-[var(--card-elevated)]">
                <div className="flex items-center justify-between">
                  <span className="font-semibold text-[var(--text)]">6. Separation strength</span>
                  <span className="text-[10px] text-accent-emerald font-semibold">Higher is better</span>
                </div>
                <p className="text-[var(--text-muted)] mt-1 leading-relaxed">
                  Confirms that differences between customer groups are significantly stronger than random variation or noise.
                </p>
              </div>
            </div>
          </div>

          {/* Section 3: CSV Format Guide */}
          <div className="space-y-2 text-xs">
            <h4 className="text-xs font-semibold uppercase tracking-wider text-[var(--text-muted)] flex items-center gap-1.5">
              <FileText className="w-3.5 h-3.5 text-accent-amber" />
              CSV Format Guide
            </h4>
            <div className="p-3.5 rounded-xl bg-[var(--card-elevated)] border border-[var(--border)] space-y-1.5 text-[var(--text-muted)] leading-relaxed">
              <p>• <strong>Structure:</strong> Exactly one row per customer.</p>
              <p>• <strong>Rows:</strong> Minimum 100 customer rows.</p>
              <p>• <strong>Columns:</strong> At least 3 numerical columns (e.g. spend, orders, balance).</p>
              <p>• <strong>Customer ID:</strong> An identifier column (like <code className="text-accent-indigo font-mono">CUST_ID</code>) is automatically detected and excluded from calculations.</p>
              <p>• <strong>File type:</strong> Standard CSV or Excel (.xlsx) file up to 50MB.</p>
            </div>
          </div>
        </div>

        {/* Footer */}
        <div className="pt-4 mt-6 border-t border-[var(--border)] flex justify-end">
          <button
            onClick={onClose}
            className="px-4 py-2 text-xs font-semibold bg-accent-indigo text-white rounded-xl hover:bg-accent-indigo/90 transition-colors"
          >
            Got it, close
          </button>
        </div>
      </div>
    </div>
  )
}
