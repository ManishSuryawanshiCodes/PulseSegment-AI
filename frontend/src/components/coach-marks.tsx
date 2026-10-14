'use client'

import React, { useState, useEffect } from 'react'
import { Sparkles, X, ArrowRight, Check } from 'lucide-react'

interface CoachMarksProps {
  onDismiss?: () => void
}

export function CoachMarks({ onDismiss }: CoachMarksProps) {
  const [visible, setVisible] = useState(false)

  useEffect(() => {
    const dismissed = localStorage.getItem('pulsesegment_coachmarks_dismissed')
    if (!dismissed) {
      setVisible(true)
    }
  }, [])

  const handleDismiss = () => {
    localStorage.setItem('pulsesegment_coachmarks_dismissed', 'true')
    setVisible(false)
    if (onDismiss) onDismiss()
  }

  if (!visible) return null

  return (
    <div className="bg-[var(--card)] border border-accent-indigo/30 rounded-xl p-4 shadow-sm relative overflow-hidden transition-all">
      <div className="absolute top-0 left-0 right-0 h-0.5 bg-gradient-to-r from-accent-indigo via-accent-emerald to-accent-indigo" />

      <div className="flex items-start justify-between gap-4">
        <div className="flex items-start gap-3">
          <div className="w-8 h-8 rounded-lg bg-accent-indigo/15 border border-accent-indigo/40 flex items-center justify-center text-accent-indigo shrink-0 mt-0.5">
            <Sparkles className="w-4 h-4" />
          </div>

          <div className="space-y-2">
            <div className="flex items-center gap-2">
              <span className="text-xs font-semibold text-[var(--text)]">Quick Start Guide</span>
              <span className="text-[10px] px-2 py-0.5 rounded-full bg-accent-indigo/10 text-accent-indigo font-medium">3-Step Flow</span>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-3 gap-3 pt-1">
              <div className="flex items-start gap-2 text-xs">
                <span className="w-5 h-5 rounded-full bg-[var(--card-elevated)] border border-[var(--border)] text-[var(--text)] font-mono font-medium flex items-center justify-center shrink-0">1</span>
                <div>
                  <span className="font-medium text-[var(--text)] block">Upload or pick sample</span>
                  <span className="text-[var(--text-muted)] text-[11px]">Preview data, inspect auto-detected types, and click Prepare.</span>
                </div>
              </div>

              <div className="flex items-start gap-2 text-xs">
                <span className="w-5 h-5 rounded-full bg-[var(--card-elevated)] border border-[var(--border)] text-[var(--text)] font-mono font-medium flex items-center justify-center shrink-0">2</span>
                <div>
                  <span className="font-medium text-[var(--text)] block">Run 16 experiments</span>
                  <span className="text-[var(--text-muted)] text-[11px]">Compare K-Means, Hierarchical, DBSCAN, and GMM objectively.</span>
                </div>
              </div>

              <div className="flex items-start gap-2 text-xs">
                <span className="w-5 h-5 rounded-full bg-[var(--card-elevated)] border border-[var(--border)] text-[var(--text)] font-mono font-medium flex items-center justify-center shrink-0">3</span>
                <div>
                  <span className="font-medium text-[var(--text)] block">Deploy personas & ROI</span>
                  <span className="text-[var(--text-muted)] text-[11px]">Personalize campaigns, simulate revenue lift, and export segments.</span>
                </div>
              </div>
            </div>
          </div>
        </div>

        <button
          onClick={handleDismiss}
          className="p-1.5 rounded-md hover:bg-[var(--card-elevated)] text-[var(--text-muted)] hover:text-[var(--text)] transition-colors shrink-0"
          title="Dismiss guide"
        >
          <X className="w-4 h-4" />
        </button>
      </div>
    </div>
  )
}
