'use client'

import React, { useState, useEffect } from 'react'
import { Sparkles, X } from 'lucide-react'

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
    <div className="bg-[var(--card)] border border-accent-indigo/30 rounded-2xl p-5 shadow-sm relative overflow-hidden transition-all animate-fadeIn max-w-[1100px] mx-auto">
      <div className="absolute top-0 left-0 right-0 h-0.5 bg-gradient-to-r from-accent-indigo via-accent-emerald to-accent-indigo" />

      <div className="flex items-start justify-between gap-4">
        <div className="flex items-start gap-3.5">
          <div className="w-9 h-9 rounded-xl bg-accent-indigo/15 border border-accent-indigo/40 flex items-center justify-center text-accent-indigo shrink-0 mt-0.5">
            <Sparkles className="w-4 h-4" />
          </div>

          <div className="space-y-2.5">
            <div className="flex items-center gap-2">
              <span className="text-sm font-bold text-[var(--text)]">Quick Start Guide</span>
              <span className="text-[11px] px-2 py-0.5 rounded-full bg-accent-indigo/10 text-accent-indigo font-semibold">
                3 Simple Steps
              </span>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-3 gap-3 pt-1">
              {/* Mark 1 */}
              <div className="flex items-start gap-2.5 text-xs">
                <span className="w-5 h-5 rounded-full bg-[var(--card-elevated)] border border-[var(--border)] text-[var(--text)] font-mono font-semibold flex items-center justify-center shrink-0">
                  1
                </span>
                <div>
                  <span className="font-semibold text-[var(--text)] block text-xs">1. Add your data</span>
                  <span className="text-[var(--text-muted)] text-[11px] leading-relaxed">
                    Drop a customer CSV or click "Try demo data". We inspect columns and missing fields automatically.
                  </span>
                </div>
              </div>

              {/* Mark 2 */}
              <div className="flex items-start gap-2.5 text-xs">
                <span className="w-5 h-5 rounded-full bg-[var(--card-elevated)] border border-[var(--border)] text-[var(--text)] font-mono font-semibold flex items-center justify-center shrink-0">
                  2
                </span>
                <div>
                  <span className="font-semibold text-[var(--text)] block text-xs">2. See your customer groups</span>
                  <span className="text-[var(--text-muted)] text-[11px] leading-relaxed">
                    Click "Find my customer groups". We clean the data, test algorithms, and pick the clearest customer segments.
                  </span>
                </div>
              </div>

              {/* Mark 3 */}
              <div className="flex items-start gap-2.5 text-xs">
                <span className="w-5 h-5 rounded-full bg-[var(--card-elevated)] border border-[var(--border)] text-[var(--text)] font-mono font-semibold flex items-center justify-center shrink-0">
                  3
                </span>
                <div>
                  <span className="font-semibold text-[var(--text)] block text-xs">3. Take action</span>
                  <span className="text-[var(--text-muted)] text-[11px] leading-relaxed">
                    Review tailored offers, copy ready-to-send emails and SMS, simulate ROI, and export customer lists.
                  </span>
                </div>
              </div>
            </div>
          </div>
        </div>

        <button
          onClick={handleDismiss}
          className="p-1.5 rounded-lg hover:bg-[var(--card-elevated)] text-[var(--text-muted)] hover:text-[var(--text)] transition-colors shrink-0"
          title="Dismiss guide"
        >
          <X className="w-4 h-4" />
        </button>
      </div>
    </div>
  )
}
