'use client'

import React from 'react'
import { Check, Lock, Sparkles } from 'lucide-react'
import { cn } from '@/lib/utils'

export type StepId = 'data' | 'groups' | 'act' | 'methodology'

export interface StepItem {
  id: 'data' | 'groups' | 'act'
  number: number
  label: string
  subtitle: string
}

export const STEPS: StepItem[] = [
  { id: 'data', number: 1, label: 'Add data', subtitle: 'Upload or demo' },
  { id: 'groups', number: 2, label: 'Your customer groups', subtitle: 'Explore segments' },
  { id: 'act', number: 3, label: 'Take action', subtitle: 'Campaigns & ROI' },
]

interface StepStepperProps {
  currentStep: StepId
  onSelectStep: (step: StepId) => void
  hasData: boolean
  hasBenchmark: boolean
}

export function StepStepper({
  currentStep,
  onSelectStep,
  hasData,
  hasBenchmark,
}: StepStepperProps) {
  const getStepState = (stepId: 'data' | 'groups' | 'act') => {
    if (stepId === currentStep) return 'active'

    switch (stepId) {
      case 'data':
        return hasData ? 'done' : 'active'
      case 'groups':
      case 'act':
        return hasBenchmark ? 'available' : 'locked'
      default:
        return 'locked'
    }
  }

  return (
    <nav aria-label="Workflow progress" className="w-full space-y-2">
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
        {/* 3-Step Main Buttons */}
        <div className="grid grid-cols-3 gap-2 sm:gap-3 flex-1">
          {STEPS.map((step) => {
            const state = getStepState(step.id)
            const isActive = step.id === currentStep
            const isDone = state === 'done'
            const isLocked = state === 'locked'

            return (
              <div key={step.id} className="relative group">
                <button
                  type="button"
                  onClick={() => {
                    if (!isLocked) {
                      onSelectStep(step.id)
                    }
                  }}
                  disabled={isLocked}
                  className={cn(
                    'w-full text-left p-3 rounded-xl border transition-all flex items-center justify-between',
                    isActive
                      ? 'bg-accent-indigo/10 border-accent-indigo text-accent-indigo shadow-sm'
                      : isDone
                      ? 'bg-[var(--card)] border-[var(--border)] text-[var(--text)] hover:border-accent-indigo/50 cursor-pointer'
                      : isLocked
                      ? 'bg-[var(--card-elevated)] border-transparent text-[var(--text-muted)] opacity-60 cursor-not-allowed'
                      : 'bg-[var(--card)] border-[var(--border)] text-[var(--text)] hover:border-[var(--text-muted)] cursor-pointer'
                  )}
                >
                  <div className="flex items-center gap-2.5 min-w-0">
                    <div
                      className={cn(
                        'w-7 h-7 rounded-full flex items-center justify-center font-mono text-xs font-semibold shrink-0 transition-colors',
                        isActive
                          ? 'bg-accent-indigo text-white'
                          : isDone
                          ? 'bg-accent-emerald text-white'
                          : isLocked
                          ? 'bg-[var(--border)] text-[var(--text-muted)]'
                          : 'bg-[var(--card-elevated)] border border-[var(--border)] text-[var(--text)]'
                      )}
                    >
                      {isDone ? (
                        <Check className="w-4 h-4 stroke-[2.5]" />
                      ) : isLocked ? (
                        <Lock className="w-3.5 h-3.5" />
                      ) : (
                        step.number
                      )}
                    </div>

                    <div className="truncate">
                      <span
                        className={cn(
                          'block text-sm font-semibold tracking-tight leading-tight truncate',
                          isActive ? 'text-accent-indigo' : 'text-[var(--text)]'
                        )}
                      >
                        {step.label}
                      </span>
                      <span className="hidden sm:block text-xs text-[var(--text-muted)] leading-tight truncate mt-0.5">
                        {step.subtitle}
                      </span>
                    </div>
                  </div>
                </button>

                {/* Locked tooltip */}
                {isLocked && (
                  <div className="absolute -top-8 left-1/2 -translate-x-1/2 hidden group-hover:block z-30 pointer-events-none whitespace-nowrap bg-zinc-900 text-zinc-100 text-xs px-2.5 py-1 rounded shadow-lg border border-zinc-800">
                    Find customer groups first
                  </div>
                )}
              </div>
            )
          })}
        </div>

        {/* Link: How we got these results */}
        <div className="flex justify-end sm:justify-auto shrink-0">
          <button
            type="button"
            onClick={() => onSelectStep('methodology')}
            className={cn(
              'px-3.5 py-2.5 rounded-xl border text-xs font-medium flex items-center gap-1.5 transition-colors',
              currentStep === 'methodology'
                ? 'bg-accent-indigo/15 border-accent-indigo text-accent-indigo font-semibold'
                : 'bg-[var(--card)] border-[var(--border)] text-[var(--text-muted)] hover:text-[var(--text)] hover:border-accent-indigo/50'
            )}
            title="Inspect how data was cleaned, compared, and chosen"
          >
            <Sparkles className="w-3.5 h-3.5 text-accent-indigo shrink-0" />
            <span>How we got these results</span>
          </button>
        </div>
      </div>
    </nav>
  )
}
