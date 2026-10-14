'use client'

import React from 'react'
import { Check, Lock } from 'lucide-react'
import { cn } from '@/lib/utils'

export type StepId = 'data' | 'prepare' | 'benchmark' | 'explore' | 'act'

export interface StepItem {
  id: StepId
  number: number
  label: string
  subtitle: string
}

export const STEPS: StepItem[] = [
  { id: 'data', number: 1, label: 'Data', subtitle: 'Upload & Schema' },
  { id: 'prepare', number: 2, label: 'Prepare', subtitle: 'Impute & Reduce' },
  { id: 'benchmark', number: 3, label: 'Benchmark', subtitle: '16 Experiments' },
  { id: 'explore', number: 4, label: 'Explore', subtitle: 'Clusters & PCA' },
  { id: 'act', number: 5, label: 'Act', subtitle: 'Personas & ROI' },
]

interface StepStepperProps {
  currentStep: StepId
  onSelectStep: (step: StepId) => void
  hasData: boolean
  hasPrepared: boolean
  hasBenchmark: boolean
}

export function StepStepper({
  currentStep,
  onSelectStep,
  hasData,
  hasPrepared,
  hasBenchmark,
}: StepStepperProps) {
  // Determine state for each step
  const getStepState = (stepId: StepId) => {
    if (stepId === currentStep) return 'active'

    switch (stepId) {
      case 'data':
        return hasData ? 'done' : 'active'
      case 'prepare':
        if (!hasData) return 'locked'
        return hasPrepared ? 'done' : 'available'
      case 'benchmark':
        if (!hasPrepared) return 'locked'
        return hasBenchmark ? 'done' : 'available'
      case 'explore':
      case 'act':
        return hasBenchmark ? 'available' : 'locked'
      default:
        return 'locked'
    }
  }

  return (
    <nav aria-label="Workflow progress" className="w-full">
      <div className="grid grid-cols-2 sm:grid-cols-5 gap-2 sm:gap-3">
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
                  'w-full text-left p-3 rounded-lg border transition-all flex items-center justify-between',
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
                      'w-6 h-6 rounded-full flex items-center justify-center font-mono text-xs font-semibold shrink-0 transition-colors',
                      isActive
                        ? 'bg-accent-indigo text-white'
                        : isDone
                        ? 'bg-accent-emerald text-white'
                        : isLocked
                        ? 'bg-[var(--border)] text-[var(--text-muted)]'
                        : 'bg-[var(--card-elevated)] border border-[var(--border)] text-[var(--text)]'
                    )}
                  >
                    {isDone ? <Check className="w-3.5 h-3.5 stroke-[2.5]" /> : isLocked ? <Lock className="w-3 h-3" /> : step.number}
                  </div>

                  <div className="truncate">
                    <span
                      className={cn(
                        'block text-xs font-semibold tracking-tight leading-tight truncate',
                        isActive ? 'text-accent-indigo' : 'text-[var(--text)]'
                      )}
                    >
                      {step.label}
                    </span>
                    <span className="block text-[11px] text-[var(--text-muted)] leading-tight truncate mt-0.5">
                      {step.subtitle}
                    </span>
                  </div>
                </div>
              </button>

              {/* Locked tooltip */}
              {isLocked && (
                <div className="absolute -top-8 left-1/2 -translate-x-1/2 hidden group-hover:block z-30 pointer-events-none whitespace-nowrap bg-zinc-900 text-zinc-100 text-[11px] px-2.5 py-1 rounded shadow-lg border border-zinc-800">
                  {step.id === 'prepare' ? 'Select or upload data first' : 'Run the benchmark first'}
                </div>
              )}
            </div>
          )
        })}
      </div>
    </nav>
  )
}
