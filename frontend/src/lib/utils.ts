import { clsx, type ClassValue } from 'clsx'
import { twMerge } from 'tailwind-merge'

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs))
}

export function formatMetric(value: number, decimals: number = 3): string {
  if (value === undefined || value === null || isNaN(value)) return 'N/A'
  return value.toFixed(decimals)
}

export function formatMs(value: number): string {
  if (value === undefined || value === null || isNaN(value)) return '0 ms'
  return `${Math.round(value)} ms`
}

export function formatPercent(value: number): string {
  if (value === undefined || value === null || isNaN(value)) return '0.0%'
  return `${value.toFixed(1)}%`
}

export function formatNumber(value: number): string {
  if (value === undefined || value === null || isNaN(value)) return '0'
  return new Intl.NumberFormat().format(Math.round(value))
}
