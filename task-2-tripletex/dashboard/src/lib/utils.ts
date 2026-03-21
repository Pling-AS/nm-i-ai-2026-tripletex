import { clsx, type ClassValue } from 'clsx'
import { twMerge } from 'tailwind-merge'
import { formatDistanceToNow } from 'date-fns'

export function formatDuration(seconds: number | null): string {
  if (seconds === null || seconds === undefined) return '-'
  
  const h = Math.floor(seconds / 3600)
  const m = Math.floor((seconds % 3600) / 60)
  const s = Math.floor(seconds % 60)

  if (h > 0) return `${h}h ${m}m`
  if (m > 0) return `${m}m ${s}s`
  return `${s}s`
}

export function formatTimeAgo(isoDate: string): string {
  if (!isoDate) return ''
  try {
    const date = new Date(isoDate)
    return formatDistanceToNow(date, { addSuffix: true })
      .replace('about ', '')
      .replace('less than a minute ago', 'just now')
      .replace(' minutes', 'm')
      .replace(' minute', 'm')
      .replace(' hours', 'h')
      .replace(' hour', 'h')
      .replace(' days', 'd')
      .replace(' day', 'd')
  } catch (e) {
    return ''
  }
}

export function shortModel(model: string): string {
  if (!model) return ''
  const parts = model.split('/')
  const name = parts[parts.length - 1]
  return name.split(':')[0]
}

export function truncate(text: string, maxLen: number): string {
  if (!text) return ''
  if (text.length <= maxLen) return text
  return text.slice(0, maxLen) + '...'
}

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs))
}
