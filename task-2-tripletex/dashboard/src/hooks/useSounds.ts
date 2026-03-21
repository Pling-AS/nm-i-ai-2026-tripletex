'use client'

import { useCallback, useRef } from 'react'
import { useStore } from '@/lib/store'

export function useSounds() {
  const { soundEnabled } = useStore()
  const audioCtxRef = useRef<AudioContext | null>(null)

  const getAudioContext = useCallback(() => {
    if (!audioCtxRef.current) {
      try {
        audioCtxRef.current = new (window.AudioContext || (window as any).webkitAudioContext)()
      } catch (e) {
        return null
      }
    }
    if (audioCtxRef.current.state === 'suspended') {
      audioCtxRef.current.resume().catch(() => {})
    }
    return audioCtxRef.current
  }, [])

  const playTone = useCallback((frequency: number, type: OscillatorType, duration: number, startTimeOffset = 0) => {
    const ctx = getAudioContext()
    if (!ctx) return

    try {
      const oscillator = ctx.createOscillator()
      const gainNode = ctx.createGain()

      oscillator.type = type
      oscillator.frequency.setValueAtTime(frequency, ctx.currentTime + startTimeOffset)

      gainNode.gain.setValueAtTime(0, ctx.currentTime + startTimeOffset)
      gainNode.gain.linearRampToValueAtTime(0.1, ctx.currentTime + startTimeOffset + 0.05)
      gainNode.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime + startTimeOffset + duration)

      oscillator.connect(gainNode)
      gainNode.connect(ctx.destination)

      oscillator.start(ctx.currentTime + startTimeOffset)
      oscillator.stop(ctx.currentTime + startTimeOffset + duration)
    } catch (e) {
    }
  }, [getAudioContext])

  const playStart = useCallback(() => {
    if (!soundEnabled) return
    playTone(440, 'sine', 0.1)
    playTone(660, 'sine', 0.15, 0.1)
  }, [soundEnabled, playTone])

  const playComplete = useCallback(() => {
    if (!soundEnabled) return
    playTone(523.25, 'sine', 0.2)
    playTone(659.25, 'sine', 0.4, 0.15)
  }, [soundEnabled, playTone])

  const playError = useCallback(() => {
    if (!soundEnabled) return
    playTone(150, 'sawtooth', 0.3)
    playTone(140, 'sawtooth', 0.4, 0.1)
  }, [soundEnabled, playTone])

  return { playStart, playComplete, playError }
}
