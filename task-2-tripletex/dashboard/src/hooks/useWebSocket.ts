'use client'

import { useEffect, useRef, useCallback } from 'react'
import { useStore } from '@/lib/store'
import { WsMessage } from '@/lib/types'
import { api } from '@/lib/api'

export function useWebSocket() {
  const { setRuns, setSettings, setConnectionStatus, connectionStatus } = useStore()
  const wsRef = useRef<WebSocket | null>(null)
  const reconnectTimeoutRef = useRef<NodeJS.Timeout | null>(null)
  const pingIntervalRef = useRef<NodeJS.Timeout | null>(null)
  const pollingIntervalRef = useRef<NodeJS.Timeout | null>(null)
  const enrichIntervalRef = useRef<NodeJS.Timeout | null>(null)
  const reconnectAttemptsRef = useRef(0)
  const prevRunStatusesRef = useRef<Map<string, string>>(new Map())

  const connect = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN || wsRef.current?.readyState === WebSocket.CONNECTING) {
      return
    }

    setConnectionStatus(reconnectAttemptsRef.current > 0 ? 'reconnecting' : 'connecting')

    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
    const host = window.location.host
    const wsUrl = `${protocol}//${host}/dashboard/ws`

    const ws = new WebSocket(wsUrl)
    wsRef.current = ws

    ws.onopen = () => {
      setConnectionStatus('live')
      reconnectAttemptsRef.current = 0

      if (pollingIntervalRef.current) {
        clearInterval(pollingIntervalRef.current)
        pollingIntervalRef.current = null
      }

      if (pingIntervalRef.current) clearInterval(pingIntervalRef.current)
      pingIntervalRef.current = setInterval(() => {
        if (ws.readyState === WebSocket.OPEN) {
          ws.send('ping')
        }
      }, 25000)
    }

    ws.onmessage = (event) => {
      try {
        const message = JSON.parse(event.data) as WsMessage

        switch (message.type) {
          case 'snapshot': {
            const newRuns = message.payload.runs
            const prev = prevRunStatusesRef.current

            for (const run of newRuns) {
              const prevStatus = prev.get(run.run_id)
              if (prevStatus === 'running' && (run.status === 'completed' || run.status === 'error')) {
                setTimeout(() => {
                  api.enrichRuns().catch(console.error)
                }, 5000)
                break
              }
            }

            prevRunStatusesRef.current = new Map(newRuns.map(r => [r.run_id, r.status]))
            setRuns(newRuns, message.payload.active_count, message.payload.total_count)
            break
          }
          case 'settings':
            setSettings(message.payload)
            break
          case 'run_event':
            break
          case 'heartbeat':
          case 'pong':
            break
        }
      } catch (err) {
        console.error('Failed to parse WebSocket message', err)
      }
    }

    ws.onclose = () => {
      handleDisconnect()
    }

    ws.onerror = () => {}
  }, [setRuns, setSettings, setConnectionStatus])

  const handleDisconnect = useCallback(() => {
    if (wsRef.current) {
      wsRef.current.close()
      wsRef.current = null
    }

    if (pingIntervalRef.current) {
      clearInterval(pingIntervalRef.current)
      pingIntervalRef.current = null
    }

    setConnectionStatus('offline')

    if (!pollingIntervalRef.current) {
      pollingIntervalRef.current = setInterval(() => {
        api.getRuns().then(data => {
          if (data && data.runs) {
            setRuns(data.runs, data.active_count, data.total_count)
          }
        }).catch(console.error)
      }, 4000)
    }

    const backoff = Math.min(1000 * Math.pow(2, reconnectAttemptsRef.current), 15000)
    reconnectAttemptsRef.current += 1

    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current)
    }

    reconnectTimeoutRef.current = setTimeout(() => {
      connect()
    }, backoff)
  }, [connect, setConnectionStatus, setRuns])

  useEffect(() => {
    connect()

    enrichIntervalRef.current = setInterval(() => {
      api.enrichRuns().catch(console.error)
    }, 30000)

    setTimeout(() => {
      api.enrichRuns().catch(console.error)
    }, 3000)

    return () => {
      if (wsRef.current) wsRef.current.close()
      if (reconnectTimeoutRef.current) clearTimeout(reconnectTimeoutRef.current)
      if (pingIntervalRef.current) clearInterval(pingIntervalRef.current)
      if (pollingIntervalRef.current) clearInterval(pollingIntervalRef.current)
      if (enrichIntervalRef.current) clearInterval(enrichIntervalRef.current)
    }
  }, [connect])

  return { connectionStatus }
}
