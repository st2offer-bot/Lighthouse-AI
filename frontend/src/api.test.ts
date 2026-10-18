import { describe, expect, it } from 'vitest'
import { formatTime, normalizeBaseUrl, safeEvidenceSrc } from './api'

describe('frontend API helpers', () => {
  it('formats seconds and hours', () => {
    expect(formatTime(44.2)).toBe('00:44')
    expect(formatTime(125)).toBe('02:05')
    expect(formatTime(3723)).toBe('1:02:03')
    expect(formatTime(-1)).toBe('00:00')
  })
  it('normalizes backend base URL', () => {
    expect(normalizeBaseUrl(' http://127.0.0.1:8000/// ')).toBe('http://127.0.0.1:8000')
  })
  it('accepts only bare base64 for evidence', () => {
    expect(safeEvidenceSrc('/9j/4AA=')).toBe('data:image/jpeg;base64,/9j/4AA=')
    expect(safeEvidenceSrc('javascript:alert(1)')).toBe(null)
  })
})
