import { useCallback, useEffect, useRef, useState } from 'react';
import { PropertyDef } from '../types';

// 数値入力ウィジェットコンポーネント（+-ボタン付き）
export function NumberInputWidget({
  prop,
  value,
  onChange,
  disabled,
}: {
  prop: PropertyDef;
  value: number;
  onChange: (value: number) => void;
  disabled?: boolean;
}) {
  const min = prop.min ?? -999999;
  const max = prop.max ?? 999999;
  const isFloat = prop.type === 'float';
  const step = prop.step ?? (isFloat ? 0.1 : 1);
  const intervalRef = useRef<number | null>(null);
  const timeoutRef = useRef<number | null>(null);
  const valueRef = useRef(value);

  // 入力中のテキストを保持するローカルステート
  const [localValue, setLocalValue] = useState(String(value));
  const [isFocused, setIsFocused] = useState(false);

  // 小数点以下の桁数を計算（stepに基づく）
  const decimals = isFloat ? Math.max(0, -Math.floor(Math.log10(step))) : 0;

  // 値を丸める
  const roundValue = useCallback((val: number) => {
    if (isFloat) {
      return Math.round(val * Math.pow(10, decimals)) / Math.pow(10, decimals);
    }
    return Math.round(val);
  }, [isFloat, decimals]);

  useEffect(() => {
    valueRef.current = value;
  }, [value]);

  // 外部からのvalue変更時にローカルステートを更新（フォーカスがない時のみ）
  useEffect(() => {
    if (!isFocused) {
      setLocalValue(String(value));
    }
  }, [value, isFocused]);

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const rawValue = e.target.value;
    // 入力中はローカルステートのみ更新（途中入力を許可）
    setLocalValue(rawValue);
  };

  const handleBlur = () => {
    setIsFocused(false);
    // フォーカスを失った時に値を確定
    if (localValue === '' || localValue === '-' || localValue === '.') {
      setLocalValue(String(min));
      onChange(min);
      return;
    }
    const numValue = parseFloat(localValue);
    if (!isNaN(numValue)) {
      const clampedValue = Math.max(min, Math.min(max, numValue));
      const finalValue = isFloat ? clampedValue : Math.round(clampedValue);
      setLocalValue(String(finalValue));
      onChange(finalValue);
    } else {
      // 無効な入力の場合は元の値に戻す
      setLocalValue(String(value));
    }
  };

  const handleFocus = () => {
    setIsFocused(true);
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === 'Enter') {
      (e.target as HTMLInputElement).blur();
    }
  };

  const startContinuous = useCallback((direction: 'up' | 'down') => {
    const doChange = () => {
      const current = valueRef.current;
      let newVal: number;
      if (direction === 'up') {
        newVal = roundValue(Math.min(max, current + step));
      } else {
        newVal = roundValue(Math.max(min, current - step));
      }
      onChange(newVal);
      setLocalValue(String(newVal));
    };

    doChange();
    timeoutRef.current = window.setTimeout(() => {
      intervalRef.current = window.setInterval(doChange, 80);
    }, 300);
  }, [step, min, max, onChange, roundValue]);

  const stopContinuous = useCallback(() => {
    if (timeoutRef.current) {
      clearTimeout(timeoutRef.current);
      timeoutRef.current = null;
    }
    if (intervalRef.current) {
      clearInterval(intervalRef.current);
      intervalRef.current = null;
    }
  }, []);

  useEffect(() => {
    return () => stopContinuous();
  }, [stopContinuous]);

  return (
    <div className="property-row nodrag">
      <label className="property-label">{prop.display_name || prop.name}</label>
      <div className="number-input-widget">
        <button
          className="number-input-btn"
          onMouseDown={() => startContinuous('down')}
          onMouseUp={stopContinuous}
          onMouseLeave={stopContinuous}
          disabled={disabled || value <= min}
        >
          -
        </button>
        <input
          type="text"
          className="property-number-input"
          value={localValue}
          onChange={handleChange}
          onBlur={handleBlur}
          onFocus={handleFocus}
          onKeyDown={handleKeyDown}
          disabled={disabled}
        />
        <button
          className="number-input-btn"
          onMouseDown={() => startContinuous('up')}
          onMouseUp={stopContinuous}
          onMouseLeave={stopContinuous}
          disabled={disabled || value >= max}
        >
          +
        </button>
      </div>
    </div>
  );
}
