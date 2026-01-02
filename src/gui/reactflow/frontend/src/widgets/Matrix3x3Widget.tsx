import { useCallback, useState, useEffect } from 'react';
import { PropertyDef } from '../types';

interface Matrix3x3WidgetProps {
  prop: PropertyDef;
  value: string;
  onChange: (value: string) => void;
}

/**
 * 3x3マトリックス入力ウィジェット
 * 値はカンマ区切り文字列で保存（例: "0,0,0,0,1,0,0,0,0"）
 */
export function Matrix3x3Widget({ prop, value, onChange }: Matrix3x3WidgetProps) {
  // 文字列を9要素の配列にパース
  const parseValue = useCallback((str: string): string[] => {
    const parts = str.split(',').map((v) => v.trim());
    const result: string[] = [];
    for (let i = 0; i < 9; i++) {
      result.push(parts[i] ?? '0');
    }
    return result;
  }, []);

  const [cells, setCells] = useState<string[]>(parseValue(value));

  // 外部からのvalue変更時に更新
  useEffect(() => {
    setCells(parseValue(value));
  }, [value, parseValue]);

  // セルの値を更新
  const handleCellChange = useCallback(
    (index: number, newValue: string) => {
      const newCells = [...cells];
      newCells[index] = newValue;
      setCells(newCells);
    },
    [cells]
  );

  // フォーカスを失った時に値を確定
  const handleBlur = useCallback(
    (index: number) => {
      const newCells = [...cells];
      // 空文字や無効な値は0に
      const numValue = parseFloat(newCells[index]);
      if (isNaN(numValue)) {
        newCells[index] = '0';
      } else {
        // 小数点以下3桁に丸める
        newCells[index] = String(Math.round(numValue * 1000) / 1000);
      }
      setCells(newCells);
      onChange(newCells.join(','));
    },
    [cells, onChange]
  );

  // Enterキーでフォーカスを外す
  const handleKeyDown = useCallback((e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === 'Enter') {
      (e.target as HTMLInputElement).blur();
    }
  }, []);

  return (
    <div className="matrix3x3-widget nodrag">
      <label className="property-label">{prop.display_name || prop.name}</label>
      <div className="matrix3x3-grid">
        {[0, 1, 2].map((row) => (
          <div key={row} className="matrix3x3-row">
            {[0, 1, 2].map((col) => {
              const index = row * 3 + col;
              return (
                <input
                  key={index}
                  type="text"
                  className="matrix3x3-cell"
                  value={cells[index]}
                  onChange={(e) => handleCellChange(index, e.target.value)}
                  onBlur={() => handleBlur(index)}
                  onKeyDown={handleKeyDown}
                />
              );
            })}
          </div>
        ))}
      </div>
    </div>
  );
}
