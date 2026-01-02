import { useEffect, useState } from 'react';
import { PropertyDef } from '../types';

// テキストエリアウィジェット（画像プレビューと同サイズ）
export function TextAreaWidget({
  prop,
  value,
  onChange,
  disabled,
  readOnly,
}: {
  prop?: PropertyDef;
  value: string;
  onChange: (value: string) => void;
  disabled?: boolean;
  readOnly?: boolean;  // 接続時は読み取り専用（コピー可能）
}) {
  const [localValue, setLocalValue] = useState(value);

  // 接続値が変わった場合はローカル値も更新
  useEffect(() => {
    setLocalValue(value);
  }, [value]);

  const handleBlur = () => {
    if (localValue !== value && !readOnly) {
      onChange(localValue);
    }
  };

  // display_nameが空文字の場合はラベルを非表示（nullish coalescingで明示的なundefined/nullのみフォールバック）
  const label = prop?.display_name ?? prop?.name;

  // rows指定がある場合は高さを計算（1行約20px + パディング）
  const style = prop?.rows ? { height: `${prop.rows * 20 + 16}px` } : undefined;

  return (
    <div className="text-area-container">
      {label && <label className="text-area-label">{label}</label>}
      <textarea
        className={`node-textarea nodrag ${readOnly ? 'connected' : ''}`}
        style={style}
        value={localValue}
        onChange={(e) => !readOnly && setLocalValue(e.target.value)}
        onBlur={handleBlur}
        placeholder="Enter text..."
        disabled={disabled}
        readOnly={readOnly}
      />
    </div>
  );
}
