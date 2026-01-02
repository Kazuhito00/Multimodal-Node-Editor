import { PropertyDef } from '../types';

interface SeekbarWidgetProps {
  prop: PropertyDef;
  value: number;
  onChange: (value: number) => void;
}

// シークバーウィジェット（ラベルなしスライダー、動画フレーム選択用）
export function SeekbarWidget({ prop, value, onChange }: SeekbarWidgetProps) {
  const min = prop.min ?? 1;
  const max = prop.max ?? 1;

  return (
    <div className="seekbar-row nodrag">
      <input
        type="range"
        className="seekbar"
        min={min}
        max={max}
        value={value}
        onChange={(e) => onChange(Number(e.target.value))}
      />
      <span className="seekbar-value">
        {value} / {max}
      </span>
    </div>
  );
}
