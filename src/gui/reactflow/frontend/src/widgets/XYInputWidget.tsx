interface XYInputWidgetProps {
  xValue: number;
  yValue: number;
  onXChange: (value: number) => void;
  onYChange: (value: number) => void;
}

// XY座標入力ウィジェット（横並び）
export function XYInputWidget({ xValue, yValue, onXChange, onYChange }: XYInputWidgetProps) {
  return (
    <div className="xy-input-row nodrag">
      <label className="xy-label">X</label>
      <input
        type="number"
        className="xy-input"
        value={xValue}
        onChange={(e) => onXChange(parseInt(e.target.value) || 0)}
      />
      <label className="xy-label">Y</label>
      <input
        type="number"
        className="xy-input"
        value={yValue}
        onChange={(e) => onYChange(parseInt(e.target.value) || 0)}
      />
    </div>
  );
}
