import { PropertyDef } from '../types';

interface ColorPickerWidgetProps {
  prop: PropertyDef;
  value: string;
  onChange: (value: string) => void;
}

// カラーピッカーウィジェット
export function ColorPickerWidget({ prop, value, onChange }: ColorPickerWidgetProps) {
  return (
    <div className="property-row nodrag">
      <label className="property-label">{prop.display_name || prop.name}</label>
      <input
        type="color"
        className="property-color-picker"
        value={value}
        onChange={(e) => onChange(e.target.value)}
      />
    </div>
  );
}
