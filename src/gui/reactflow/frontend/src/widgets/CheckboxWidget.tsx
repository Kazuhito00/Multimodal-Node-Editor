import { PropertyDef } from '../types';

interface CheckboxWidgetProps {
  prop: PropertyDef;
  checked: boolean;
  onChange: (checked: boolean) => void;
  disabled?: boolean;
}

// チェックボックスウィジェット
export function CheckboxWidget({ prop, checked, onChange, disabled = false }: CheckboxWidgetProps) {
  return (
    <div className="property-row nodrag">
      <label className={`property-checkbox-label${disabled ? ' disabled' : ''}`}>
        <input
          type="checkbox"
          className="property-checkbox"
          checked={checked}
          onChange={(e) => onChange(e.target.checked)}
          disabled={disabled}
        />
        <span>{prop.display_name || prop.name}</span>
      </label>
    </div>
  );
}
