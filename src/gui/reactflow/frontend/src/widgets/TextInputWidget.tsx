import { useEffect, useState } from 'react';
import { PropertyDef } from '../types';

// テキスト入力ウィジェットコンポーネント
export function TextInputWidget({
  prop,
  value,
  onChange,
  disabled,
}: {
  prop: PropertyDef;
  value: string;
  onChange: (value: string) => void;
  disabled?: boolean;
}) {
  const [localValue, setLocalValue] = useState(value);

  useEffect(() => {
    setLocalValue(value);
  }, [value]);

  const handleBlur = () => {
    if (localValue !== value) {
      onChange(localValue);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter') {
      onChange(localValue);
    }
  };

  return (
    <div className="property-row nodrag">
      <label className="property-label">{prop.display_name || prop.name}</label>
      <input
        type="text"
        className="property-text-input"
        value={localValue}
        onChange={(e) => setLocalValue(e.target.value)}
        onBlur={handleBlur}
        onKeyDown={handleKeyDown}
        placeholder={prop.placeholder || ''}
        disabled={disabled}
      />
    </div>
  );
}
