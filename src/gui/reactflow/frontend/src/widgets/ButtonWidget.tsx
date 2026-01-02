import { PropertyDef } from '../types';

interface ButtonWidgetProps {
  prop: PropertyDef;
  onClick: () => void;
  disabled?: boolean;
}

// ボタンウィジェット（クリックでプロパティをtrueに設定し、即座にfalseに戻す）
export function ButtonWidget({ prop, onClick, disabled = false }: ButtonWidgetProps) {
  const label = prop.button_label || prop.display_name || prop.name || 'Button';

  return (
    <div className="property-row nodrag">
      <button
        className="property-button"
        onClick={onClick}
        disabled={disabled}
      >
        {label}
      </button>
    </div>
  );
}
