import { useEffect, useState } from 'react';
import { createPortal } from 'react-dom';

// アップロードモーダルのプロパティ
interface UploadModalProps {
  fileName: string;
  progress: number;
  onCancel: () => void;
}

// 現在のテーマを取得
function getCurrentTheme(): 'dark' | 'light' {
  const appContainer = document.querySelector('.app-container');
  if (appContainer?.classList.contains('light')) {
    return 'light';
  }
  return 'dark';
}

// アップロード進捗モーダルコンポーネント
export function UploadModal({ fileName, progress, onCancel }: UploadModalProps) {
  const [theme, setTheme] = useState<'dark' | 'light'>(getCurrentTheme);

  // テーマの変更を監視
  useEffect(() => {
    const appContainer = document.querySelector('.app-container');
    if (!appContainer) return;

    const observer = new MutationObserver(() => {
      setTheme(getCurrentTheme());
    });

    observer.observe(appContainer, { attributes: true, attributeFilter: ['class'] });

    return () => observer.disconnect();
  }, []);

  // progress が -1 の場合は準備中（インデターミネート）
  const isPreparing = progress < 0;

  return createPortal(
    <div className={`upload-modal-overlay ${theme}`}>
      <div className="upload-modal">
        <div className="upload-modal-header">
          <span className="upload-modal-title">
            {isPreparing ? 'Preparing...' : 'Uploading...'}
          </span>
          <button
            className="upload-modal-close"
            onClick={onCancel}
            title="Cancel upload"
          >
            ✕
          </button>
        </div>
        <div className="upload-modal-content">
          <div className="upload-modal-filename" title={fileName}>
            {fileName}
          </div>
          <div className="upload-modal-progress">
            {isPreparing ? (
              <div className="upload-modal-progress-indeterminate" />
            ) : (
              <div
                className="upload-modal-progress-bar"
                style={{ width: `${progress}%` }}
              />
            )}
          </div>
          <div className="upload-modal-percent">
            {isPreparing ? '...' : `${progress}%`}
          </div>
        </div>
      </div>
    </div>,
    document.body
  );
}
