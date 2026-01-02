import { useCallback, useRef, useState } from 'react';
import { PropertyDef } from '../types';
import { UploadModal } from '../components/UploadModal';

// アップロードレスポンスの型定義
interface UploadResponse {
  path: string;
  first_frame?: string;
  frame_count?: number;
}

// ファイルピッカーウィジェットコンポーネント
export function FilePickerWidget({
  prop,
  value,
  onChange,
  onFirstFrame,
  onFrameCount,
  disabled,
}: {
  prop: PropertyDef;
  value: string;
  onChange: (value: string) => void;
  onFirstFrame?: (imageData: string) => void;
  onFrameCount?: (frameCount: number) => void;
  disabled?: boolean;
}) {
  const inputRef = useRef<HTMLInputElement>(null);
  const xhrRef = useRef<XMLHttpRequest | null>(null);
  const [isUploading, setIsUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [uploadFileName, setUploadFileName] = useState('');

  // アップロードをキャンセル
  const cancelUpload = useCallback(() => {
    if (xhrRef.current) {
      xhrRef.current.abort();
      xhrRef.current = null;
    }
    setIsUploading(false);
    setUploadProgress(0);
    setUploadFileName('');
    if (inputRef.current) {
      inputRef.current.value = '';
    }
  }, []);

  // ファイルをアップロード（XMLHttpRequestで進捗追跡）
  const uploadFile = useCallback((file: File): Promise<UploadResponse> => {
    return new Promise((resolve, reject) => {
      const xhr = new XMLHttpRequest();
      xhrRef.current = xhr;

      // 進捗イベント
      xhr.upload.onprogress = (e) => {
        if (e.lengthComputable) {
          const percent = Math.round((e.loaded / e.total) * 100);
          setUploadProgress(percent);
        }
      };

      // 完了イベント
      xhr.onload = () => {
        xhrRef.current = null;
        if (xhr.status >= 200 && xhr.status < 300) {
          try {
            const data = JSON.parse(xhr.responseText);
            resolve(data);
          } catch {
            reject(new Error('Invalid response format'));
          }
        } else {
          reject(new Error(`Upload failed: ${xhr.status}`));
        }
      };

      // エラーイベント
      xhr.onerror = () => {
        xhrRef.current = null;
        reject(new Error('Network error'));
      };

      // キャンセルイベント
      xhr.onabort = () => {
        xhrRef.current = null;
        reject(new Error('Upload cancelled'));
      };

      // タイムアウト設定（10分）
      xhr.timeout = 600000;
      xhr.ontimeout = () => {
        xhrRef.current = null;
        reject(new Error('Upload timeout'));
      };

      // リクエスト送信
      const formData = new FormData();
      formData.append('file', file);

      xhr.open('POST', '/api/upload');
      xhr.send(formData);
    });
  }, []);

  // ファイル選択ハンドラ
  const handleFileSelect = useCallback(async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) {
      console.log('No file selected');
      return;
    }

    // ファイルアップロード時にSTOPイベントを発火
    window.dispatchEvent(new CustomEvent('node-file-upload'));

    console.log('Uploading file:', file.name, file.size, 'bytes');
    setIsUploading(true);
    setUploadProgress(-1);  // -1 = 準備中（インデターミネート表示）
    setUploadFileName(file.name);

    try {
      const data = await uploadFile(file);
      console.log('Upload success, path:', data.path);
      onChange(data.path);

      // 動画の最初のフレームがあればプレビューに設定
      if (data.first_frame && onFirstFrame) {
        onFirstFrame(data.first_frame);
      }

      // 動画のフレーム数があればシークバーを更新
      if (data.frame_count && onFrameCount) {
        onFrameCount(data.frame_count);
      }
    } catch (error) {
      if (error instanceof Error) {
        if (error.message === 'Upload cancelled') {
          console.log('Upload cancelled by user');
        } else {
          console.error('Error uploading file:', error.message);
        }
      }
    } finally {
      setIsUploading(false);
      setUploadProgress(0);
      setUploadFileName('');
      // inputをリセット（同じファイルを再選択できるようにする）
      if (inputRef.current) {
        inputRef.current.value = '';
      }
    }
  }, [onChange, onFirstFrame, onFrameCount, uploadFile]);

  const fileName = value ? value.split(/[/\\]/).pop() : '';

  return (
    <div className="property-row nodrag">
      <label className="property-label">{prop.display_name || prop.name}</label>
      <div className="file-picker-widget">
        <input
          ref={inputRef}
          type="file"
          accept={prop.accept}
          onChange={handleFileSelect}
          style={{ display: 'none' }}
          disabled={disabled || isUploading}
        />
        <button
          className="file-picker-btn"
          onClick={() => inputRef.current?.click()}
          disabled={disabled || isUploading}
        >
          Open
        </button>
        <span className="file-picker-name" title={value}>
          {fileName || 'No file'}
        </span>
      </div>

      {/* アップロード中はモーダルを表示 */}
      {isUploading && (
        <UploadModal
          fileName={uploadFileName}
          progress={uploadProgress}
          onCancel={cancelUpload}
        />
      )}
    </div>
  );
}
