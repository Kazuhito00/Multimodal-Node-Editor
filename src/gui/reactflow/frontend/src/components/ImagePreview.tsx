import { memo, useCallback, useRef, useEffect } from 'react';

// 常時プレビュー表示するノードのdefinitionIdリスト
// 注意: App.tsx の alwaysPreviewNodeIds と同期すること
const ALWAYS_PREVIEW_NODES = [
  'image.input.webcam',
  'image.input.browser_webcam',
  'image.input.image',
  'image.input.video',
  'image.input.video_frame',
  'image.input.rtsp',
  'image.input.url_image',
];

interface ImagePreviewProps {
  definitionId: string;
  imageData?: string;
  isStreaming: boolean;
  isPaused: boolean;
  isClickable?: boolean;
  isDraggable?: boolean;
  waveformStyle?: boolean;
  onClick?: (e: React.MouseEvent<HTMLImageElement>) => void;
  onDrag?: (normalizedX: number) => void;
}

// 画像プレビューコンポーネント
// 条件に応じて画像/Loading/プレースホルダーを表示
function ImagePreviewComponent({
  definitionId,
  imageData,
  isStreaming,
  isPaused,
  isClickable = false,
  isDraggable = false,
  waveformStyle = false,
  onClick,
  onDrag,
}: ImagePreviewProps) {
  const isActiveOrPaused = isStreaming || isPaused;
  const isAlwaysPreview = ALWAYS_PREVIEW_NODES.includes(definitionId);
  const isDraggingRef = useRef(false);
  const imgRef = useRef<HTMLImageElement>(null);

  // プレビュー表示条件: 常時プレビューノードはimageDataがあれば表示、それ以外はアクティブ時のみ
  const shouldShowPreview = isAlwaysPreview ? !!imageData : isActiveOrPaused && !!imageData;

  // Loading表示条件: Webcamは常時、RTSPはストリーミング中のみ
  const isWebcam = definitionId === 'image.input.webcam';
  const isRtsp = definitionId === 'image.input.rtsp';
  const showLoading = !imageData && (isWebcam || (isRtsp && isStreaming));

  // ドラッグ開始
  const handleMouseDown = useCallback(
    (e: React.MouseEvent<HTMLImageElement>) => {
      if (!isDraggable) return;
      e.stopPropagation();
      e.preventDefault();
      isDraggingRef.current = true;

      // 初期位置を計算
      const rect = e.currentTarget.getBoundingClientRect();
      const x = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width));
      onDrag?.(x);
    },
    [isDraggable, onDrag]
  );

  // ドキュメントレベルのマウスイベント
  useEffect(() => {
    if (!isDraggable) return;

    const handleDocumentMouseMove = (e: MouseEvent) => {
      if (!isDraggingRef.current || !imgRef.current) return;
      e.stopPropagation();
      e.preventDefault();

      const rect = imgRef.current.getBoundingClientRect();
      const x = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width));
      onDrag?.(x);
    };

    const handleDocumentMouseUp = () => {
      isDraggingRef.current = false;
    };

    document.addEventListener('mousemove', handleDocumentMouseMove);
    document.addEventListener('mouseup', handleDocumentMouseUp);

    return () => {
      document.removeEventListener('mousemove', handleDocumentMouseMove);
      document.removeEventListener('mouseup', handleDocumentMouseUp);
    };
  }, [isDraggable, onDrag]);

  if (shouldShowPreview) {
    const classNames = ['node-image'];
    if (isClickable || isDraggable) classNames.push('clickable', 'nodrag');
    if (waveformStyle) classNames.push('waveform-style');

    return (
      <img
        ref={imgRef}
        src={`data:image/jpeg;base64,${imageData}`}
        alt="preview"
        className={classNames.join(' ')}
        onClick={isClickable ? onClick : undefined}
        onMouseDown={isDraggable ? handleMouseDown : undefined}
      />
    );
  }

  const placeholderClass = waveformStyle
    ? 'node-image-placeholder waveform-style'
    : 'node-image-placeholder';

  if (showLoading) {
    return <div className={`${placeholderClass} node-image-loading`}>Loading...</div>;
  }

  return <div className={placeholderClass} />;
}

export const ImagePreview = memo(ImagePreviewComponent);
