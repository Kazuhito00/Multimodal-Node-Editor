// テキスト表示ウィジェット（読み取り専用、画像プレビューと同サイズ）
export function TextDisplayWidget({ text }: { text: string }) {
  // JSON形式の場合は整形して表示
  const displayText = formatIfJson(text);

  return (
    <div className="node-text-display nodrag">
      {displayText}
    </div>
  );
}

// JSON形式ならインデント整形、それ以外はそのまま返す
function formatIfJson(text: string): string {
  if (!text) return '';
  const trimmed = text.trim();
  if (!trimmed.startsWith('{') && !trimmed.startsWith('[')) {
    return text;
  }
  try {
    const parsed = JSON.parse(trimmed);
    return JSON.stringify(parsed, null, 2);
  } catch {
    return text;
  }
}
