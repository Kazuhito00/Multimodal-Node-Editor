// 実行環境の型定義
export type Runtime = { is_colab: boolean };

// 実行環境に応じたAPI URLを生成
export function makeApiBase(isColab: boolean): string {
  return isColab ? '/api' : 'http://localhost:8000/api';
}

// 実行環境に応じたWebSocket URLを生成
export function makeWsUrl(isColab: boolean): string {
  if (!isColab) {
    return 'ws://localhost:8000/api/ws/stream';
  }
  const protocol = location.protocol === 'https:' ? 'wss' : 'ws';
  return `${protocol}://${location.host}/api/ws/stream`;
}

// ユニークID生成
export function generateId(prefix: string): string {
  return `${prefix}-${Math.random().toString(36).substring(2, 10)}`;
}

// localStorageからテーマを取得（初回読み込み用）
export function getInitialTheme(): 'dark' | 'light' {
  try {
    const saved = localStorage.getItem('theme');
    if (saved === 'light' || saved === 'dark') {
      return saved;
    }
  } catch {
    // localStorage使用不可
  }
  return 'light';
}
