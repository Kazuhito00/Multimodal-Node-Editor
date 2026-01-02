import { createContext, useContext, useState, ReactNode, useCallback } from 'react';

// ドラッグ状態のコンテキスト
// ノードドラッグ中はBrowserWebcamのフレームキャプチャを一時停止するために使用
interface DraggingContextType {
  isDragging: boolean;
  setDragging: (dragging: boolean) => void;
}

const DraggingContext = createContext<DraggingContextType>({
  isDragging: false,
  setDragging: () => {},
});

export function DraggingProvider({ children }: { children: ReactNode }) {
  const [isDragging, setIsDragging] = useState(false);

  const setDragging = useCallback((dragging: boolean) => {
    setIsDragging(dragging);
  }, []);

  return (
    <DraggingContext.Provider value={{ isDragging, setDragging }}>
      {children}
    </DraggingContext.Provider>
  );
}

export function useDragging() {
  return useContext(DraggingContext);
}
