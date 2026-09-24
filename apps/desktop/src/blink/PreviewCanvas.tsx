import { useEffect, useLayoutEffect, useRef, useState } from "react";

/** Paint decoded pixels into the viewport instead of promoting a transformed
 * IMG layer. Native WebKit can leave that layer blank despite successful decode.
 * The hidden image is only a decoder; review readiness follows a canvas paint. */
export function PreviewCanvas({
  src,
  label,
  width,
  height,
  imageWidth,
  imageHeight,
  view,
  onPaint,
  onError,
  className = "",
  decoderId,
  pixelated = false,
}: {
  src: string;
  label: string;
  width: number;
  height: number;
  imageWidth?: number;
  imageHeight?: number;
  view?: { scale: number; x: number; y: number };
  onPaint?: () => void;
  onError: () => void;
  className?: string;
  decoderId?: string;
  pixelated?: boolean;
}) {
  const canvas = useRef<HTMLCanvasElement>(null);
  const decoder = useRef<HTMLImageElement>(null);
  const [decoded, setDecoded] = useState(0);
  const callbacks = useRef({ onPaint, onError });
  callbacks.current = { onPaint, onError };
  const tracksReview = Boolean(onPaint);
  useEffect(() => {
    if (!tracksReview) return;
    const visible = () => {
      if (!document.hidden) setDecoded((value) => value + 1);
    };
    document.addEventListener("visibilitychange", visible);
    return () => document.removeEventListener("visibilitychange", visible);
  }, [tracksReview]);
  useLayoutEffect(() => {
    const image = decoder.current;
    const surface = canvas.current;
    if (!image?.complete || !image.naturalWidth || !surface) return;
    const context = surface.getContext("2d");
    if (!context) {
      callbacks.current.onError();
      return;
    }
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    surface.width = Math.max(1, Math.round(width * dpr));
    surface.height = Math.max(1, Math.round(height * dpr));
    context.imageSmoothingEnabled = !pixelated;
    context.setTransform(dpr, 0, 0, dpr, 0, 0);
    context.fillStyle = "#05060a";
    context.fillRect(0, 0, width, height);
    const iw = imageWidth || image.naturalWidth,
      ih = imageHeight || image.naturalHeight;
    const scale = Math.min(width / iw, height / ih);
    const pose = view ?? { scale, x: (width - iw * scale) / 2, y: (height - ih * scale) / 2 };
    context.translate(pose.x, pose.y);
    context.scale(pose.scale, pose.scale);
    try {
      context.drawImage(image, 0, 0, iw, ih);
    } catch {
      callbacks.current.onError();
      return;
    }
    const token = window.requestAnimationFrame(() => {
      if (!document.hidden) callbacks.current.onPaint?.();
    });
    return () => window.cancelAnimationFrame(token);
  }, [src, decoded, width, height, imageWidth, imageHeight, view?.scale, view?.x, view?.y, pixelated]);
  return (
    <>
      <canvas
        ref={canvas}
        role="img"
        aria-label={label || undefined}
        aria-hidden={label ? undefined : true}
        className={className}
      />
      <img
        ref={decoder}
        src={src}
        alt=""
        hidden
        data-testid={decoderId}
        onLoad={() => setDecoded((value) => value + 1)}
        onError={() => callbacks.current.onError()}
      />
    </>
  );
}
