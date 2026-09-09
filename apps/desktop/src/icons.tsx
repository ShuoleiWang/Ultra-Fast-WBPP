import type { SVGProps } from "react";

type IconProps = SVGProps<SVGSVGElement>;

const base = {
  width: 20,
  height: 20,
  viewBox: "0 0 24 24",
  fill: "none",
  stroke: "currentColor",
  strokeWidth: 1.8,
  strokeLinecap: "round" as const,
  strokeLinejoin: "round" as const,
  "aria-hidden": true,
};

export const SparkIcon = (props: IconProps) => <svg {...base} {...props}><path d="M12 2l1.2 5.1L18 9l-4.8 1.9L12 16l-1.2-5.1L6 9l4.8-1.9L12 2Z"/><path d="M19 15l.6 2.4L22 18l-2.4.6L19 21l-.6-2.4L16 18l2.4-.6L19 15Z"/></svg>;
export const FolderIcon = (props: IconProps) => <svg {...base} {...props}><path d="M3 6.5a2 2 0 0 1 2-2h5l2 2h7a2 2 0 0 1 2 2v8.5a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V6.5Z"/><path d="M3 9h18"/></svg>;
export const CheckIcon = (props: IconProps) => <svg {...base} {...props}><path d="m5 12 4.2 4L19 6.5"/></svg>;
export const ChevronIcon = (props: IconProps) => <svg {...base} {...props}><path d="m9 18 6-6-6-6"/></svg>;
export const CpuIcon = (props: IconProps) => <svg {...base} {...props}><rect x="6" y="6" width="12" height="12" rx="2"/><path d="M9 1v3m6-3v3M9 20v3m6-3v3M1 9h3m-3 6h3m16-6h3m-3 6h3"/><rect x="9" y="9" width="6" height="6" rx="1"/></svg>;
export const PlayIcon = (props: IconProps) => <svg {...base} {...props}><path d="m8 5 11 7-11 7V5Z"/></svg>;
export const StopIcon = (props: IconProps) => <svg {...base} {...props}><rect x="6" y="6" width="12" height="12" rx="1"/></svg>;
export const FileIcon = (props: IconProps) => <svg {...base} {...props}><path d="M6 2h8l4 4v16H6V2Z"/><path d="M14 2v5h5M9 13h6m-6 4h4"/></svg>;
export const TelescopeIcon = (props: IconProps) => <svg {...base} {...props}><path d="m5 8 10-5 3 6-10 5-3-6Z"/><path d="m11 12 2 9m-2-5-5 5m7-5 5 5M3 7l3 6"/></svg>;
