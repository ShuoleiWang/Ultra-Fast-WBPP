import type { SVGProps } from "react";

// Line symbols drawn on a 16 pt grid with a 1.5 px stroke, after SF Symbols.
type IconProps = SVGProps<SVGSVGElement>;
const base: IconProps = {
  width: 16,
  height: 16,
  viewBox: "0 0 16 16",
  fill: "none",
  stroke: "currentColor",
  strokeWidth: 1.5,
  strokeLinecap: "round",
  strokeLinejoin: "round",
  "aria-hidden": true,
};

export const SparkIcon = (props: IconProps) => (
  <svg {...base} {...props}>
    <path d="M8 1.8l1.7 3.6 3.9.5-2.9 2.7.8 3.9L8 10.6l-3.5 1.9.8-3.9-2.9-2.7 3.9-.5z" />
  </svg>
);
export const FolderIcon = (props: IconProps) => (
  <svg {...base} {...props}>
    <path d="M2 4.5A1.5 1.5 0 0 1 3.5 3h3l1.5 1.5h4.5A1.5 1.5 0 0 1 14 6v5.5a1.5 1.5 0 0 1-1.5 1.5h-9A1.5 1.5 0 0 1 2 11.5v-7Z" />
  </svg>
);
export const FilesIcon = (props: IconProps) => (
  <svg {...base} {...props}>
    <path d="M4.5 2.5h5l3 3v8h-8v-11Z" />
    <path d="M9.5 2.5v3h3" />
  </svg>
);
export const ImportIcon = (props: IconProps) => (
  <svg {...base} {...props}>
    <path d="M8 2.5v7.5m0 0 2.8-2.8M8 10 5.2 7.2" />
    <path d="M2.5 10.5v2A1.5 1.5 0 0 0 4 14h8a1.5 1.5 0 0 0 1.5-1.5v-2" />
  </svg>
);
export const LightIcon = SparkIcon;
export const FlatIcon = (props: IconProps) => (
  <svg {...base} {...props}>
    <rect x="2.2" y="3.4" width="11.6" height="9.2" rx="1.6" />
    <path d="M2.2 9.6h11.6" />
  </svg>
);
export const DarkIcon = (props: IconProps) => (
  <svg {...base} {...props}>
    <circle cx="8" cy="8" r="5.4" />
    <path d="M8 2.6v10.8" />
  </svg>
);
export const BiasIcon = (props: IconProps) => (
  <svg {...base} {...props}>
    <rect x="2.4" y="5.2" width="11.2" height="5.6" rx="1.4" />
  </svg>
);
export const ProcessIcon = (props: IconProps) => (
  <svg {...base} {...props}>
    <circle cx="8" cy="8" r="5.6" />
    <path d="M6.8 5.6v4.8L10.4 8 6.8 5.6Z" fill="currentColor" stroke="none" />
  </svg>
);
export const ResultIcon = (props: IconProps) => (
  <svg {...base} {...props}>
    <rect x="2" y="3" width="12" height="10" rx="1.6" />
    <path d="m2.6 11.4 3.2-3.6 2.4 2.6 1.9-2.2 3.3 3.2" />
    <circle cx="5.4" cy="6" r="1" />
  </svg>
);
export const InspectorIcon = (props: IconProps) => (
  <svg {...base} {...props} strokeWidth={1.3}>
    <rect x="1.6" y="2.6" width="12.8" height="10.8" rx="2" />
    <path d="M10.2 2.6v10.8" />
  </svg>
);
export const SearchIcon = (props: IconProps) => (
  <svg {...base} {...props}>
    <circle cx="7" cy="7" r="4.2" />
    <path d="M10.2 10.2 13.5 13.5" />
  </svg>
);
export const CheckIcon = (props: IconProps) => (
  <svg {...base} {...props} strokeWidth={1.8}>
    <path d="M3.2 8.4l3.1 3.1 6.5-6.9" />
  </svg>
);
export const WarnIcon = (props: IconProps) => (
  <svg {...base} {...props}>
    <path d="M8 2.6 14 13H2L8 2.6Z" />
    <path d="M8 6.5v3.2M8 11.6v.2" />
  </svg>
);
export const StopIcon = (props: IconProps) => (
  <svg {...base} {...props}>
    <rect x="3" y="3" width="10" height="10" rx="1.5" />
  </svg>
);
export const XIcon = (props: IconProps) => (
  <svg {...base} {...props}>
    <path d="m4 4 8 8M12 4l-8 8" />
  </svg>
);
export const ChevronIcon = (props: IconProps) => (
  <svg {...base} {...props}>
    <path d="m6 3.5 4.5 4.5L6 12.5" />
  </svg>
);
export const CpuIcon = (props: IconProps) => (
  <svg {...base} {...props}>
    <rect x="4" y="4" width="8" height="8" rx="1.5" />
    <path d="M6 1.5v2.5m4-2.5v2.5M6 12v2.5m4-2.5v2.5M1.5 6H4m-2.5 4H4m8-4h2.5M12 10h2.5" />
    <rect x="6.5" y="6.5" width="3" height="3" rx=".6" />
  </svg>
);
export const PlayIcon = (props: IconProps) => (
  <svg {...base} {...props}>
    <path d="M5 3.5v9l7.5-4.5L5 3.5Z" fill="currentColor" stroke="none" />
  </svg>
);
export const FileIcon = FilesIcon;
export const RevealIcon = (props: IconProps) => (
  <svg {...base} {...props}>
    <path d="M2 4.5A1.5 1.5 0 0 1 3.5 3h3l1.5 1.5h4.5A1.5 1.5 0 0 1 14 6v5.5a1.5 1.5 0 0 1-1.5 1.5h-9A1.5 1.5 0 0 1 2 11.5v-7Z" />
    <path d="M6 9.5h4m0 0L8.6 8.1M10 9.5l-1.4 1.4" />
  </svg>
);
export const ExportIcon = (props: IconProps) => (
  <svg {...base} {...props}>
    <path d="M8 10V2.5m0 0L5.2 5.3M8 2.5l2.8 2.8" />
    <path d="M2.5 9.5v2.5A1.5 1.5 0 0 0 4 13.5h8a1.5 1.5 0 0 0 1.5-1.5V9.5" />
  </svg>
);
export const TelescopeIcon = (props: IconProps) => (
  <svg {...base} {...props}>
    <path d="m3.5 6 7-3.5 2 4-7 3.5-2-4Z" />
    <path d="m7.5 9 1.3 5m-1.3-3.3L4.5 14m4.5-3.3L12 14" />
  </svg>
);
export const ClockIcon = (props: IconProps) => (
  <svg {...base} {...props}>
    <circle cx="8" cy="8" r="5.6" />
    <path d="M8 4.8V8l2.2 1.4" />
  </svg>
);

/** The application mark: three light frames converging on one verified star (also the app icon). */
export const BrandMark = ({ size = 20, ...props }: IconProps & { size?: number }) => (
  <svg width={size} height={size} viewBox="0 0 64 64" aria-hidden="true" {...props}>
    <defs>
      <linearGradient id="ufwbpp-bg" x1="0" y1="0" x2="1" y2="1">
        <stop offset="0" stopColor="#0B1030" />
        <stop offset=".55" stopColor="#1A2C70" />
        <stop offset="1" stopColor="#5A2E96" />
      </linearGradient>
      <radialGradient id="ufwbpp-glow" cx=".7" cy=".4" r=".42">
        <stop offset="0" stopColor="#9CC2FF" stopOpacity=".6" />
        <stop offset="1" stopColor="#9CC2FF" stopOpacity="0" />
      </radialGradient>
    </defs>
    <rect x="2" y="2" width="60" height="60" rx="14" fill="url(#ufwbpp-bg)" />
    <rect x="2" y="2" width="60" height="60" rx="14" fill="url(#ufwbpp-glow)" />
    <rect x="2.5" y="2.5" width="59" height="59" rx="13.5" fill="none" stroke="#fff" strokeOpacity=".18" />
    <g fill="#BED2FF" stroke="#E6F0FF" strokeWidth="1">
      <rect x="11" y="30" width="22" height="18" rx="3" fillOpacity=".14" strokeOpacity=".45" />
      <rect x="16" y="25" width="22" height="18" rx="3" fillOpacity=".22" strokeOpacity=".6" />
      <rect x="21" y="20" width="22" height="18" rx="3" fillOpacity=".34" strokeOpacity=".8" />
    </g>
    <g stroke="#F4F8FF" strokeLinecap="round" fill="none">
      <path d="M45 12v24M33 24h24" strokeWidth="1.6" strokeOpacity=".92" />
      <path d="M39 18l12 12M51 18L39 30" strokeWidth="1" strokeOpacity=".6" />
    </g>
    <circle cx="45" cy="24" r="3.2" fill="#fff" />
    <circle cx="45" cy="24" r="6.5" fill="#fff" fillOpacity=".22" />
  </svg>
);
