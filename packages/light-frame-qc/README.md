# Light Frame QC

独立、保守、可解释的天文 light frame 质量筛选程序。它递归读取多个文件或文件夹，按相机、画幅、binning、滤镜、曝光、gain/offset 和天区分组，然后根据场星配准、相对测光、未截断检出星数、空间掉星、背景与纹理证据识别过云和大面积硬遮挡。

程序只给出建议：不会修改 FITS/XISF header，不会移动、重命名或删除任何原片。所有缩略图、JSON、CSV 和 HTML 都写入显式指定的输出目录。

## 判定类别

- `KEEP`：证据充分，未发现需要复核的异常。
- `REVIEW`：有异常，但不足以安全自动拒绝。
- `REJECT_CLOUD`：至少两类不同云证据一致，并包含强证据；这些证据并非严格统计独立。
- `REJECT_OCCLUSION`：大连通区域掉星，同时具有边界、背景或纹理证据。
- `UNASSESSABLE`：组内帧太少、星点不足、配准失败或格式能力不足。

“整张图变暗”本身不会直接触发拒绝。程序使用大量场星估计全局亮度比例；如果有可靠 `AIRMASS` 或高度角，会拟合每个滤镜自己的正常消光趋势。目标中心的像素亮度不作为云拒绝条件；非常大的扩展目标仍可能影响背景或检星数，因此相关异常会优先降级到人工复核。

局部透明度不依赖一张“永远正确”的参考片：每帧先配准到共同几何坐标，再按网格建立多帧较亮包络，分别记录局部变暗和变亮。被选作几何参考的帧也会重新接受这套组内共识评分。

程序也记录 NINA HFR、SEP FWHM 和星点椭圆率的组内稳健离群值。明显失焦、差 seeing 或全场拉线会进入 `REVIEW`；目前不会仅凭单个星形指标自动永久拒绝。

## 独立 Quality Gate

分类结论和发布门禁是两个不同层次：`Decision.KEEP` 只表示现有缺陷分类器没有命中问题；只有独立 `qualityGate.disposition=PASS` 才表示必需检查已经完成并允许自动进入 WBPP。

Quality Gate 同时检查：

- 源身份、LIGHT 角色、可解码性、有限像素覆盖和动态范围；
- 至少 8 张同类帧，以及至少 3 张同夜基线；
- 真实 peer 配准边、匹配星数/比例、RMS 和共同视场；
- 每夜 zeropoint + 同滤镜 airmass 趋势后的透明度残差；
- 局部透明度、检出星保留率、背景和噪声；
- 原生像素尺度 FWHM、NINA HFR、椭圆率、拉长星比例与方向一致性；
- 大面积连通缺星、清晰边界及背景/纹理支持。

门禁输出为：

- `PASS`：所有必需证据完整，且没有需要复核的证据；
- `REVIEW`：样本不足、单一异常、轻度云、focus/seeing、背景或配准异常；可用绑定 SHA-256 的人工 `APPROVE` 提升；
- `HARD_FAIL`：技术损坏、强同向全场拖线、强硬遮挡，或至少两个不同证据家族共同确认的强云；不能人工提升。

HFR 与 FWHM 合并为同一个形态证据家族，星数与匹配完整率也不会重复投票。整体亮度变化只有在每夜 zeropoint 和 airmass 模型之后仍异常才作为云证据，避免把拍摄高度角变化直接判坏。

## 安装

需要 Python 3.11 或更新版本。建议使用隔离环境：

```bash
# 从仓库根目录开始：
cd packages/light-frame-qc
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

检查环境和输入发现：

```bash
light-frame-qc doctor /path/to/night-1 /path/to/night-2
```

## 分析

```bash
light-frame-qc analyze \
  /path/to/night-1 \
  /path/to/night-2 \
  --output /path/to/light-qc-report
```

默认逐帧处理，适合 26MP/60MP 原片并控制内存。机器内存充足时可以使用有限并行：

```bash
light-frame-qc analyze /path/to/lights \
  --output /path/to/report \
  --workers 2
```

每个 worker 都可能解码一张完整的压缩图像，不建议盲目设成 CPU 核心数。

性能边界：一次新的分析会为每张 Light 完整读取两遍文件内容，一遍计算 SHA-256 并绑定源身份，一遍生成像素预览并完成检星；`prepare-wbpp` 生成计划时复用这份身份，不会再做第三遍 Light 哈希。`apply-wbpp-plan` 对每个实际复制的文件只做一次来源流式读取，并在同一循环中校验来源 SHA-256；私有 staging 中的每个目标只完整哈希一次。原子目录改名后以 plan 绑定的 digest、inode、size、mtime 和 ctime 快照确认仍是同一批文件，不再第三次读取整套目标。被排除或折叠为重复项的来源只检查 stat 坐标，manifest 明确写为 `stat-only-no-current-content-hash`，不声称 apply 时重新验证了其内容。旧 schema-1 计划如果没有这项显式策略，仍保留原来的所有来源完整 SHA-256 预检。

输出目录包括：

- `report.html`：可筛选的审阅报告和空间缺星热图；
- `frames.csv`：每帧结论、分数、原因码和关键指标；
- `results.json`：完整、版本化的机器可读结果；
- `thumbnails/`：只用于人工审阅的拉伸 PNG，不参与测量。

同一输出目录可以安全重跑。报告文件会原子更新；每次运行的缩略图使用新的运行标识，不覆盖旧文件。

## 自动准备 WBPP 输入目录

`prepare-wbpp` 会先识别 FITS/XISF 的真实角色，只把 header/XISF 元数据明确标记为 `LIGHT` 的帧送入质检。`MasterFlat`、raw flat、dark、bias 和 master light 不会再被当成低星数灯帧。

默认只生成计划和质检报告，不创建最终目录：

```bash
light-frame-qc prepare-wbpp \
  "/path/to/download-night-1" \
  "/path/to/download-night-2" \
  --output "/path/to/shield-wbpp-ready" \
  --report-output "/path/to/shield-wbpp-qc" \
  --workers 2
```

检查 `report.html` 和 `prepare-plan.json` 后，直接应用已有计划即可发布，不会重新测量、检星或配准；来源 SHA-256 与复制流合并计算，目标在原子发布前独立完整哈希：

```bash
light-frame-qc apply-wbpp-plan "/path/to/shield-wbpp-qc/prepare-plan.json"
```

计划会绑定同目录 `results.json` 的 SHA-256，并逐帧复核 source identity、Quality Gate、人工裁决和匹配 master flat。应用前不要编辑或移动 `results.json`；`planId` 是完整性校验和而非外部数字签名。

也可以不先审阅，在第一次命令上直接增加 `--apply` 一次完成：

```bash
light-frame-qc prepare-wbpp \
  "/path/to/download-night-1" \
  "/path/to/download-night-2" \
  --output "/path/to/shield-wbpp-ready" \
  --report-output "/path/to/shield-wbpp-qc" \
  --workers 2 \
  --apply
```

程序会在输入目录及向上两级的目录顶层自动寻找带权威 `MasterFlat` 元数据的 `masterFlat*.fits/xisf`。也可以显式提供：

```bash
light-frame-qc prepare-wbpp /path/to/downloads \
  -o /path/to/wbpp-ready \
  --flat-library /path/to/calibration-library \
  --master-flat /path/to/specific/masterFlat-R.xisf
```

Master flat 必须与灯帧的画幅、通道数、binning、滤镜和 CFA 状态精确匹配；相机、gain、offset 或 readout mode 在双方都有记录时也不得冲突。相同 SHA-256 的历史 flat 副本会折叠；兼容但内容不同的多个 flat 会以 `AMBIGUOUS_MASTER_FLAT` 失败，不按日期擅自猜选。当前版本只复用已经积分好的 master flat，不尝试自动校准 raw flat。

缺失 binning 或无法确定 CFA/mono 状态会 fail-closed。对 `QHY...M`、`ASI...MM` 等明确的单色相机型号可推断为 mono；无法从 header 或明确相机型号证明时，不把两个 `UNKNOWN` 当成精确匹配。

`--flat-library` 向自动发现集合追加候选；`--master-flat` 则对相同画幅/binning/滤镜/CFA profile 明确覆盖自动候选。若同一 profile 显式给出两个内容不同的 master flat，仍会按歧义失败。

生成布局如下：

```text
target-wbpp-ready/
  target-panel-1--<稳定ID>/
    LIGHT/R/*.fits
    LIGHT/G/*.fits
    LIGHT/B/*.fits
    FLAT/R/masterFlat*.xisf
    FLAT/G/masterFlat*.xisf
    FLAT/B/masterFlat*.xisf
  target-panel-2--<稳定ID>/
    ...
  prepare-manifest.json
```

在 WBPP 中每次只对一个目标子目录执行 **Add Directory**。不要直接把包含四块马赛克的最外层目录作为一次默认 WBPP 输入，否则相同画幅和滤镜的不同天区可能被错误地注册或积分到同一组。

只有 Quality Gate `PASS` 才会进入 `LIGHT`；Gate `REVIEW/HARD_FAIL` 只写入 manifest 并继续留在下载目录。人工 `APPROVE` 只能提升 Gate `REVIEW`，不能提升 `HARD_FAIL`；也可用 `REJECT` 否决 PASS。裁决必须通过 `--adjudication` 绑定绝对路径和 SHA-256：

```json
{
  "schemaVersion": 1,
  "records": [
    {
      "path": "/absolute/path/frame.fits",
      "sha256": "<64 hexadecimal characters>",
      "action": "APPROVE",
      "reason": "manual preview review",
      "reviewer": "operator"
    }
  ]
}
```

发布过程先在同一父目录的私有 staging 中以 create-only 临时文件复制；来源的 SHA-256 在复制流中核对，目标再独立核对 SHA-256、大小和 mtime，完整 manifest 写入后才一次性 no-replace 改名为最终目录。相同 master flat 供多个目标复用时，已与 device/inode/size/mtime/SHA-256 绑定的来源验证结果可复用，但每个目标文件仍单独 create-only、fsync 并独立哈希。失败不会留下半套成品；同一计划重跑会对现有成品完整重新哈希并返回 `ALREADY_COMPLETE`，不同计划或目录漂移绝不覆盖。原始文件不会被移动、重命名、删除或改写。

可复现的小型 I/O 计数基准见 [`benchmarks/README.md`](benchmarks/README.md)。它只生成临时 FITS，不包含用户素材或本机路径。

当前 WBPP-ready 目录只包含 light 和 master flat。Master dark 仍应在 WBPP 中从受控的共享校准库单独指定。

## PixInsight 辅助测量

核心分类是独立实现，使用 Astropy、SEP 和 Astroalign，不复制 PixInsight/WBPP 源码，也不声称重现 `PSFSignalWeight` 的私有数值算法。

如果用户已经通过自己有许可的 PixInsight 安装生成了 SubframeSelector v3 JSON，可以只读导入其中的 31 列全局测量作为辅助信息：

```bash
light-frame-qc analyze /path/to/lights \
  --output /path/to/report \
  --pixinsight-measurements /path/to/subframe-selector-v3.json
```

导入器要求请求路径、零基行号和所有 31 列严格匹配；`weight` 与未文档化的 `unused01` 不会用于云或遮挡判断。程序不会启动、链接、复制或重新分发 PixInsight 二进制与脚本。

## 支持格式

- FITS：`.fit`、`.fits`、`.fts` 以及 Astropy 能解码的 `.fz`；
- XISF：整数或浮点、未压缩 attachment 可流式读取，其他存储形式由 `xisf` 解码；
- 2D mono 和常见 3 通道图像。

默认只允许单张完整解码不超过 512 MiB。超过能力边界的压缩图会明确标记为 `UNASSESSABLE`，不会尝试危险的无限制分配。

## 参数与保守边界

默认参数在 [default-config.json](default-config.json)。复制后通过 `--config` 使用即可。首版自动拒绝规则刻意偏保守：

wheel 安装后也可用 `light-frame-qc show-config` 输出同一份内置配置，再保存为自己的 JSON。

独立门禁的完整默认阈值及其不可变 policy digest 可用 `light-frame-qc show-gate-policy` 查看；每帧报告和 prepare manifest 都记录实际使用的 digest。

- Quality Gate 组内少于 8 帧不得自动 PASS；同一观测夜少于 3 帧也只能 REVIEW；
- Quality Gate 配准默认要求至少 30 个匹配星、匹配比例至少 25%、RMS 不超过 1.5 preview pixel；
- 云自动拒绝要求总分至少 4、至少两类不同证据且至少一个强证据；
- 局部薄云可由网格测光残差高分位发现，但单一空间证据只进入 `REVIEW`；
- 缺星统计只在候选帧与参考帧的共同视场内进行，大幅 dither/旋转产生的非重叠边缘不算遮挡；
- 硬遮挡自动拒绝要求大连通掉星区域、清晰边界，以及独立的背景或纹理支持；
- 不同滤镜绝不直接比较亮度或星数。

均匀薄云与未建模的大气透明度变化在信息上可能不可区分；窄带少星、整组都坏、缺少时间/高度角、露水、失焦和跟踪失败也可能只能进入 `REVIEW` 或 `UNASSESSABLE`。未校准暗角、尘斑或大幅传感器旋转也可能影响背景证据；自动墙判定因此要求共同视场内同时出现掉星、边界和背景/纹理支持。在真实标注数据完成跨夜验证前，本程序的拒绝结果应视为候选隔离清单，而不是永久删除授权。

## 测试

```bash
# 从仓库根目录开始：
cd packages/light-frame-qc
python -m pip install -e ".[test]"
PYTHONPATH=src python -m pytest -q
```
