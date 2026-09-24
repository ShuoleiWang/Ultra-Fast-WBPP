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

性能边界：一次新的分析会为每张 Light 完整读取两遍文件内容，一遍计算 SHA-256 并绑定源身份，一遍生成像素预览并完成检星。

输出目录包括：

- `report.html`：可筛选的审阅报告和空间缺星热图；
- `frames.csv`：每帧结论、分数、原因码和关键指标；
- `results.json`：完整、版本化的机器可读结果；
- `thumbnails/`：只用于人工审阅的拉伸 PNG，不参与测量。

同一输出目录可以安全重跑。报告文件会原子更新；每次运行的缩略图使用新的运行标识，不覆盖旧文件。

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

独立门禁的完整默认阈值及其不可变 policy digest 可用 `light-frame-qc show-gate-policy` 查看；每帧报告 都记录实际使用的 digest。

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
