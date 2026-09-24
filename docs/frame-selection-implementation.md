# 无人值守筛片：实现规格（准备稿）

状态：2026-09-19 准备稿。设计依据 [frame-selection-plan.md](frame-selection-plan.md)。**代码在 `perf/in-memory-integration` 的 PR 合入 main 之后再动**；本稿只固定接口、算法、文件与任务，供实现时直接执行。引用的行号以 main `13310a1` 为准，合入后需复核。

## 0. 范围与开关

- 新增 `SelectionParameters`（`ufwbpp/selection/parameters.py`），挂在 `E2ERequest.selection` 与 recipe（`recipe.py`）上：
  - `policy: "legacy-gate" | "unattended-v1"`，默认先保持 `legacy-gate`，验证后切换默认值；
  - `priority: "depth" | "balanced" | "resolution"`（默认 `balanced`）；
  - `aggressiveness: "conservative" | "standard" | "aggressive"`（默认 `standard`，只影响灰区阈值）；
  - `counterfactual: "off" | "analytic" | "binned-rerun"`（默认 `analytic`）；
  - `region_weights: bool`（P3 前默认 False）。
- 所有新行为都在 `policy="unattended-v1"` 下生效；`legacy-gate` 路径逐位不变（现有 E2E 测试作为回归护栏）。
- receipt 新增 `qualityControl.selection`（策略、优先级、每帧决定与理由、权重、反事实结果、区域图摘要）；`qc/manifest.json` 的 `qualityGate` 保持不变（仍是证据来源）。

## 1. 模块布局

```
ufwbpp/selection/
  parameters.py      SelectionParameters, PriorityProfile(p 指数、FWHM 截止 k、灰区阈值表)
  features.py        FrameSelectionFeatures: 从 FrameResult/FrameMeasurement/registration 提取 ~20 个特征
                     + 时间序列特征（同夜按 observed_at 排序的斜率/单调性）
  guards.py          硬缺陷规则（L3）：来自 quality_gate 的 HARD_FAIL 子集 + 透明度下限 + 配准失败
  weights.py         统一权重：w_i = (T_i^2/σ_i^2) · FWHM_i^(-2p) · c_i · n_i；输出 quality_weights 供 integrate_expressions
  region.py          16×16 网格 → 区域权重图（平滑、边缘余量、双线性上采样描述）；输出 WeightMapSpec
  policy.py          决策：exclude / keep / keep_with_maps；生成 SelectionDecision 与理由码
  counterfactual.py  oracle：解析留一 ΔQ（深度、背景、FWHM 代理）与 4×4 分箱重跑模式
  report.py          qc/selection-report.html 与 receipt 片段
ufwbpp/reference/   （评估参照，已有目录约定）
  selection_oracle_numpy.py       oracle 的朴素 NumPy 参照（测试用）
```

现有 `lightframeqc` 不改判定逻辑；只新增两个测量项（§4）。

## 2. 数据契约

```python
@dataclass(frozen=True, slots=True)
class FrameSelectionFeatures:
    path: str; night_id: str; observed_at: float | None
    transparency: float | None          # 恒星尺度提示 T（相对参考帧，已含曝光校正）
    extra_extinction_mag: float | None  # 气质量包络之上的额外消光
    noise_sigma: float | None           # 归一化后 4×4 块噪声（block-mean-effective-noise-v2）
    fwhm_native: float | None; fwhm_night_ratio: float | None; fwhm_trend_slope: float | None
    wing_fraction: float | None; wing_trend_slope: float | None   # 结露代理（§4）
    ellipticity_median: float | None; orientation_coherence: float | None
    spatial_dimming_p90_mag: float | None; star_completeness: float | None
    background_z: float | None; noise_z: float | None
    overlap_fraction: float | None; registration_rms: float | None; registration_ok: bool
    occlusion_area: float | None; occlusion_density: float | None; boundary_support: float | None
    gate_hard_fail_codes: tuple[str, ...]; gate_review_codes: tuple[str, ...]
    grid: GridEvidence | None           # 16×16: transparency residual, completeness, background_delta, texture_ratio

@dataclass(frozen=True, slots=True)
class SelectionDecision:
    path: str
    action: Literal["EXCLUDE", "KEEP", "KEEP_WITH_MAP"]
    weight: float                       # 归一化前的 w_i（EXCLUDE 时 0）
    confidence: float                   # c_i ∈ [0.25, 1]
    reasons: tuple[str, ...]            # 例如 SEL_TRANSPARENCY_BELOW_NORMALIZATION_FLOOR
    map_digest: str | None              # KEEP_WITH_MAP 时区域图 sha256
    counterfactual: CounterfactualResult | None
```

`CounterfactualResult`：`delta_depth_mag`, `delta_background_sigma`, `delta_fwhm_proxy_px`, 各自的 bootstrap 区间与 `mode`（analytic / binned-rerun）。

## 3. 算法

### 3.1 权重（P2）

- `T_i`：`estimate_stellar_scale_hints` 的接受尺度（`ufwbpp_registration/quality.py:235`），参考帧为 1；不可用时 1 并将 `c_i` 乘 0.7。
- `σ_i`：`calibration._normalized_noise_weights` 现有的块噪声估计（`calibration.py:2253`）已在归一化后测量，因此 `1/σ_i^2` 即 `T_i^2/σ_raw,i^2`；**不要再乘一次 T²**。
- PSF 项 `FWHM_i^(-2p)`：p = 0（depth）/ 1（balanced）/ 2（resolution），FWHM 用全分辨率星点 r50（§4），回退到预览 FWHM × 尺度。
- `n_i`：夜间因子，= 夜内最佳 FWHM 与全体最佳 FWHM 之比的 −2p 次方（避免整夜偏软时逐帧重复惩罚）。
- 合成：`w_i = noise_i · fwhm_term_i · n_i · c_i`，再按 `_combined_integration_weights` 的现有归一化与 `[median/16, median·16]` 夹取。
- 替换点：`e2e.py:2440-2455` 现用 `normalize_quality_weights(run.analyses)` 生成 `quality_weights`；`unattended-v1` 下改为 `selection.weights.quality_weights(features, priority)`，`native_quality_score` 保留在 receipt 作对照。

### 3.2 反事实 oracle（P0）

解析留一（默认）：在 `integrate_expressions` 的每个 tile 内，对拒绝后的样本有 `S = Σ w_i x_i a_i`, `W = Σ w_i a_i`（a 为接受标志）。对每帧 i：`m_{-i} = (S − w_i x_i a_i)/(W − w_i a_i)`。指标按 tile 累加、tile 内不存整栈：

- 深度代理：`σ_eff(8)` = 8 × MADN(平面去除后 8×8 块均值)，只在星掩模外的 tile（星掩模来自参考帧目录，半径按 r50 与流量分档）；ΔQ_depth,i = −2.5·log10(σ_eff,−i / σ_eff,all)（正 = 去掉 i 更深 = i 有害）。
- 背景代理：256 px 背景图的 P2 残差 RMS（复用评估标准 §5 的定义），Δ 同上。
- FWHM 代理：`FWHM_master^2 ≈ Σ w_i FWHM_i^2 / Σ w_i`，留一解析。
- 置信区间：按 tile 自举（B=200）。
- 拒绝掩模来自全集（二阶近似）；receipt 注明 `rejectionMaskSource: "full-stack"`。
- 成本：每 tile 多做 N 次归约，全分辨率 61 帧约 30–60 s；`binned-rerun` 模式为 4×4 分箱后完整重跑（用于标定与基准，不进日常运行）。
- 内核：`MaskedWeightedMean` 旁增加 `LeaveOneOutTileStatistics`（单源头文件 `kernels/reduction/leave_one_out.h`），输入与现有请求相同，输出每帧的块均值 MADN 累加量；先用 NumPy 参照实现并测试，再做原生。

### 3.3 决策（P1）

顺序：护栏 → 权重 → 灰区 → 反事实确认。

1. 护栏（EXCLUDE）：`GATE_NOT_A_LIGHT_FRAME`、`GATE_MEASUREMENT_*`、`GATE_IDENTITY_*`、`GATE_FINITE_FRACTION_HARD`、`GATE_NEAR_CONSTANT_IMAGE`、`GATE_FRAGMENTED_TRAILING_HARD`、`GATE_COHERENT_TRAILING_HARD`、`GATE_OCCLUSION_HARD`（面积 > 0.50 时；≤ 0.50 转区域图）、配准失败、`T < 0.50`（归一化 `minimum_scale`）、`extra_extinction_mag > 0.75`。
2. 证据不足类 REVIEW（`GATE_INSUFFICIENT_COHORT`、`GATE_NIGHT_UNRESOLVED`、`GATE_INSUFFICIENT_NIGHT_BASELINE`、`GATE_MORPHOLOGY_SAMPLE_REVIEW`、`GATE_SOURCE_COUNT_MISSING`、`GATE_FINITE_FRACTION_REVIEW`、`GATE_COMMON_FOOTPRINT_REVIEW`、`GATE_REFERENCE_NOT_CONNECTED`）：KEEP，`c_i = 0.5`。
3. 缺陷类 REVIEW（云、消光、背景、噪声、对焦、拖长、遮挡 REVIEW）：进入灰区：按优先级表取阈值（例如 resolution 下 `fwhm_night_ratio > 1.3` 直接 EXCLUDE；depth 下 > 2.0 才 EXCLUDE），其余 KEEP 并降权 `c_i = 0.5–0.8`；局部云/遮挡且 `region_weights=True` → KEEP_WITH_MAP。
4. 反事实确认：对第 3 步产生的每个 EXCLUDE，以及权重最低的 10% KEEP 帧，计算 ΔQ；EXCLUDE 仅在 `ΔQ_depth,lo > +0.005 mag` 或 `ΔQ_background,lo > 0.05 σ` 时成立，否则降级为 KEEP（`c_i = 0.5`）；KEEP 帧若 `ΔQ_depth,hi < −0.01 mag` 且 `ΔQ_background,hi < 0`（去掉它反而更差）则恢复 `c_i = 1`。
5. 比例护栏：软原因 EXCLUDE 超过 30% 帧 → 对整组做留组反事实，只保留能自证的排除。
6. 面板帧数 < 2 的阻塞逻辑保持（`e2e.py:4489-4513`）。

### 3.4 区域权重（P3）

- 来源：`analysis.py:675-700` 的网格；权重 `m = clip(1 − dimming_mag/0.45, 0, 1) · [completeness ≥ 0.35] · [|background_z| < 3 ∨ texture_ratio > 0.5]`，遮挡连通域内置 0，向外膨胀 1 格作余量，Gaussian σ = 1 格平滑。
- 存储：每帧 16×16 float32 + 节点坐标（沿用 `offset_grid` 的双线性约定，`calibration._add_offset_grid_rows` 的插值路径可直接复用为乘性图）。
- 内核：`MaskedMeanRequest` 增加可选 `frameMajorWeightMaps`（与样本同布局，或 16×16 节点 + 在内核内插值）；单源头文件 `kernels/reduction/weighted_mean.h` 增加 `perSampleWeight` 访问器，CPU/CUDA 驱动同时获得。
- 覆盖图：`coverageFraction` 改为有效权重比 Σ w_i m_i / Σ w_i，另存 `effectiveWeight` 图。

## 4. 测量新增（lightframeqc）

- 全分辨率星点 r50/FWHM：对预览目录里最亮 200 颗非饱和星，在原图上取 25×25 星像做 `sep.flux_radius`（memory 记录预览 FWHM 约有 2× 膨胀）。
- 翼部比例 `F(2·FWHM)/F(10 px)` 及其同夜时间斜率；FWHM 时间斜率。均写入 `FrameFeatures`（新增字段，`models.py:274-315`）与 manifest。
- pier side：header `PIERSIDE`，缺失时留空。

## 5. 测试

- 单元：`test_selection_weights.py`（公式、夹取、夜间因子、缺失特征回退）、`test_selection_policy.py`（每条规则一个用例 + 三种优先级 + 比例护栏）、`test_selection_region.py`（网格 → 图，遮挡膨胀、平滑、上采样与 `_add_offset_grid_rows` 一致）、`test_counterfactual.py`（解析留一 vs 朴素 NumPy 参照逐位；4×4 重跑与解析在合成栈上一致到 CI 内）。
- 差分：`legacy-gate` 下 E2E 输出逐位不变（用 `bit-identity-verification` 的方法）。
- 合成缺陷注入 harness（`benchmarks/selection_defect_injection.py`）：在 NGC 7331 真实帧上注入薄云/结露/失焦/遮挡与良性对照，输出召回/误报表；作为 P1 的验收。
- 基准协议（`benchmarks/selection_benchmark.py`）：六个基线（全保留+旧权重、全保留+新权重、legacy-gate、unattended-v1、手工、SFS 阈值）× 数据集 → `evaluate_masters.py` 判定 + ΔQ 混淆矩阵。

## 6. 任务顺序（合入后执行）

| # | 任务 | 依赖 | 估算 |
|---|---|---|---|
| 1 | `selection/` 骨架、参数、recipe/E2E 接线、`legacy-gate` 差分测试 | PR 合入 | 1 天 |
| 2 | `counterfactual.py` NumPy 参照 + 解析留一（tile 内累加）+ 测试 | 1 | 2 天 |
| 3 | 在 NGC 7331 与 B fixture 上跑 oracle，输出当前 QC 的混淆矩阵与报告 | 2 | 0.5 天（Opus 执行） |
| 4 | `guards.py`/`policy.py`（无人值守策略）+ `report.py` | 1 | 2 天 |
| 5 | 合成缺陷注入 harness + 验收 | 4 | 2 天 |
| 6 | `weights.py` 接入 + 用 ΔQ 验证 | 2,4 | 2 天 |
| 7 | 全分辨率 r50、翼部比例、时间斜率测量 | — | 1.5 天 |
| 8 | 区域权重：`region.py` + 内核逐样本权重（单源）+ 覆盖图 | 4 | 4 天 |
| 9 | 基准协议脚本与首份结果；默认策略切换决定 | 3,5,6,8 | 2 天 |
| 10 | 原生 `LeaveOneOutTileStatistics` 内核（可选加速） | 2 | 2 天 |

## 7. 与近期主线改动的关系

- 拒绝尺度模型 v2 与块噪声权重 v2（`99e84ad`）：oracle 使用 v2 的掩模与噪声；权重公式以 v2 的块噪声为 σ_i。
- 快速 Radon 拖线检测（同上）：像素级，不影响帧级决策；`GATE_FRAGMENTED_TRAILING_HARD` 仍是帧级护栏。
- `perf/in-memory-integration`：积分在内存中进行，oracle 的 tile 内累加应挂在同一条路径上，避免二次读盘。
- 桌面端一步式流程（`db4271f`）：`unattended-v1` 下 Review 页只展示决定与理由，不再需要审批才能包含帧；保留强制包含/排除两种覆盖。

## 8. 进度（2026-09-19）

- 已实现（本地，未提交）：`selection/{parameters,features,guards,policy,counterfactual}.py`、`reference/selection_oracle_numpy.py`；`integrate_expressions` 的 `tile_observer` 钩子；`integrate_registered_group`/`_cpu_with_receipt` 透传；`_run_portable_pipeline_fits(_integration_tile_observers=...)`；`E2ERequest.selection`、recipe `selection` 块、`runtime` 接线；E2E 在 `unattended-v1`/`include-all` 下按决策放行、按置信度缩放配准质量权重、逐 tile 累积留一反事实并写 `qc/selection.json`；测试 `test_selection.py`（7 项，含与朴素参照逐值比对）与 `test_e2e_selection.py`（保留薄证据 REVIEW 帧并降权；全 PASS 时与 legacy 逐位一致）；脚本 `benchmarks/selection_oracle_report.py`（混淆矩阵）与 `benchmarks/selection_defect_injection.py`（真实帧注入合成缺陷）。
- 反事实的实现与设计的差异：为控制开销，留一统计在 8×8 块和上做（块级加权均值），而非逐像素；星掩模由整合 tile 的中值 + 5σ 阈值得到；背景代理用 tile 行 × 256 px 段的粗网格做二阶多项式残差 RMS；拒绝掩模取自全集。此版本的反事实只做**报告与建议**（`suggestion`），不改变已做的决定；二次积分留待真实数据验证后加入。
- 待做：任务 3（NGC 7331 混淆矩阵，运行中）、任务 5 验收、任务 6 统一权重、任务 7 测量新增、任务 8 区域权重、任务 9 基准协议、任务 10 原生留一内核。

### 8.1 首轮真实数据结果与标定（2026-09-19）

- **NGC 7331 四晚 61 帧（`include-all`）**：门与 oracle 完全一致——2 帧 HARD_FAIL 被护栏排除（多家族云 + 透明度低于下限 + 额外消光），61 帧 PASS 中 oracle 判 20 帧 BENEFICIAL、41 帧 NEUTRAL、0 帧 HARMFUL；6 帧 L 的 ΔQ_depth 为正但置信区间跨零（最大 +0.007 mag）。每组约 66 个统计 tile；反事实开销合计 6.2 s（占 103 s 运行的 6%）。这套数据没有 REVIEW 帧，未能检验"证据不足即保留"的路径。报告：本地 oracle 报告（`benchmarks/selection_oracle_report.py` 的输出，未入库）。
- **合成缺陷注入（DATE_0322 的 12 帧 R，注入 6 种）**首轮暴露了三个标定问题，已修正：
  1. QC 的 `transparency_ratio` 相对于星数最多的参考帧，晴帧也只有 ~0.8；绝对下限 0.5 会误杀 0.65 透明度的薄云帧。改为相对于组内 PASS 帧中值（薄云 0.42/0.81 = 0.52 → 保留降权）。
  2. `GATE_MULTI_FAMILY_CLOUD_HARD` 对均匀薄云和离焦帧都会触发（离焦帧的小孔径通量比看起来像云）。均匀透明度损失可由归一化尺度建模，因此从护栏移到灰区（置信度 0.5），只保留透明度相对下限与额外消光 > 0.75 mag 两条护栏。
  3. 结露帧（σ=2.2 px 模糊 + 光晕）在门里没有任何证据（预览 FWHM 比 1.29，未到 1.30；文件名里的 NINA HFR 未变）。选择层现在对**所有**帧应用 PSF 规则：超过优先级截止排除，否则权重乘 (FWHM/组中值)^(−2p)（balanced 下 1.29× 的帧权重 ×0.60）。预览 FWHM 灵敏度不足（真实 4 px 被测成 8.5 px）仍是任务 7 要解决的问题。
- 注入观测：遮挡 25% → REVIEW → 保留 0.5（区域权重前的临时处理）；拖线 → REVIEW → 保留 0.5 并叠加 PSF 因子；良性均匀变暗 → PASS、无误报。
- 观察到的工程问题并已修：recipe 的 `selection` 块在 CLI 往返时被拒绝（`serializable()` 曾输出派生键）；`run-project` 的运行目录在 `details/runs/*`；真实 NINA 帧下 FakeSolver 的固定中心会被天测门拒绝（基准脚本改用按提示解算的 `HintFakeSolver`）。

### 8.2 第二轮（2026-09-19 晚）

- **反事实开始行动**：确认有害（tile ≥ 16、区间越过阈值、且相对组内 median + 3·MADN 为离群）的已放行帧被排除并二次积分（最多 `maxIntegrationPasses`=3 轮，累计排除受 30% 软护栏约束；归一化参考帧与面板最少 2 帧的约束下保留）。`qc/selection.json` 记录 `reintegration{status, excluded, keptBecause, passes}` 与 `integrationPasses`。
- **优先级语义**：resolution 下 FWHM 代理 Δ > 0.10 px 且离群即有害；depth 不看 PSF；balanced 只用权重因子。
- **遮挡**：面积 ≥ 15% 且密度 ≤ 0.5 的连通缺失区在区域权重落地前软排除（合成注入证实：块噪声权重会奖励被遮挡的低噪声区域，遮挡帧曾拿到组内最大权重）。
- **原生 PSF 测量**（`lightframeqc/native_psf.py`）接入：QC 在原图上对最亮 200 颗星切 25 px 邮票测 r50/FWHM/翼部比例，写入 `FrameMeasurement.native_psf` 与 `FrameFeatures.psf_*`；选择特征优先用它（`fwhmSource: native-stamps`），预览 FWHM 只作回退。
- **合成注入结果（12 帧 R，v4）**：balanced：薄云被反事实确认排除、遮挡软排除、离焦/拖线降权、结露保留但权重 ×0.62，良性无误报（recall 0.8）；resolution：结露与拖线也被反事实确认排除，recall 1.0，无误报；include-all：oracle 判薄云与离焦有害（+0.15 / +0.04 mag）。
- **NGC 7331（unattended-v1 / balanced）**：1 帧 B 被护栏排除（透明度 0.26×、额外消光 1.07 mag），1 帧 R 降权 0.51，无二次积分；L 通道对 PI：G_8 = 1.020 [1.011, 1.030]、FWHM 相同、高阶背景更平、显示域区域跨度更小，判定"不劣且 SNR 更高"（深度门因 CI 宽为 WARN）。

### 8.3 第三轮（2026-09-19 夜）：原生 PSF 修复与区域权重

- **原生 PSF 之前从未生效**：Opus 复跑 v5 发现 63/63 帧 `fwhmSource` 仍是 `preview-scaled`，`wingFraction` 全为 null。原因是 `native_psf.py` 用 `memmap=True` 打开 16 位 FITS（BZERO=32768），astropy 拒绝并抛出 `ValueError`，`_native_psf_summary` 把异常吞成 `{"error": ...}`。修复：`open_native_image()` 以 `do_not_scale_image_data` 内存映射原始整数，按星像小块施加 BSCALE/BZERO（BLANK→NaN），XISF 直接附件走 `np.memmap`、压缩 XISF 走解码；不再整帧解码。真实 R 帧实测：r50 = 2.03 px，FWHM = 4.05 px（预览估计 8.4），翼部比例 0.148，200 颗星，单帧测量 0.36 s。新增测试覆盖 BZERO/BSCALE/BLANK 与多通道布局。
- **区域权重（P3）已实现**：`selection/region.py` 从 QC 网格（完备度、透明度残差、背景差、纹理比）生成每帧 16×16 权重：缺星连通域（≥2 格）置 0 并向外膨胀 1 格；透明度残差超过 0.05 mag 死区后线性降权，0.45 mag 归零；背景异常且纹理消失的格置 0；零区外约 1 格的高斯过渡；干净帧不生成图（最小权重 > 0.98）。`FrameExpression` 新增 `weight_grid/_x/_y`（像素坐标节点，边缘夹取），`integrate_expressions` 按带插值成逐样本权重交给原生 masked-mean V2（或 NumPy 参照，逐位一致），覆盖图改为有效权重比，`TileObservation.sample_weights` 让留一 oracle 用同样的权重累加块和；Metal 路径遇到权重图自动回落 CPU。策略：有图的帧不再因遮挡软排除，`GATE_OCCLUSION_REVIEW`/`GATE_SPATIAL_DIMMING_REVIEW` 不再扣置信度；图空白超过 50% 的帧软排除（`SEL_REGION_MOSTLY_BLANK`）。回执：`qc/selection.json.regionWeights`（每帧节点、零格数、证据计数）、pixel 回执每组 `regionWeightMaps` 与积分 `execution.regionWeights`。
- 默认仍为 `regionWeights: false`，待缺陷注入（遮挡/局部云）与 NGC 7331 验证后决定是否随 `unattended-v1` 默认开启。

### 8.4 v6 结果与修正（2026-09-19 夜）

v6（原生 PSF 生效后，12 帧注入，balanced/resolution × 区域权重开/关）暴露三个问题，均已修正：

1. **整夜零点偏移把干净帧全打成 REVIEW**：注入的 3 帧暗帧让该夜的零点比最佳夜低于阈值，`GATE_NIGHT_ZEROPOINT_SHIFT_REVIEW` 给了全夜每一帧，策略按缺陷灰区降到 0.6（`cleanFalsePositives: 6`）。这类"整夜/单帧均匀变暗"由归一化尺度精确建模，透明度与额外消光护栏已限制其幅度，噪声权重随之自动降权，因此 `GATE_NIGHT_ZEROPOINT_SHIFT_REVIEW` 与 `GATE_TEMPORAL_EXTINCTION_REVIEW` 现在不扣置信度（`SEL_KEEP_NORMALIZED_*`）。
2. **干净帧也生成了区域权重图**（每帧 24–33 个"变暗"格，最小权重 0.25–0.42）：原始残差网格相对参考帧含共有的平场/参考帧结构（顶行 +0.1 mag、底行 −0.1 mag）与暗星单格离群（到 0.3 mag，MADN 约 0.035）。改为使用 QC 的共识残差 `consensusDimmingResidualMag`（本帧减去队列晴空包络，中位数居中），死区提高到 0.08 mag，且只接受 8 连通 ≥ 3 格的变暗斑块；`patchy_cloud`（87 格、p90 0.45 mag）不受影响。
3. **有图的遮挡帧仍被反事实判定有害**：帧级噪声权重在整帧格点上估计，遮挡区噪声极低（σ≈3 对 15）使中位数绝对差腰斩、权重膨胀约 3.5×，清晰区被过度加权。现在 `_frame_noise_estimates` 对带权重图的帧把图值 < 0.5 的样本置 NaN 后再估计噪声。
4. 另外 `GATE_SPATIAL_DIMMING_STRONG` 在有图时也视为图已处理（不再双重扣权）；重叠加记录里增加确认有害的反事实证据（Δdepth、CI、Δbackground），否则第二轮报告会丢失第一轮的数字。

v6 其他结论：原生 PSF 让缺陷帧在 FWHM 上清晰分离（失焦 8.2 px、拖线 8.1 px、结露 6.6 px 对干净 3.6–4.3 px；翼部比例结露 0.084 对干净 0.13–0.15，注意结露注入是"核+晕"，翼部比例反而下降，真实结露需实测）；NGC 7331 unattended/balanced + 区域权重：L 判定 EQUIVALENT（v2 为 INCONCLUSIVE），B 判定 INCONCLUSIVE（v2 为 WORSE，平顶星计数 175→165 转 PASS），G_8 略降（L 1.020→1.012）而 FWHM 由略差转为略好，是 balanced 优先级的预期取舍；QC 测量 4.73→5.48 s（+16%），总耗时 103.9→114.5 s（其中 27 张"化妆性"权重图是无谓开销，修正 2 应消除）。

### 8.5 v7 结果：坐标系错位与最终修正（2026-09-19 夜）

v7（修正 1–3 后）：干净/良性帧全部 KEEP 1.0（`cleanFalsePositives 0`，`benignFalsePositives 0`），标准六缺陷召回 1.0；NGC 7331 耗时回到 105 s，L EQUIVALENT / B INCONCLUSIVE 不变。但带权重图的帧在第一轮积分中权重异常（遮挡帧占总权重 44.5%，Δdepth +0.55 mag），局部云帧也被反事实按"背景变差"剔除，且干净帧仍有微弱图。

单轮诊断（`--max-passes 1`）定位到根因：**QC 网格在 QC 参考帧（星最多的帧，本例来自 2026-09-16 夜）的坐标系，而像素管线的配准参考由 `ufwbpp_registration` 另选（本例 2026-09-12 夜的帧），两夜之间是 180° 子午线翻转**。权重图被原样贴到配准帧上，于是清晰区被清零、遮挡区以全权重进入，噪声估计的掩模也盖在了清晰区。修正：

- `RegionWeightMap.transformed(matrix, height, width)`：把图按 `S·Q_f·S⁻¹·T_f⁻¹`（T_f 管线配准矩阵、Q_f QC 配准矩阵、S 预览→原生尺度）重采样到配准帧坐标系（每格中心映射后双线性取值，落在 QC 参考帧外的格权重 1）；E2E 在送入管线前逐帧转换，回执标注 `frame: registered`。
- 变暗项改为 `ramp × 10^(−0.4·r)`（透过率因子反映未重标定样本的信噪比），斑块需 ≥ 4 个 8 连通格且中位残差 ≥ 0.12 mag；无零格且均值 > 0.99 的图视为化妆性、不生成。
- 噪声估计对带图的帧分层掩模（先排除图 < 0.9 的样本，样本不足再退到 < 0.5，再不足则不掩模）。
- 反事实确认规则：背景变差只作平局裁决——深度有显著收益（CI 上界 < −0.01 mag）的帧不因背景残差被剔除。

修正后的单轮诊断：覆盖图的低覆盖区落在遮挡真实位置（右下象限 0.874）；局部云帧以 1.0 权重保留且反事实有益（Δdepth −0.062 mag，Δbackground −0.013σ）；遮挡帧 Δdepth −0.113 mag（深度显著有益）、Δbackground +0.095σ（平局裁决下保留，置信度 0.6 来自遮挡/噪声/星数保留的灰区码）。遮挡帧的噪声权重仍为干净帧的 2.5 倍，核对原始帧后确认是真实物理差异：2026-09-16 夜的天光 527 ADU、噪声 8.9 ADU，对 2026-09-12 夜 572–601 ADU、14.8–17.8 ADU。

### 8.6 v8 结果与收尾（2026-09-19 深夜）

v8（坐标系修正后）：四组注入均无误杀（`cleanFalsePositives 0`、`benignFalsePositives 0`）；局部云帧在有区域权重时以 1.0 权重保留，反事实为有益（balanced：Δdepth −0.062 mag、Δbackground −0.013σ；无区域权重时只能以 0.66 权重保留且背景 +0.028σ）；遮挡帧保留（置信度 0.6），Δdepth −0.113 mag（显著有益），Δbackground +0.095σ；覆盖图的低覆盖象限与遮挡真实位置一致（两夜各自翻转方向都正确）。NGC 7331：耗时 105.9 s，L EQUIVALENT / B INCONCLUSIVE 不变，L 主图像素逐位不变（仅头信息不同）。全量 971 通过。

两处遗留与处理：

- 遮挡帧的 +0.095σ 背景残差不是归一化偏移网格外溢（该帧的偏移网格 33×49、幅度 48–63 ADU，平滑；遮挡区未被抬升，靠权重图置零排除）。它来自权重随位置变化本身：任何帧的归一化都有几 ADU 的残差，权重在遮挡区为 0、其他区为 26%，主图背景就出现 ~0.1 ADU 的台阶——统计显著（bootstrap CI 不含 0），天文上无意义。`harmfulBackgroundSigma` 从 0.05 提高到 0.25（块噪声单位，约合评估器 0.05 像素噪声的容差），并保留"背景只作平局裁决"规则；harness 的 oracle 标签改用同一规则。
- 单个"缺星格"（3 颗预期星全部未匹配）会生成只清零一格的图（NGC 7331 两帧、注入集一帧）；孤立缺星格不再视为遮挡，遮挡连通域须 ≥ 2 格（8 连通）。重采样时零区改为"最近格为零则为零"，避免亚格配准偏移让边缘余量通过插值变淡（v8 中零格从 81 缩到 72）。

默认值：`regionWeights` 改为 true（仅对无人值守策略生效；`legacy-gate` 不生成图），`policy` 默认仍是 `legacy-gate`，是否切换由用户在更多数据集上决定。另外核实：注入集中来自 2026-09-16 夜的帧（遮挡、良性对照）权重是干净帧的 2–3 倍，是真实物理差异（该夜天光 527 ADU、原始噪声 8.9 ADU，对 572–601 / 14.8–17.8），不是估计偏差。
