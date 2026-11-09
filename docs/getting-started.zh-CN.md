# 上手指南

Ultra-Fast WBPP 的第一次运行，一步一步来，面向用 N.I.N.A. 拍摄、也许手头有 PixInsight 的同好。走完之后，你会在一个新文件夹里拿到线性的、带天文坐标的主图。结尾的常见问题回答大家第一次跑之前最常问的事。English: [getting-started.md](getting-started.md).

## 需要准备什么

- 一台 Apple Silicon 的 Mac（macOS 14 或更新），或者 Windows 10/11 x64 的电脑。真实数据流程在一台 M3 Pro 和一台 Ryzen 7 5800H 笔记本上验证过；其他机器运行通用配置（[验证矩阵](validation-matrix.md)，英文）。
- FITS 或 XISF 格式的 Light，加上原始的 Flat/Dark/Bias 或已有的 Master（PixInsight WBPP 的 XISF Master 可以直接用）。不支持单反的 RAW 文件。
- 输出卷上的空余磁盘空间：全分辨率中间帧约占每张 Light 每像素 12 字节（26 MP 的相机约每张 310 MB）。
- 本机装一个解算器（第 2 步）。应用不内置。

## 1. 拿到应用

发布版本会出现在 [Releases 页面](https://github.com/ShuoleiWang/Ultra-Fast-WBPP/releases)。目前的用法是从源码构建（[指南](building-from-source.md)，英文）。在 Mac 上：

```bash
git clone https://github.com/ShuoleiWang/Ultra-Fast-WBPP.git
cd Ultra-Fast-WBPP
make bootstrap BOOTSTRAP_PYTHON=python3.12
make desktop-dev                                     # 直接从代码目录运行应用
# 或者，打一个本地的 .app：
CARGO_PROFILE_RELEASE_STRIP=none make desktop-build-macos-prerelease
```

Windows 的构建见 [windows.md](windows.md#developing-on-windows)（英文）。

**第一次打开。**

- macOS：应用为 ad-hoc 签名、未经公证。macOS 15 起右键 → 打开的绕过方式已经失效。打开"系统设置 → 隐私与安全性"，点**仍要打开**；或者去掉隔离标记：`xattr -dr com.apple.quarantine "Ultra-Fast WBPP.app"`。
- Windows：安装包未签名。SmartScreen 会提示"Windows 已保护你的电脑"，选"更多信息" → **仍要运行**。

界面默认是英文。语言切换在工具栏里，选择会被记住。

## 2. 配置解算器

每张主图发布前都要做天文定位（星图解析）并核验，所以应用在每台电脑上都需要一个解算器（[为什么不内置](licensing.md)，英文）。

1. **安装解算器。**
   - macOS：`brew install astrometry-net`，会装上 `solve-field`。
   - Windows：安装 [ASTAP](https://www.hnsky.org/astap.htm)（`astap_cli.exe`），并把 D20 或 D50 之类的星表数据库解压到 ASTAP 的文件夹里（[Windows](windows.md)，英文）。
2. **在应用里下载索引集。** 打开"高级选项与 solver 设置"，阅读并接受提供方条款，下载受管的 Astrometry.net 索引集（约 350 MB）。下载后按哈希校验并记录。然后点**重新检测配置**，解算器卡片应显示*就绪*。
3. **可选：星表重拟合。** 装了 Gaia DR2 LITE 分块星表（587 MiB）后，发布的 WCS 会在 Gaia 恒星上按 TAN-SIP（最高 3 阶）重新拟合，真实主图上留出验证约 0.15″；而在快速牛反上，解算器自己的解差了约 2.5″（[天文定位](recipes/astrometry.md)，英文）。应用暂时还不能下载这套星表，请用引擎的命令行：

   ```bash
   ultra-fast-wbpp catalog list
   ultra-fast-wbpp catalog install astrometry-net-5205-lite --accept-provider-terms <catalog list 打印的 id> --progress-json
   ```

   源码目录里命令是 `.venv/bin/ultra-fast-wbpp`。打包好的 macOS 应用里，引擎位于 `Ultra-Fast WBPP.app/Contents/Resources/resources/ufwbpp-engine/ufwbpp-engine-aarch64-apple-darwin/ufwbpp-engine-aarch64-apple-darwin`。细节见[离线解算星表](recipes/offline-solver-catalogs.md)（英文）。

换电脑时把索引目录（`<数据根目录>/catalogs/astrometry-net`）整个拷到新电脑的同一位置，然后打开应用或运行 `ultra-fast-wbpp catalog doctor` 即可，程序会自动改指向新位置；解算器程序本身仍需在新电脑上安装（[搬迁星表](recipes/offline-solver-catalogs.md)，英文）。

## 3. 导入几晚的素材

把所有东西一次拖到窗口上：每一晚的 Light、原始 Flat/Dark/Bias 或已有 Master、整棵 N.I.N.A. 文件夹树，或者你为 PixInsight WBPP 整理好的文件夹。

- 帧类型、目标、滤镜、曝光、增益和相机几何从头信息读取。头信息里没有类型的帧按文件夹或文件名识别，和 WBPP 一样；有冲突会显示出来，绝不猜。应用让你确认的类型要确认一下。
- 校准按 WBPP 的规则逐晚配对：`NIGHT_1`、`DATE_2026-03-22`、`SESSION-2` 这样的文件夹让每晚的 Flat 只配这一晚的 Light，放在夜晚文件夹之外的 Bias/Dark 库对每一晚都有效。折中（用了别的曝光的 Dark、Flat 没有自己的 Dark）会以警告显示。WBPP 已经校准或配准过的 Light 会被拒绝，请导入原始 Light（[校准](recipes/calibration.md)，英文）。
- 导入的 Master Dark 默认按包含 Bias 处理；如果你的已经减过 Bias，在校准高级设置里改。
- 检视器里的*校准检查*告诉你哪些配上了、哪些缺、哪些有歧义。
- 选一个输出文件夹；结果会写进它里面的一个新目录，这个选择会被记住。你的源文件永远不会被改写。

## 4. 闪视筛片：看完每个通道

点**闪视筛片（N 张 Light）**。应用测量每张 Light，然后打开一个暂停状态、按时间排序的复核界面，一次一个通道（目标 × 滤镜）。

- 一个通道的每张 Light 都对齐到该通道的参考帧，并共用一套拉伸，所以帧与帧之间变的是天空，不是显示。
- 每帧旁边的标记是测量出来的：相对本通道干净夜晚的天光、星点数、消光、原生 FWHM、椭率、配准质量、梯度。每个标记都写明数值和阈值。它们只是提示，不会替你丢片。
- `←` `→` 逐帧，`P` 播放，`F` 跳到下一个标记帧，按住 `C` 与参考帧 A/B 对比。`K` 保留并下一张，`D` 丢弃并下一张，`N` 整晚丢弃，`Z` 撤销。`1`–`9` 或 `Tab` 切换通道，`空格` 切换保留/丢弃，`R` 跳到参考帧。夜晚标题栏上有**整晚保留** / **整晚丢弃**。
- 该通道的每一帧都显示过之后，点**确认 &lt;滤镜&gt; 通道并继续**。之后再改决定，这个通道要重新确认。没能显示出来的预览不算看过；重试，或者明确把这帧丢弃。
- 每个通道至少要保留两张 Light。

标记、阈值和参考帧的规则写在[闪视筛片](recipes/blink-screening.md)（英文）里。

## 5. 处理

每张 Light 都看过、每个通道都确认、解算器就绪、输出文件夹已选，**开始处理**才会亮起；在那之前，底栏会列出还缺什么。默认设置就是推荐设置。"高级选项与 solver 设置"里可以开 drizzle（1×–4×）、关掉超级亮度，或者开启需手动开启的进阶算法。

运行会重新测量每一张 Light，严格采用你的决定，生成或复用校准 Master，配准、归一化、带稳健像素拒绝的积分，逐通道天文定位并核验解。进度按阶段实时显示。在 M3 Pro 上，61 张 26 MP 的 Light、四个滤镜约 76 秒；同一项目在 Ryzen 7 5800H 笔记本上要 4.5 分钟。

失败的运行不发布任何东西。有通道解算不出来时，这次尝试的回执和证据保留在 `<输出>.unsolved` 里，不写出任何科学产品。

## 6. 结果

普通（单面板）项目的新目录里有：

```
<输出>/
  L.fits  R.fits  G.fits  B.fits   # 每个滤镜一张线性主图，头信息里是校验过的 WCS
  RGB.fits 或 LRGB.fits            # R、G、B 齐全时（有 L 则为 LRGB）
  SuperL.fits                      # 各宽带通道的超级亮度（默认开启）
  previews/                        # 拉伸过的 PNG 预览，用来检查
  receipt.json                     # 输入、决定、算法标识、哈希、采用的解
  details/                         # shared-calibration/、runs/<目标>/ 及各阶段的回执
```

马赛克项目是每个滤镜一张共享画布上的马赛克，另有 `details/mosaics/<滤镜>/`（[马赛克](recipes/project-mosaic-rgb.md)，英文）。Drizzle 会多出科学、权重和覆盖产品（[drizzle](recipes/drizzle.md)，英文）。

主图是线性的、未拉伸的。用 PixInsight、Siril 或任何 FITS 查看器打开，像处理 WBPP 的主图一样做后期；WCS 已经在头信息里，并且在星表恒星上核验过。`SuperL.fits` 是宽带通道按各自噪声加权合成的一张更深的亮度图，星点明显更差的通道不参与；可以当亮度主图用，也可以不理它。

## 常见问题

**需要 PixInsight 吗？** 不需要。Ultra-Fast WBPP 替代的是 WBPP 做的预处理。拉伸、去梯度、反卷积和色彩校准还是要用后期软件；主图在 PixInsight 里就是普通的线性 FITS。

**我的数据会被上传吗？** 不会。解算器、索引和引擎都在你的电脑上运行。唯一的下载是你主动要的索引文件。

**可以直接导入为 WBPP 整理好的文件夹吗？** 可以，原样导入。帧类型从头信息或文件夹名（`Lights`、`Flats`、`Darks`、`Bias`、`DarkFlats`）读取，`NIGHT`、`DATE`、`SESSION`、`PANEL` 这些分组关键字按 WBPP 的读法解析，WBPP 的 XISF Master 当作已有 Master 使用。WBPP 校准或配准过的 Light 会被拒绝（[校准](recipes/calibration.md#files-from-pixinsight-wbpp)，英文）。

**"开始处理"为什么是灰的？** 底栏会说原因：有通道没看完或没确认、有通道保留的 Light 少于两张、解算器或索引集未就绪、没选输出文件夹、有帧类型没确认，或者校准的高级修改没保存。

**我用的是彩色相机。** Bayer 的 Light 会被识别，处理成 R、G、B 三个通道的主图和一个 RGB 产品，也可以做 Bayer drizzle（[OSC / CFA](recipes/osc-cfa.md)，英文）。这条路径目前只有合成数据验证，还没跑过真实的彩色相机数据。

**窄带，或者只有两个滤镜？** 可以。每个滤镜各自成一张主图；R、G、B 不齐全的项目照样发布主图，只是没有彩色产品。

**马赛克的多个面板？** 把所有面板一起导入。N.I.N.A. 的面板名（`M 31 Panel 3`）或 `PANEL_3` 文件夹会把它们分开；所有面板一起做一次天测拟合，放到同一张画布上，按重叠区恒星匹配。已在真实的 M31 马赛克（2×2、五面板、六面板）上验证（[马赛克](recipes/project-mosaic-rgb.md)，英文）。某块面板里 Tycho-2 星表的恒星太少时需要 Gaia 分块星表，应用会提示。

**运行说某个通道解算不出来。** 这次尝试保留在 `<输出>.unsolved` 里。检查解算器和索引集是否就绪（**重新检测配置**），Light 的像元比例是否在已装索引覆盖的范围内，再看看 `.unsolved` 目录里的回执。自己的解没过门禁的滤镜，在别的滤镜解算成功时会沿用同网格的解，由星表重拟合核验；这需要 Gaia 分块星表。

**为什么我跑不到 76 秒？** 那个数字是 M3 Pro 上 61 张 26 MP 的 Light。耗时随 Light 的数量和尺寸、单核速度和内存带宽变化（Ryzen 7 5800H 笔记本跑同一项目要 4.5 分钟）。Gaia 星表重拟合在那个项目上多花约 15 秒（79 秒对 64 秒）。Drizzle 是第二次积分：四个滤镜全做 2× drizzle 时跑了 178 秒（含 1 GB 的 drizzle 产品），当时普通运行是 97 秒（[features §1](features.md#1-speed-76-seconds-for-61--26-mp-lights)，英文）。

**跑两次结果会变吗？** 同一台机器上单色主图逐位一致。RGB 像素相同，但文件里的校验和关键字不同。macOS 和 Windows 的主图因数学库舍入不同而有差异，用容差门比较，不比哈希。

**主图和 WBPP 的比怎么样？** 用公开的标准和同一批帧的 WBPP 主图逐项测量：历史的分箱噪声增益 G₈ 在四个滤镜上为 1.01–1.03，PSF、背景和测光都在容差内；因为标准里还有门禁没测量，所以不认证完全等价（[features §5](features.md#5-master-quality-against-pixinsight-wbpp)、[标准](master-evaluation-standard.md)，英文）。

**能跳过闪视筛片让机器自己选吗？** 桌面端不能：每个通道都要看完并确认。命令行有默认的 `legacy-gate` 和配方选项 `unattended-v1`，后者在积分内部做留一复核（[自动筛片](recipes/automatic-screening.md)，英文）。

**索引文件在哪里，能拷到另一台电脑吗？** macOS 在 `~/.ultra-fast-wbpp/catalogs/astrometry-net`，Windows 在 `%LOCALAPPDATA%\Ultra-Fast-WBPP\catalogs\astrometry-net`（改名前装的版本仍用 `~/.openastroflow` 或 `%LOCALAPPDATA%\OpenAstroFlow`），`ultra-fast-wbpp catalog doctor` 会打印出来。把这个文件夹拷到另一台电脑的同一位置再打开应用即可，程序会自动改指向新位置；解算器程序本身仍需在新电脑上安装（[搬迁星表](recipes/offline-solver-catalogs.md)，英文）。

**遇到问题去哪里反馈？** 开一个 issue，附上 `receipt.json`（里面没有像素数据；发之前看一眼有没有你不想公开的文件路径）；运行失败的话，再附上 `.unsolved` 目录里的回执。想自己动手修，见 [CONTRIBUTING.md](../CONTRIBUTING.md)（英文）。
