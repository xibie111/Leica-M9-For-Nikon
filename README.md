# Leica M9 For Nikon

面向 Nikon NEF 的电脑本地「徕卡一瞬 · M9」风格处理工具，提供原生 NEF / 自动转 DNG 两条输入路径、全尺寸细节处理、Windows 图形界面和批量队列。

本项目基于 [benbaobaoshigemi/OffLineM9](https://github.com/benbaobaoshigemi/OffLineM9) 独立维护。原有渲染管线、StyleTrans low/high/colorfix 网络权重和逆向资料来自上游，其风格调校还原自小米 17 Ultra「徕卡一瞬 · M9」。本项目新增的工作集中在尼康适配、桌面使用体验和可重复验证，没有重新训练一套模型。

## 相比上游，增加了什么

以最初引入的上游版本 `17cd094` 为基准：

| 方面 | 上游基础 | 本项目增加的能力 |
|---|---|---|
| 输入 | 以 DNG 为入口的离线渲染 | 原生 NEF 解码、自动转换无损马赛克 DNG，两路明确选择 |
| 尼康适配 | 通用 DNG 元数据与颜色矩阵 | NEF 曝光、白平衡、方向、有效照片区域及缺失相机矩阵适配 |
| 大图处理 | 大图先缩入约 4K 工作画布，最后放大 | `original` 模式将风格映射回全尺寸，并用全尺寸输入运行 colorfix |
| 日常使用 | 单张命令行 | Windows 界面、多选与文件夹导入、串行批量、完成当前张后停止、失败项重新入队 |
| 拍摄信息 | 基础 JPEG 输出 | 从原 NEF 白名单保留基础 EXIF，按成片方向与尺寸写入，不重新编码图像 |
| 批量模型 | 每张独立加载 | 同一轮队列复用严格校验后的模型，整轮完成或停止后释放，下一轮重新加载 |
| 输出与错误处理 | 通用渲染及可选后端 | NEF 入口严格要求网络成功；完整写入后发布、防覆盖、明确记录失败 |
| 验证 | 上游逆向与节点校验 | Z7 II / Z6 III 实拍、输入输出哈希、同图管线对照、窗口及队列回归 |

主摄 `main` 与长焦 `tele` 都是上游已有管线。本项目完成了它们在同一批 NEF 上的对照，并将日常 NEF 入口统一为完整的 **tele 管线**；这不是强制切换 low/high 网络，也不是在主摄成片上额外提亮。

## 快速开始

当前安装与启动脚本面向 **Windows x64、Python 3.12**，Python 需包含 Tkinter。CUDA 路径需要相容的 NVIDIA 显卡和驱动；依赖安装到项目内的 `.venv`。

在仓库根目录打开 PowerShell，按设备选择一种安装方式：

~~~powershell
# NVIDIA CUDA 环境
powershell -NoProfile -ExecutionPolicy Bypass -File .\setup_nef.ps1 -Device cuda

# 纯 CPU 环境
powershell -NoProfile -ExecutionPolicy Bypass -File .\setup_nef.ps1 -Device cpu
~~~

安装需要联网下载依赖。完成后启动界面：

~~~powershell
.\launch_nef_gui.cmd
~~~

1. 点击“添加照片”多选 NEF，或“添加文件夹”导入当前层文件；文件夹导入不递归。
2. 选择输出目录，按需设置分辨率与设备。
3. 点击“直接 NEF 处理”；如需走转换路径，选择“自动转 DNG 处理”。
4. 在队列中查看当前照片、阶段、成功或失败状态；处理结束后可打开输出目录。

“停止队列”会等待当前照片保存和清理完成，再停止后续任务；待处理照片可继续执行。单张失败后队列继续，失败项可以重新入队。队列仅保存在本次会话中。

更新代码后，请关闭旧窗口再重新启动。安装、解释器选择、下载源与排错详见 [NEF 使用说明](docs/NEF_GUIDE.md)。

### 单张命令行

直接处理 NEF：

~~~powershell
.\process_nef.cmd "D:\Photos\DSC_0001.NEF" --method native --output-dir "D:\Photos\M9"
~~~

通过 Adobe DNG Converter 处理同一张 NEF：

~~~powershell
.\process_nef.cmd "D:\Photos\DSC_0001.NEF" --method dng --output-dir "D:\Photos\M9"
~~~

`dng` 路径需要另行安装支持该机型的 **Adobe DNG Converter**。转换结果为临时的无损压缩马赛克 RAW DNG，后续仍运行完整渲染；转换失败不会自动改走 native。

也可以直接使用项目 Python 环境：

~~~powershell
.\.venv\Scripts\python.exe -m m9.nef "D:\Photos\DSC_0001.NEF" --method native --resolution original --device auto --output-dir "D:\Photos\M9"
~~~

| 参数 | 可选值 | 默认值 |
|---|---|---|
| `--method` | `native`、`dng` | `dng`，保留既有 CLI/API 兼容性；直接处理需显式写 `native` |
| `--resolution` | `original`、`phone` | `original` |
| `--device` | `auto`、`cuda`、`cpu` | `auto`，CUDA 可用时使用 CUDA，否则使用 CPU |
| `--write-json` | 额外导出处理记录 | 关闭 |
| `--converter` | 转换器可执行文件路径 | 自动查找标准安装位置 |

界面将“直接 NEF 处理”放在前面。显式请求 CUDA 但设备不可用时会报错，任务失败后不会自动更换设备、输入路径或分辨率。

## 当前处理管线

~~~text
NEF
 ├─ native：rawpy / LibRaw 原生解码
 └─ dng：Adobe 转换临时马赛克 DNG，再解码
         ↓
线性相机 RGB、D50 白平衡、曝光与色彩元数据
         ↓
tele AE 重测光、ADRC/TMC、色彩变换、TDL、gamma、CV
         ↓
StyleTrans low/high 风格推理 + colorfix 细节处理
         ↓
LeicaFilter LUT 与风格暗角
         ↓
保留基础 EXIF、嵌入 Display P3 ICC 的 SDR JPEG
~~~

日常 NEF 的单张 API、CLI、GUI 和批量队列共用处理核心，固定采用 `camera="tele"`、D50、额外 EV=0、JPEG 质量 95。使用上游 `ae_tele` 和 tele IPE 调校，保留原网络权重及选择门限，不额外增加曝光、对比度、阴影或人脸提亮。

- **EV=0 仍会执行原有 AE。** 它表示没有额外手动曝光补偿，不表示保留查看器里的亮度。AE 短曝光增益、中间调目标和 ADRC 是不同指标，负的短曝光 EV 不能直接当作成片整体变暗的档数。
- **tele 不裁切放大画面。** LeicaFilter 的 zoom 沿用上游规则：原片 35mm 等效焦距除以 23，限制在 3.2–4.3，缺失或非正值取 4.3。它是滤镜调校参数。
- **low/high 是共享网络。** 仍按原 lux/CCT 条件选择，随后运行 colorfix；不存在本项目另行训练的“长焦网络”。
- **原片与旧成片不覆盖。** 新文件名示例为 `DSC_0001_M9_tele_native_original.jpg`，重名时自动递增。JSON 默认不导出，开启后记录参数、模型状态、哈希和耗时，查看 JPEG 不依赖它。

原始照片范围与方向按相机元数据处理。切换依据、入口范围及回归结果见 [tele 默认管线验收](docs/NEF_TELE_DEFAULT.md)。

### 拍摄信息与批量模型

成片默认保留原 NEF 的相机、镜头、拍摄时间、快门、光圈、ISO、焦距与拍摄曝光补偿。native/DNG 均以原 NEF 为来源；拍摄曝光补偿仅作记录，渲染仍为额外 EV=0。方向写为正常，尺寸写为实际成片；保留 P3 ICC，不误标 sRGB，不复制 GPS、序列号、MakerNote 或缩略图。可选字段缺失或损坏时跳过并记录；JSON 关闭也会写 EXIF。

同一轮 GUI 队列首次严格加载 low、high、colorfix，后续逐张复用模型和权重身份，每张仍独立计算色温、照度和选择网络。整轮完成、停止或关闭等待当前照片完成后释放；下一轮重新加载。单张失败会清除模型，下一张重新初始化。单张 CLI/API 自行管理会话；自定义批处理器继续按原约定工作。不会缓存照片或推理输出，也没有改变网络精度、处理分辨率或推理 batch。实现与实测见 [EXIF 与模型会话验收](docs/NEF_EXIF_SESSION.md)。

在此基础上，固定 CUDA 画布复用有界输入缓冲区，P3 导出复用 128 行缓冲区，减少逐块转换与复制。三组成对四张批次的 24 份成片逐字节一致，整批中位数 101.94→91.54 秒；各轮波动较大，不能承诺固定提速比例，内存峰值也未降低。精度、分辨率、权重与调校保持不变。测量方法、全部样本和限制见 [成片一致的提速验收](docs/NEF_STRICT_SPEED.md)。

随后复核耗时波动并测试 CPU 线程/分条配置：旧记录中 CPU 前端只占慢快批次阶段差额的约 2.5%–4.8%，缺少遥测，不能追溯唯一系统原因；本机没有找到稳定胜出的替代配置。当前修复小图产生空条带的问题，线程数限制在有效任务数以内，常规照片保持原分条与工作线程。详细对照与本轮遥测见 [CPU 前端与耗时波动验收](docs/NEF_CPU_FRONTEND.md)。

2026-10-18 至 19 日分别试验 colorfix CUDA Graph、LeicaFilter 连续 GPU 搬运及组合，每项完成五组交错配对和末尾基线控制。含初始基线共 190 张性能测试成片的 JPEG 字节、像素、压缩数据、ICC、EXIF 均一致，但没有候选建立超过同期波动的整批收益；三项均已归档，默认继续使用原管线。完整耗时、启动成本、资源与验收见 [GPU 执行与滤镜搬运实验](docs/NEF_GPU_EXECUTION.md)。

### original 与 phone

两种模式都会生成经过 M9 风格处理的成片，名称表示工作分辨率策略：

| 模式 | 处理方式 |
|---|---|
| `original` | 小图运行风格网络，将风格映射回全尺寸，colorfix 使用全尺寸输入保留更多细纹 |
| `phone` | 沿用先缩入 4096×3072 工作画布的细节分支，再恢复输出尺寸，供对照使用 |

`phone` 不是手机应用，也不是未处理原图。`original` 保留输出尺寸和更多细节来源，但网络内部仍有 8 位处理、纹理柔化与 JPEG 有损压缩，不能称为 RAW 无损输出。

## 尼康支持范围

已用 **Z7 II 和 Z6 III** 的真实 NEF 完成处理验证，包含 FX/DX、横竖方向与多种场景。Z7 II 的无损压缩 NEF 已覆盖 native/DNG 两条路径。

项目收录固定 **LibRaw 0.22.1** 官方表中的 93 条尼康 `adobe_coeff` 矩阵资料，其中 85 条三通道条目可用于符合条件的缺项补全，8 条四通道条目保留来源数据；另保留老款机型的 `simple_coeff` 数据。

已有有效矩阵优先，仅在解码库返回完整全零矩阵、机型明确匹配且输入条件满足时补全。处理照片时不联网下载矩阵，也不靠近似型号猜测。

**矩阵收录不等于所有机型都已实拍支持。** 当前 NEF 流程限于三色 RGB、2×2 Bayer；HE/HE★、四通道及其他特殊编码不在本次新增支持范围。机型、数据来源与拒绝条件见 [尼康矩阵目录](docs/NIKON_MATRICES.md)。

## 已完成的验证

以下为截至 **2026-10-18** 的实际记录，不代表所有机型或场景的画质保证：

| 验证范围 | 实际结果 |
|---|---|
| Z7 II 样片 | 18 个不同 NEF 样本，分轮验证两路输入、两种分辨率、全尺寸细节与资源使用 |
| Z6 III 样片 | 38 个不同 NEF 样本完成 native 批量处理，原片哈希前后相同 |
| main / tele 对照 | 38 张完整配对，并提供 Windows 原生 RAW 默认显影与内嵌 JPEG 参考；重跑的 38 张 main 与旧版逐字节一致 |
| tele 日常入口 | 4 张真实 GUI 队列成片与冻结 tele 结果逐字节一致；另完成 Z7 II 默认 DNG CLI 实测 |
| EXIF 与模型会话 | 两台机身保留基础 EXIF；像素、ICC 与压缩图像数据不变；同策略下共享/独立会话成片逐字节一致 |
| 保持成片一致的提速 | 三组交错整批对照的 12 对 JPEG 全字节、像素、ICC、EXIF 一致；真实 45MP Z7 II 两路输入全字节一致 |
| CPU 前端与波动复核 | 83 例分条/线程筛选位一致；保留小图非空任务修复；7 批遥测对照完整 JPEG 一致，同基线约 24% 漂移不能认作代码提速 |
| GPU 执行与滤镜搬运实验 | 三候选各五组配对，190 张含基线成片一致；另完成 8 份 45MP 双路和两套 GUI 停止关闭，三候选均未超过波动、已归档 |
| 自动化回归 | 当前 522 项常规与 6 项 CUDA 短例已一并 **528 项通过**；默认命令跳过 CUDA 短例，覆盖矩阵、EXIF、几何、模型会话、输出发布、数值一致性及 CLI/GUI/队列连接 |

两台机身合计 **56 个内容不同的 NEF 样本**。历史 main、native/DNG、original/phone 验收各有覆盖范围，不应理解为 56 张都运行过全部组合。

tele 在本批对照中改善了许多主体偏暗的照片，同时也让部分夜景高光与噪点更明显，不能保证每张照片主观上都更好。Windows 原生 RAW 默认帧与相机内嵌 JPEG 都是显影参考；其显示编码亮度统计不能换算成真实曝光 EV，也不能代替手机实机验证。

运行自动化测试：

~~~powershell
.\.venv\Scripts\python.exe -m pytest tests -q
~~~

详细证据与限制：

- [NEF 实拍与全尺寸验收](docs/NEF_VALIDATION.md)
- [批量队列验收](docs/NEF_BATCH_VALIDATION.md)
- [Z6 III 兼容性修复与 38 张实测](docs/NEF_Z6III_VALIDATION.md)
- [tele 默认管线验收](docs/NEF_TELE_DEFAULT.md)
- [EXIF 与模型会话验收](docs/NEF_EXIF_SESSION.md)
- [成片一致的提速验收](docs/NEF_STRICT_SPEED.md)
- [CPU 前端与耗时波动验收](docs/NEF_CPU_FRONTEND.md)
- [GPU 执行与滤镜搬运实验](docs/NEF_GPU_EXECUTION.md)
- [相机矩阵与 RAW 元数据说明](docs/RAW_MATRIX_NOTES.md)

## 当前边界

- 本项目复用的是小米「徕卡一瞬 · M9」风格管线，不承诺等同真正 Leica M9 传感器或小米手机实拍。
- 上游离线 AE 仍有未完整复现的直方图、场景和人脸等子调整；跨相机测光与色彩也未取得实机等价或校准色彩证明。
- native 路径的场景色温采用单矩阵近似；与 DNG 路径的估算不同，可能影响调校插值和模型选择。
- 日常 NEF 入口当前输出 SDR JPEG，未提供手机端、Ultra HDR/TIFF 导出、新增降噪、人像分割或镜头校正。
- 真实 CPU 全尺寸性能、长时间内存稳定性、校准色卡及部分暖光近距离人像仍未完整覆盖。安装或设备自检通过不等于完整渲染验收。

这些边界不会通过额外自动增亮、任意曝光补偿或重新调曲线来掩盖。项目开发约束见 [AGENTS.md](AGENTS.md)。

## 原始 DNG 渲染与技术资料

上游研究入口继续保留，独立于日常 NEF 入口。它仍默认 `main`，可显式指定 `main` 或 `tele`：

~~~powershell
.\.venv\Scripts\python.exe -m m9.render "D:\Photos\input.dng" -o "D:\Photos\M9\main.jpg" --camera main --no-hdr
.\.venv\Scripts\python.exe -m m9.render "D:\Photos\input.dng" -o "D:\Photos\M9\tele.jpg" --camera tele --no-hdr
~~~

| 路径 | 内容 |
|---|---|
| `m9/` | NEF 入口、界面与队列，以及上游渲染、网络、滤镜与 HDR 研究实现 |
| `m9/assets/` | 原有 AE/IPE/LeicaFilter 调校与 Display P3 ICC |
| `m9/data/` | 尼康矩阵目录及来源、版权与许可说明 |
| `re/weights/` | 上游还原的 low/high/colorfix 权重 |
| `tests/` | 输入、矩阵、几何、风格处理、输出与界面回归测试 |
| `tools/` | 参数提取、矩阵目录生成、运行验证与逆向辅助工具 |
| `docs/` | 使用说明、适配依据与分阶段验收记录 |
| `decode/`、`re/`、`emu/` | 保留的上游权重解码、逆向资料和原库校验工具 |

进一步阅读：[上游项目](https://github.com/benbaobaoshigemi/OffLineM9)、[逆向发现](FINDINGS.md)、[风格分析](M9-style-analysis.md)、[权重解码工具链](decode/HANDOFF.md)。尼康矩阵数据的提取与许可说明见 [LibRaw NOTICE](m9/data/LibRaw-NOTICE.txt)。
