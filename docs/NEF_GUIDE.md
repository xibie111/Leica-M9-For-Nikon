# Nikon NEF 本地处理

这份入口把 Nikon NEF 接到现有 M9 后续处理。日常 NEF 单张 CLI/API、GUI 和批量队列统一使用完整的长焦 tele 调校、D50 白平衡逻辑、额外 EV=0，输出质量 95、嵌入 Display P3 ICC 的普通 SDR JPEG。原始 NEF 不会改写。

tele 直接使用上游 `ae_tele` 和 tele IPE、色彩、gamma、TMC 调校，共享原 low/high StyleTrans 与 colorfix 网络及选择门限。它改变的是整条已有相机管线，不额外加曝光、曲线或阴影修饰。滤镜 zoom 按原 NEF 的 35mm 等效焦距除以 23，限制在 3.2–4.3；缺失、零或负值取 4.3。此参数用于 LeicaFilter LUT 和暗角，不裁切或放大照片。窗口固定显示“调校：tele（长焦）”，没有新增选择开关。

独立的原始 DNG 命令 `python -m m9.render` 仍默认 main；其显式 `--camera tele` 用法和通用前端默认值保持原状。NEF 默认切换的范围、实拍对照和证据见 [tele 默认管线验收](NEF_TELE_DEFAULT.md)。

## 当前电脑启动

双击项目根目录的 `launch_nef_gui.cmd`。“添加照片”可多选 NEF；“添加文件夹”导入该文件夹当前层的 NEF，按文件名排序，不递归子目录。列表显示文件名、来源目录、状态和结果位置。选择输出目录，然后选择其中一个处理按钮执行所有待处理项：

- **直接 NEF 处理**：使用 rawpy/LibRaw，直接解码 NEF，并使用相机白平衡、曝光信息和 LibRaw 相机矩阵。此路径的场景色温为单矩阵估算近似值。
- **自动转 DNG 处理**：调用本机 Adobe DNG Converter，创建临时无损压缩、保留马赛克的 DNG，检查元数据后进入 DNG 前端。临时文件在本次任务结束后清理。

两条路径共用后续处理，但色彩可能不同，应对同一张照片做对照。队列始终一次只处理一张，窗口继续显示当前照片、阶段和日志。本轮进度按已结束任务数计算，成功和失败均计入；阶段没有虚构百分比。

分辨率默认 `original`，设备默认 `auto`。`auto` 在 CUDA 可用时使用 CUDA，否则使用 CPU；显式 `cuda` 不可用时会报错。CPU 处理完整尺寸照片可能需要较长时间。

输出命名示例：`DSC_0001_M9_tele_native_original.jpg`。`_tele_` 区分新默认管线与旧 main 成片；处理记录中的 `camera_tuning="tele"` 和 `zoom` 表示实际采用的调校及滤镜参数。默认只生成 JPEG；需要旁边的 JSON 处理记录时，勾选“导出 JSON 处理报告”。该选项默认关闭，关闭时窗口的阶段和日志仍正常显示。已有同名结果不会覆盖，不同来源的同名照片自动递增输出名。JPEG 完整写入后才发布为正式文件；失败时不会留下同名的半成品 JPEG。

JPEG 默认包含原 NEF 的基础 EXIF：相机、镜头、拍摄时间、快门、光圈、ISO、焦距及拍摄曝光补偿。native/DNG 使用相同原始来源；拍摄补偿只作记录，不再次用于渲染。成片方向为正常、尺寸为实际输出，P3 ICC 保留且 EXIF 不标作 sRGB。不复制 GPS、序列号、MakerNote 或缩略图。可选字段缺失、损坏或过长会跳过并记入处理记录；不依赖 JSON 开关，也不重新编码 JPEG 像素。

## 队列、停止与失败

同一规范化绝对路径只入队一次，已完成项也不会因重复添加而再入队；不同路径不会按内容自动合并。空闲或停止后可以移除选中的待处理项，也可以清空列表；清空列表不会删除已经保存的成片。

每次开始或继续时，固定该轮的输入方法、分辨率、设备、输出目录和 JSON 选项。运行期间不能修改这些设置或队列。“停止队列”会等待当前照片处理、保存和清理完成，再停止后续任务；剩余照片保持待处理，再点击任一处理按钮即可继续，使用新一轮选定的参数。

每轮队列在工作线程中管理一个模型会话，首次严格加载并校验三个网络，后续照片复用模型与权重身份。每张照片仍独立选择 low/high，不缓存照片或推理结果。单张失败时清除模型，下一张重新加载；整轮完成或停止后释放，清理完成才恢复界面操作，下一轮重新初始化。单张 CLI/API 自动创建和释放自己的会话。释放缓存不代表 CUDA 驱动上下文或整卡显存读数归零。

单张失败会在列表和日志记录原因，然后继续下一张；不逐张弹出阻塞对话框。结束后集中查看失败项，点击“失败项重新入队”后，再点击处理按钮执行。成功项不会因此重跑。处理失败不会自动切换方法、设备或分辨率，也不会回滚已成功保存的照片。可用“打开输出目录”查看结果。

运行中关闭窗口，可选“完成当前张后关闭”或“返回”。选择关闭后不再开始下一张，当前照片成功或失败并清理完毕后才关闭。队列只保存在本次会话中，关闭后不会恢复；成片及已开启的 JSON 报告仍保存在输出目录。单张 CLI 和 `process_nef()` 接口不变，本阶段没有增加批量 CLI。

批量自动化和三张真实 NEF 连续运行结果见 [批量验收记录](NEF_BATCH_VALIDATION.md)，可视入口为 `E:\m9output\batch_validation_20261017\查看批量验收.html`。更新后重新打开窗口即可载入队列界面。

基础 EXIF、会话生命周期、成片像素不变及模型复用性能实测见 [EXIF 与模型会话验收](NEF_EXIF_SESSION.md)。

当前版本进一步减少固定 CUDA 画布的逐块转换、复制和 P3 导出分配，仍使用相同的 float32 网络、权重、调校、分辨率与串行顺序。三组交错四张批次的完整成片全部相等，整批耗时中位数 101.94→91.54 秒；机器波动较大，不能保证固定提速比例，内存峰值略增。真实 45MP Z7 II 的 native/DNG 也分别与冻结基线全字节一致。全部耗时样本、缓冲区边界和回归证据见 [成片一致的提速验收](NEF_STRICT_SPEED.md)。更新后关闭旧窗口并重新启动即可载入。

2026-10-18 又复核了 CPU 线程数和分条大小，未采用没有稳定收益的配置，保留常规照片调度；仅修复小图空条带并按有效任务数限制工作线程。旧批次波动不能主要归因于 CPU 前端，本轮没有新的固定提速结论。控制重复、系统遥测和精确输出检查见 [CPU 前端与耗时波动验收](NEF_CPU_FRONTEND.md)。

## 脚本入口

在项目根目录执行；所有带空格或中文的路径均建议加引号：

```powershell
.\.venv\Scripts\python.exe -m m9.nef "D:\照片\DSC_0001.NEF" --method native --output-dir "D:\照片\M9 输出"
.\.venv\Scripts\python.exe -m m9.nef "D:\照片\DSC_0001.NEF" --output-dir "D:\照片\M9 输出"
.\.venv\Scripts\python.exe -m m9.nef "D:\照片\DSC_0001.NEF" --method dng --resolution phone --device cuda --output-dir "D:\照片\M9 输出"
.\.venv\Scripts\python.exe -m m9.nef "D:\照片\DSC_0001.NEF" --write-json --output-dir "D:\照片\M9 输出"
```

也可以使用 `process_nef.cmd`，参数相同：

```powershell
.\process_nef.cmd "D:\照片\DSC_0001.NEF" --method native --resolution original --device cpu --output-dir "D:\照片\M9 输出"
```

参数：`--method dng|native`（默认 dng）、`--resolution original|phone`（默认 original）、`--device auto|cuda|cpu`（默认 auto）、`--write-json`（额外导出 JSON，默认不导出）。使用 `--help` 查看完整参数。

窗口按“直接 NEF 处理”在前、DNG 在后的顺序显示，Tab 键顺序相同。CLI/API 的 method 默认值仍为 dng；脚本选择直接 NEF 时需显式传 `--method native`。

自动查找 Adobe DNG Converter 的标准安装位置。若安装在别处，可在启动窗口或脚本前设置：

```powershell
$env:M9_DNG_CONVERTER = "D:\Adobe\Adobe DNG Converter.exe"
.\launch_nef_gui.cmd
```

转换器缺失或失败只影响 dng 路径；可另外选择 native 路径。处理程序不会为选定任务自动改选路径。

## 独立环境与复现

依赖安装在项目内 `.venv`，不修改系统或 Miniconda 的包。新环境使用 Python 3.12；安装脚本优先查找用户目录下的 Miniconda Python，也可以指定解释器：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\setup_nef.ps1
powershell -NoProfile -ExecutionPolicy Bypass -File .\setup_nef.ps1 -Python "C:\Path\To\Python\python.exe"
```

纯 CPU 环境可传 `-Device cpu`。安装需要下载依赖和 PyTorch，CUDA 版 PyTorch 下载量约 3.5 GB，可能耗时较长。`requirements-nef.txt` 固定直接依赖；PyTorch 固定为 `2.8.0+cu128`（CPU 为 `2.8.0+cpu`），本机验收环境的完整版本见 `requirements-nef-lock.txt`。CUDA 环境需要相容的 NVIDIA 显卡和驱动，CPU 安装不能执行 CUDA。

脚本默认从 PyPI 和 PyTorch 官方索引安装。如果当前网络无法访问，可显式指定镜像；本次环境下载使用了以下镜像：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\setup_nef.ps1 -IndexUrl "https://pypi.tuna.tsinghua.edu.cn/simple" -TorchIndexUrl "https://mirrors.nju.edu.cn/pytorch/whl/cu128"
```

安装时临时解压目录使用项目内 `.work\temp`，pip 禁用下载缓存，减少系统盘额外占用；仍需给项目所在磁盘留出 PyTorch 下载和解压空间。已有匹配的本地安装包时，可传 `-TorchWheel "D:\下载\torch-2.8.0+cu128-cp312-cp312-win_amd64.whl"`，避免再次下载。CPU 安装包名称应对应 `2.8.0+cpu`。

安装脚本不修改系统代理。若终端继承了失效代理，须在运行安装的当前进程中按自己的网络配置处理；应用不会替用户改动系统网络设置。安装末尾会检查依赖、三份权重文件是否存在，并实际执行一次所选设备的张量运算；完整网络和真实 NEF 的验收见后文。

三份模型权重必须同时存在：

```text
re/weights/low.npz
re/weights/high.npz
re/weights/colorfix.npz
```

NEF 入口严格要求 StyleTrans 成功执行。缺失依赖、模型、权重不相容、CUDA 显存不足都会明确失败；不会静默缩图，也不会仅套预览 LUT 后报告成功。设备选择同时用于网络和 LeicaFilter。

## NEF 矩阵与色温来源

rawpy 的 `rgb_xyz_matrix` 名字容易引起方向误读；其底层返回 LibRaw 的 `cam_xyz`。实现对照 [LibRaw `cam_xyz_coeff`](https://github.com/LibRaw/LibRaw/blob/master/src/utils/utils_dcraw.cpp) 与 [LibRaw 相机表](https://github.com/LibRaw/LibRaw/blob/master/src/tables/colordata.cpp) 中 Nikon Z 7 / Z 7_2 的处理，使用 XYZ → camera 方向，不因属性名盲目转置。

直接 NEF 路径把相机 as-shot 白平衡换成 neutral，以单矩阵求场景白点，再用 McCamy 估算 CCT；处理记录标注 `cct_is_approximate=true` 和来源。它不是 DNG 的双光源标定。DNG 路径保留既有双矩阵插值逻辑，转换后检查色彩矩阵、as-shot neutral、曝光、CFA 马赛克、无损压缩和有效尺寸。

程序已收录固定 LibRaw 0.22.1 表中的全部尼康矩阵：93 条 XYZ→camera（85 条三通道、8 条四通道），并单独保留 E880/E900/E990 的旧式矩阵。仅在库返回全零矩阵、明确机型匹配且满足标准 CFA 输入条件时补全三通道矩阵；已有有效矩阵的路径保持原样。四通道、HE/HE*、未知机型及损坏矩阵不因收录目录而获得支持。完整机型、来源、范围和回归见 [尼康矩阵目录](NIKON_MATRICES.md)。此适配不更改 M9 曝光、风格或模型门限，也不会自动切换到 DNG。

38 张 Z6 III 全部成功重跑的结果、原件哈希和 Z7 II 回归见 [Z6 III 修复验收](NEF_Z6III_VALIDATION.md)；成片在 `E:\m9output\native_matrix_fix_20261017\photos`，查看入口为上层的 `查看修复结果.html`。更新后需重新打开窗口才能载入修复。

## 分辨率与画质边界

`original` 保留全尺寸原图，单独创建符合原有补边和尺寸流程的预览给 1024×768 风格网络，再按同一空间映射恢复风格图。colorfix 每次处理一个 544×544 块、步长 512，原图输入来自全尺寸图像。LeicaFilter 暗角按整图坐标分段计算。

`phone` 保留原有先缩进 4096×3072 的流程作为对照。现有 DNG 命令的默认行为保持兼容：

```powershell
.\.venv\Scripts\python.exe -m m9.render input.dng -o out.jpg
```

全尺寸输出不等同 RAW 位深或细节完全无损：风格网络内部仍使用 8 位数据，网络可能柔化纹理；JPEG 有损压缩。此实现不保证等同小米实拍或真正 Leica M9。不包含手机端、Ultra HDR、TIFF、新降噪、人像分割或镜头校正。

## 验证与真实样片

运行自动化检查：

```powershell
.\.venv\Scripts\python.exe -m pytest tests -q
```

自动化测试、合成细节图与替代网络后端可检查矩阵方向、空间映射、原图块来源、边缘处理、分段一致性、错误路径和窗口响应。这些结果不能替代真实相机画质证明。具体实际运行结果见[本次验收报告](NEF_VALIDATION.md)；安装检查本身也不能证明完整网络运行成功。

2026-10-17 已用用户指定的 5 张 Z7 II NEF 完成两路输入与两种分辨率的 20 份结果、同位置 100% 裁切、实际资源测量及连续三张检查，见[验收报告](NEF_VALIDATION.md)。结果在 `E:\m9output`，对照与记录在 `E:\m9output\validation_20261017`。暖光人物、长期内存趋势和 CPU 全尺寸性能仍未验证。

同日又增加 13 张内容不同的新照片，完成 13 份 native original、5 份同图 DNG original 和 0274 的两份 phone 对照。全图、100% 裁切和日志入口为 `E:\m9output\extended_validation_20261017\查看扩测.html`。其中 0189/0274 两路估算 CCT 跨过现有模型选择阈值，可能出现更明显的色调/反差差异；详细资源、输入路径中断及覆盖缺口见验收报告末节。

日常可先用首个“直接 NEF 处理”按钮和 `original`；CLI 要显式写 `--method native`。DNG 可用于同图色彩比较，尤其在估算 CCT 接近 4692 K 时。两路不是校准色彩真值；original 较 phone 保留更多细纹，也保留更多噪声，默认风格可能让暗部偏黑。

这次实拍还修复了 Nikon 相机 inset crop：FX 输出 8256×5504、DX 输出 5408×3600，竖图按方向换向。Adobe 对少数快门值的元数据改写只有在完整 RAW 马赛克一致等条件成立时才被接受，实际 M9 测光仍采用原 NEF 曝光值。更新代码后，请关闭旧窗口，再双击启动脚本重新打开，确保载入修复。

后续对同一张 NEF 分别运行 native/dng 与 original/phone 时，如需保存处理记录，请勾选“导出 JSON 处理报告”或添加 `--write-json`。查看同位置 100% 裁切，并确认记录中 StyleTrans 已成功运行；比较纹理、肤色、色阶、块接缝和网络柔化。建议覆盖日光、暖光人像、树叶或织物、高光、高 ISO。测试人员只读取用户提供的样片路径，不扫描整个磁盘寻找照片。关闭新任务的 JSON 导出不会删除已有报告或验收记录。

需要复核时，可将导出的处理记录与成片一起保留。记录包含输入方法、尺寸、设备、色温来源、曝光调整、模型状态、各阶段耗时，以及本次资源采样结果。RSS 为进程内存的间隔采样值，GPU 分配峰值与整卡显存占用含义不同；不要把单一数值当成整机资源上限。
