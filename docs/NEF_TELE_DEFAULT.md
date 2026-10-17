# NEF 默认 tele 管线

按用户在 main/tele 同图对照后的明确选择，日常 NEF 处理统一切换为上游完整 `camera="tele"`。此前的 main 成片和验收保留为历史记录；本次不重新调校任一管线。

## 实际入口与参数

- `m9.nef.process_nef()` 是单张 API、`python -m m9.nef`、GUI 默认处理器和批量队列共用的处理核心。核心向真实前端传入 `camera="tele", ev=0.0`，由既有代码选取 `ae_tele.json` 及 tele CC、TDL、gamma、CV、TMC 调校。
- low/high StyleTrans 与 colorfix 仍用原权重和门限：`lux>260 && CCT<4692` 选 low，其余选 high，随后运行 colorfix。严格网络开启、D50、original 默认分辨率、auto 默认设备、JPEG 95/P3/SDR 及 JSON 默认关闭保持现状。
- 直接复用 `m9.render._tele_zoom()`，读取原 NEF 的 `FocalLengthIn35mmFilm`，按 `/23` 限制至 3.2–4.3，缺失或非正值为 4.3。此 zoom 仅用于 LeicaFilter，不改变照片像素范围或尺寸；DNG 路径也从原 NEF 求该参数。
- 报告记录 `camera_tuning="tele"` 和实际 `zoom`。新命名为 `原名_M9_tele_native_original.jpg` 或 `原名_M9_tele_dng_original.jpg`，按现有规则递增防覆盖；旧 main 文件继续保留。
- GUI 固定显示“调校：tele（长焦）”；没有新增管线开关。native/DNG 按钮顺序、CLI/API 的 method 默认 dng、队列及错误处理规则不变。

原始通用 DNG renderer `m9.render`、`frontend.b2y` / `b2y_from_raw` 的默认 camera 仍为 main，其显式 tele 功能保持原状。没有修改 Nikon 矩阵目录、RAW 解码、AE/影调算法、LUT、权重、网络门限或滤镜实现。

## 验证依据

冻结对照位于 `E:\m9output\tele_vs_main_20261017_154604`：38 份 main/tele 配对、原件哈希、代码和权重快照、实际网络调用记录。代表性新输出必须与该目录中的 tele JPEG 核对，不能以“看起来更亮”作为接入正确的证据。

本轮独立证据位于 `E:\m9output\tele_default_20261017`，查看入口为 `查看tele默认验证.html`。

## 本轮验收结果

- 完整自动化 **382 passed in 11.84s**。覆盖两种解码方法、实际 tele 前端、原 NEF 焦距缺失/非法/上下界/中间值、滤镜和报告参数、防覆盖、CLI/GUI/队列接线，以及原始 DNG renderer 的 main 默认边界。Python 编译和 diff 空白检查通过。
- 真正 `NefWindow(root)` 的默认处理器，经 native 按钮和串行队列处理 0192 强光深阴影、0398 夜街、1567 逆光人像、1628 河边 DX 四张原片：4/4 成功，43.20 秒，实际 CUDA、original、JSON 默认关闭。high 网络执行 4 次、colorfix 336 次，未替换推理。
- 四份新 JPEG 与冻结 tele 成片**逐字节相同**，AE、原始元数据/裁切、CCT、lux、模型、zoom 均相同。前三张为 4032×6048，DX 为 2656×3984；zoom 为 3.2、3.2、4.3、4.3，P3 和完整解码通过。Tk 心跳 1715 次，最大观测间隔 0.662 秒；这不是长期稳定性保证。
- 实际 `python -m m9.nef` 命令处理 Z7 II `E:\m9input\DSC_0150.NEF`，没有指定 method/resolution/device，因此采用 dng/original/auto；经 Adobe 转换和完整 tele 管线成功输出 5408×3600 P3 JPEG，实际 CUDA、StyleTrans high、zoom=4.3、额外 EV=0。该项显式加 `--write-json` 核对磁盘报告，未改默认关闭行为；它没有冻结 tele JPEG 配对，不作逐字节对照声明。
- 五个原件哈希不变。35 个受保护的解码、矩阵、AE、前端、调校资产、权重、网络、滤镜、通用 renderer、队列及依赖文件哈希不变。

逐项依据为 `evidence/gui_native_001.json`、`cli_dng_001.json`、`pytest_001.log`、`before_001.json` 和最终 `final_validation_001.json`。本次没有重跑全部 38 张，也没有扩展成所有机型或所有场景的画质保证。

## 使用注意

关闭旧窗口，再双击 `launch_nef_gui.cmd` 才能载入新代码。已经保存的 main 成片不会改写；已打开旧窗口中的 Python 模块不会热更新。

tele 可能改善原来主体偏暗的照片，也可能让夜景灯牌、高光和噪点更明显；不保证每个场景主观上都更好。对照中的显示编码 Y 统计不能换算成真实曝光 EV。当前矩阵、色温近似和全尺寸细节的边界继续见 [NEF 使用说明](NEF_GUIDE.md) 与 [尼康矩阵目录](NIKON_MATRICES.md)。
