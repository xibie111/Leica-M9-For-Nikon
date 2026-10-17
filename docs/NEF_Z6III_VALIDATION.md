# Z6 III 零颜色矩阵修复与 38 张实测

日期：2026-10-17。用户粘贴的日志显示，`E:\街拍` 中指定的 38 张照片在 native/original/auto、JSON 关闭时全部于 RAW 阶段报 `RAW colour matrix must be a finite nonsingular 3 x 3 XYZ-to-camera matrix`。

## 原因与修复

仅检查日志中的 38 个路径：全部为 `NIKON Z6_3`（Z6 III）、固件 2.00，内容哈希互异。当前 rawpy 0.25.1 / LibRaw 0.21.4 能解包它们，白平衡有效，但 `rgb_xyz_matrix` 全部为零。首张通过未修复的 `load_nef()` 实际复现同一异常。

LibRaw 0.21.4 相机表没有 Z6_3，后续 [0.22.0 的官方表](https://github.com/LibRaw/LibRaw/blob/0.22.0/src/tables/colordata.cpp#L1036-L1037) 为该机型提供独立系数。此次仅在确定机型、厂家、标准压缩 CFA 结构且完整矩阵恰为全零时，按上游 float32/10000 语义回补该条目。原有有效矩阵保持原路径；未知机型、损坏矩阵、非法白平衡、CCT 和尺寸仍失败。HE/HE* 及不完整数据被该适配拒绝，没有自动转 DNG。

没有升级解码依赖或更改 AE、曝光、曲线、StyleTrans、colorfix、LeicaFilter、模型门限。相机矩阵来源会写入日志和内存报告，开启 JSON 时也进入旁边的处理记录。系数、格式边界和一手来源详见 [矩阵说明](RAW_MATRIX_NOTES.md)。

## 验证结果

- 完整自动化：**206 passed in 10.09s**；其中 RAW 输入测试 106 项。新增覆盖准确的官方系数、既有矩阵保持原样、损坏/未知矩阵拒绝、HE 标记与异常条带拒绝、白平衡/CFA 检查、FX/DX 裁切和八种方向。
- 修复后实际解码 Z6 III FX 和 DX 成功。原有 Z7 II `E:\m9input\DSC_0150.NEF` 完整 native 线性 RGB 哈希、cam2srgb 和 CCT 与修复前完全相同。
- 原日志的 **38 张全部重新处理成功，失败 0**，由真实 Tk 队列逐张执行 native/original/auto，实际设备 CUDA、`StyleTrans=on`，JSON 关闭。没有用缩图、替代网络或 DNG 冒充成功。
- 38 个原 NEF 处理前后 SHA-256 相同。38 份 JPEG 的哈希、完整解码、尺寸及 P3 ICC 全部核对；照片目录没有 JSON、锁或临时残留。原始失败日志和成功重跑记录均保留。

| 范围 | 数量 | 正式输出尺寸 |
|---|---:|---|
| FX 竖图 | 22 | 4032×6048 |
| FX 横图 | 11 | 6048×4032 |
| DX 竖图 | 5 | 2656×3984 |

FX 的原始尺寸为 6064×4040、照片范围 `[8,4,6048,4032]`；DX 为 4000×2672、照片范围 `[8,8,3984,2656]`。沿用原有整像素裁切与方向映射，没有为匹配尺寸而重采样。

另外目视抽查 0015、0067、1628 的全图预览和中心 100% 裁切，共 6 张证据图；所查区域未见明显绿偏、洋红偏、方向错误或块状断层。暗部较深如实保留，没有据此修改 M9 影调；此抽查不构成校准色彩、全图逐像素或原始细节等价证明。范围与观察在 `visual_spotcheck_001.json`。

整轮处理约 **475.266 秒**，各张发布前时间 **6.3971–23.3553 秒**；不是冻结负载下的性能排名。Tk 心跳执行 **19638 次**，最大观测间隔约 **1.188 秒**。进程 RSS 50 ms 采样峰值最大 **2.906 GiB**，CUDA allocator 保留峰值 **1010 MiB**。此轮不能证明长期内存稳定或所有交互都即时响应。

## 结果与证据

新目录：`E:\m9output\native_matrix_fix_20261017`。**`photos` 中是这 38 张正式 JPEG**；`查看修复结果.html` 提供全部缩略图和正式成片链接。没有覆盖用户已有结果。

- `original_user_log.txt`：原始 38 张失败日志。
- `inventory_001.json`、`profile_preflight_001.json`：精确选定的 38 个文件、元数据、原始零矩阵、白平衡、格式和新矩阵检查。
- `before_fix_001.json`、`after_decode_001.json`：实际错误复现及 Z7 II 逐像素回归依据。
- `upstream_reference_001.json`：官方源码链接、系数及标准 Nikon 解码器函数的版本间哈希对照。
- `pytest_001.log`、`batch_render_001.json`、`render_console_001.log`、`final_checks_001.json`：测试、逐张处理、Tk 队列和完整性记录。

此次只对这些标准压缩 Z6 III 文件验证成功，不扩大为 HE/HE*、其他未知相机或校准色彩精度的承诺。场景 CCT 仍是单矩阵近似，现有 M9 风格保持原样。旧窗口中已经导入的 Python 模块不会自动更新，需关闭旧窗口、重新运行 `launch_nef_gui.cmd` 才会载入修复；本批成片已保存，无需为了验证再次渲染。
