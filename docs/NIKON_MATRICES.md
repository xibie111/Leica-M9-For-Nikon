# 尼康相机矩阵目录

本次把已取得的 **LibRaw 0.22.1 官方表中全部尼康系数**合入本地数据目录。固定来源版本，不在处理照片时联网下载，也没有升级本机 rawpy / LibRaw。它是相机矩阵目录，不是所有尼康相机及 RAW 编码均已实拍验收的声明。

## 数据与来源

完整机型、别名、原始系数、源文件行号和 SHA-256 位于 [nikon_matrices.json](../m9/data/nikon_matrices.json)。

| 上游表 | 数量 | 方向与用途 |
|---|---:|---|
| `adobe_coeff` 三通道 | 85 条 | 3×3 XYZ→camera；符合输入条件且解码库矩阵全零时用于补全 |
| `adobe_coeff` 四通道 | 8 条 | 4×3 XYZ→camera；完整保留，当前 RGB Bayer 流程不使用 |
| `simple_coeff(3)` | 1 组 | E880/E900/E990 共用的 3×4 相机通道→`rgb_cam` float 矩阵；只保留来源数据 |

93 条 `adobe_coeff` 包含 151 个明确型号拼写，折叠大小写和空白后为 124 个唯一键。型号来自上游相机清单、Nikon 别名表以及明确的共享系数规则，覆盖 D 系列、Z 系列、Nikon 1 和 COOLPIX 的已有表项。共享矩阵的多台相机不等于多个独立标定。

四通道条目为 COOLPIX 2500、4500、5000、5700、700、800、950、995，保留全部 12 个系数，不能截取前三行用于 RGB。`simple_coeff` 的方向和缩放不同，也不能混入 XYZ→camera 表。

固定版本的相机清单共 110 个型号：104 个有 `adobe_coeff`，另 3 个有上述 `simple_coeff`。COOLPIX 3700、S6 和 Coolscan NEF 在这两张表中没有可用系数，程序不会为它们编造矩阵。清单中部分相机使用 NRW 或特殊传感器；数据收录不会扩大当前 `.NEF`、三色 RGB、2×2 Bayer 的输入范围。

一手来源：[矩阵表](https://github.com/LibRaw/LibRaw/blob/0.22.1/src/tables/colordata.cpp)、[机型归一化与别名](https://github.com/LibRaw/LibRaw/blob/0.22.1/src/metadata/normalize_model.cpp)、[相机清单](https://github.com/LibRaw/LibRaw/blob/0.22.1/src/tables/cameralist.cpp)。实际字节取自已校验的官方 rawpy 0.27.1 发行源码包内的 LibRaw，版本头为 0.22.1；压缩包哈希和路径也保存在 JSON。尚未取得 0.22.2 源码作逐项比较，不声称后续版本没有新增或改变条目。

提取数据按上游可选许可中的 CDDL-1.0 分发，完整许可、版权和提取说明见 [NOTICE](../m9/data/LibRaw-NOTICE.txt)。

## 生效条件

已有非零 LibRaw 矩阵继续经过原有验证与处理；不会用目录覆盖。只有完整 4×3 矩阵恰为全零时，才查询目录：

1. 厂家必须准确为 Nikon / Nikon Corporation；型号仅折叠大小写、连续空白和可选的 `NIKON ` 前缀，再匹配目录中的明确名称。没有运行时型号前缀猜测，`D50X`、`D5000garbage` 等不会借用邻近机型。
2. 只接受三通道 9 系数条目。系数按上游 `float / 10000.f` 换算，不应用表中的 black/maximum 字段。
3. 已由 LibRaw 成功解码为 RGB Bayer；原始 CFA 单条带结构满足 Compression=1 或 34713、PhotometricInterpretation=32803、SamplesPerPixel=1，条带偏移为有限非负整数且头部完整。34713 条带的 JPEG XS `ff10ff50` 标记被拒绝，HE/HE* 不在该回补范围。
4. 继续执行有限性、矩阵秩、D50 白点、as-shot 白平衡与场景 CCT 检查。未知、非零损坏、NaN 或不合布局的矩阵仍失败。

日志及处理记录明确写出采用的机型表项、实际型号、上游版本、URL、换算后矩阵及条带签名。失败不会自动转 DNG。DNG 路径维持原有双光源矩阵插值，M9 后续 AE、影调、StyleTrans、colorfix、LeicaFilter 和模型门限维持现有实现。

## 复现与回归

提取器会校验三个源文件的固定 SHA-256，剔除注释中的旧矩阵，再检查全部条目、型号覆盖和别名冲突。项目根目录可执行：

```powershell
.\.venv\Scripts\python.exe tools\update_nikon_matrices.py --source-dir .work\rawpy-0.27.1\external\LibRaw --check
.\.venv\Scripts\python.exe -m pytest tests\test_nikon_matrices.py tests\test_raw_input.py -q
```

本轮真实回归证据保存在 `E:\m9output\nikon_matrix_catalog_20261017`：

- 全套自动化 **347 passed in 11.18s**，其中矩阵目录与 RAW 输入测试 247 项；生成器 `--check` 和独立源码/别名/许可审查通过。
- Z7 II `DSC_0150`、Z6 III FX `DSC_0015` / DX `DSC_1628` 修改前后完整 native 线性 RGB SHA-256、cam2srgb 和 CCT 全部相同，源文件哈希不变。
- `DSC_1628` 另跑完整 native / original / auto 流程，实际使用 CUDA、StyleTrans=on。输出 JPEG 与前一轮成片逐字节相同，照片目录没有 JSON 旁文件。
- 保护的前端、M9 单张入口、批量、GUI、风格、滤镜、网络后端与依赖文件哈希不变。

其余机型由数据完整性、机型匹配、三通道矩阵数值与入口拒绝条件测试覆盖，没有逐机实拍样本；矩阵来源正确也不等于校准色彩精度已经验证。此前 38 张 Z6 III 的实测仍见 [原验收记录](NEF_Z6III_VALIDATION.md)。更新后须关闭旧窗口，再运行 `launch_nef_gui.cmd` 载入新目录。
