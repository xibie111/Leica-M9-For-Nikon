# NEF 矩阵、几何与验证依据

核查日期：2026-10-17。测试环境为 rawpy 0.25.1、LibRaw 0.21.4。矩阵方向先用生成的 CFA DNG 核对，随后在用户提供的真实 Nikon Z7 II NEF 上验证了 Adobe 转换及两路 RAW 解码。本文不将 RAW 解码检查等同于后续网络与成片画质验收。

后续已把 Z6 III 的单机回补扩展为固定 LibRaw 0.22.1 全部尼康矩阵目录，详见 [尼康矩阵目录](NIKON_MATRICES.md)。下文保留最初方向核查和 Z6 III 修复时的 0.22.0 来源记录；当前程序使用 0.22.1 目录，仅补全明确机型的全零矩阵，并支持标准单条带 CFA 的 Compression=1 或 34713。原 Z6 III 系数和真实样片输出保持相同。

## 矩阵方向的一手依据

1. [rawpy 官方绑定源码](https://github.com/letmaik/rawpy/blob/main/rawpy/_rawpy.pyx)：检查的是 rawpy 0.27.1 发行源码包中 `_rawpy.pyx` 第 916–929 行。`rgb_xyz_matrix` 将底层 `cam_xyz` 逐项原样返回，没有转置：

   ```python
   matrix[i,j] = self.p.imgdata.rawdata.color.cam_xyz[i][j]
   ```

2. [LibRaw 官方 0.21.4 源码包](https://www.libraw.org/data/LibRaw-0.21.4.tar.gz)：与安装的 LibRaw 版本相同。`src/utils/utils_dcraw.cpp` 第 282–312 行的 `cam_xyz_coeff` 先进行以下乘法，再逐行归一化并求逆，建立相机到 RGB 的变换。因此 `cam_xyz` 的行对应相机通道，方向为 **XYZ → camera**：

   ```cpp
   cam_rgb[i][j] += cam_xyz[i][k] * LibRaw_constants::xyz_rgb[k][j];
   ```

3. 同一 LibRaw 源码包的 `src/tables/colordata.cpp` 第 1003–1004 行提供 Nikon Z 7 的矩阵，注释明确同时覆盖 `Z 7_2`。Z7 II 使用当前输入由 LibRaw 给出的有效矩阵；后续新增的 Z6 III 缺项回补规则见下节。[上游相机表](https://github.com/LibRaw/LibRaw/blob/master/src/tables/colordata.cpp)

来源获取记录：LibRaw 源码直接取自官网，SHA-256 为 `6be43f19397e43214ff56aab056bf3ff4925ca14012ce5a1538a172406a09e63`。rawpy 0.27.1 官方发行源码经 [Tuna 包镜像](https://pypi.tuna.tsinghua.edu.cn/packages/f3/ae/c1c7816ed3f3cbf7ca284a37371243500f4eb39b6a32f7368b585c4ef5c4/rawpy-0.27.1.tar.gz)取得，SHA-256 为 `3194d64ff690ac945e1a43237edae8a18f1f493751924de1ae2bcef473c0fb79`；这是源码传输镜像，不是第三方解释。在线主分支行号可能变化，上述行号指当次下载的固定发行源码。

## Z6 III 缺失相机表项的适配

用户后续提供的 38 张失败 NEF 均为 `NIKON Z6_3`、固件 2.00。安装的 rawpy 0.25.1 / LibRaw 0.21.4 能解包这些文件，但 `rgb_xyz_matrix` 为全零。该版本相机表没有 `Z6_3`，而 [LibRaw 0.22.0 第 1036–1037 行](https://github.com/LibRaw/LibRaw/blob/0.22.0/src/tables/colordata.cpp#L1036-L1037) 提供了独立的 Z6_3 系数：

```text
11206  -4286  -941
-4879  12847  2251
 -745   1654  7374
```

按上游第 1901–1903 行的 `float / 10000.f` 语义换算，保持 XYZ→camera 方向。两个版本的 `nikon_read_curve()`、`nikon_load_raw()` 函数内容核对相同，支持对当前标准压缩样片只补回缺失的颜色矩阵；没有据此宣称所有新机型解码都兼容。来源和函数哈希记录在 `E:\m9output\native_matrix_fix_20261017\upstream_reference_001.json`。

回补只在厂家为 Nikon、型号准确为 `NIKON Z6_3`、完整 4×3 矩阵恰为全零、三色 RGB Bayer、标准单条带 CFA NEF（Compression=34713）时生效。读取原始数据条带开头，拒绝 JPEG XS 的 `ff10ff50` 标记及不完整数据；该物理字节标记来自 [上游 HE 检测](https://github.com/LibRaw/LibRaw/blob/0.22.0/src/metadata/tiff.cpp#L2254-L2267)，与 TIFF 大小端无关。HE/HE* 未列为支持范围。

已有有效矩阵优先且不替换；未知机型的零矩阵、非零奇异矩阵、NaN、非法布局仍失败。后续有限性、秩、D50 白点、as-shot 白平衡和场景色温检查没有放宽。日志显式说明使用官方回补，记录包含版本、URL、矩阵和原始条带签名。没有换用 Z6/Z6 II 矩阵、单位矩阵或自动 DNG 路径，也没有升级整个解码依赖。

真实 Z7 II `E:\m9input\DSC_0150.NEF` 修复前后的完整 native 线性 RGB SHA-256、cam2srgb 和 CCT 相同；证据为 `before_fix_001.json` 与 `after_decode_001.json`。本次只修相机元数据兼容，不修改后续 AE、曲线、StyleTrans、滤镜或调校参数。

## 当前环境的运行核对

`tests/test_raw_input.py::test_installed_libraw_reads_embedded_xyz_to_camera_and_orientation` 使用 tifffile 生成包含非对称 Nikon 矩阵、AsShotNeutral 与 CFA 的 DNG，并通过安装的 rawpy 0.25.1 / LibRaw 0.21.4 解码。读回的 `rgb_xyz_matrix[:3]` 与写入的 XYZ→camera 矩阵逐项相等；TIFF Orientation=6 由 LibRaw 执行旋转。测试检查实际输出和相机到 sRGB 的变换，而不只检查属性名称。

两路使用同一个既有 D50 白平衡与相机色彩变换约定。native 路径用 LibRaw 单矩阵和 as-shot neutral 估计场景白点，再估计 CCT，记录明确标为近似。DNG 保留既有双光源矩阵插值。矩阵方向正确不等于两路颜色已经实拍一致；两路仍须同图对照。

## RAW 范围与 DNG 默认裁切

LibRaw 0.21.4 的生成 DNG 实验确认：相同的 120×88 目标范围，放在 128×96 ActiveArea 内作为 DefaultCrop，默认解码仍可输出 128×96；直接把 ActiveArea 设成该范围则输出 120×88。仅比较解码宽高可能误拒绝有效转换。根预览和 RAW SubIFD 的尺寸也可能不同，因此校验识别 CFA SubIFD。

当转换后尺寸不同时，只接受能够证明完整包含源 active pixels 的精确映射：两边 RAW 传感器尺寸、方向相同，DNG 的整数 DefaultCropOrigin/Size 在同一传感器坐标中恰好等于源 active rectangle，且该矩形完全在转换后的 active area 内。实现只执行轴交换、反向和整像素裁切，不重采样；记录原坐标。没有这项证明、存在分数坐标或转换器已裁掉源有效像素时明确报错。不会缩小源范围来强行匹配。

单元测试覆盖八种 LibRaw 方向标志、边界与像素位置、分数裁切、已丢失像素和不同传感器尺寸；生成 DNG 的实际 LibRaw 解码也覆盖 DefaultCrop 表达。

### 真实 Z7 II 的相机照片范围

真实样片发现 LibRaw 的 `width/height` 包含照片外侧边框，不能直接当作相机成片尺寸。FX 样片的 RAW 为 8288×5520，相机 inset crop 为 `[16,8,8256,5504]`；DX 样片为 5440×3616，相机 inset crop 为 `[16,8,5408,3600]`。相机内嵌 JPEG 尺寸也与上述照片范围一致。

Adobe 14.0.1 转换后的 FX DNG 保留 8288×5520 ActiveArea，RAW SubIFD 的 DefaultCropOrigin 为 `(16/1,8/1)`、DefaultCropSize 为 `(8256/1,5504/1)`，与源 NEF 的 LibRaw crop 字段一致。因此 NEF 入口以源相机 inset crop 定义照片范围：native 在 demosaic 后按方向进行整数裁切，dng 路径按同一源范围核对 DefaultCrop 并裁切。原有独立 DNG 入口的默认行为不变。

实测 `DSC_0149.NEF`（FX、Orientation=8）两路均返回 5504×8256；`DSC_0366.NEF`（DX、横图）两路均返回 5408×3600，全部像素有限。此验证仅执行 RAW 解码与色彩元数据检查，没有运行 StyleTrans。完整实拍对照、网络细节与资源测量见成片验收报告。

## 转换器与失败边界

命令使用独立临时目录与参数数组：`-c -p0 -d <临时目录> <NEF>`，没有线性化、缩图或有损选项。本机 Adobe 14.0.1 已完成上述真实 FX/DX NEF 转换，输出通过 RAW 校验；模拟进程测试另覆盖中文空格路径、非零退出、超时、无输出，以及成功或解码失败后的临时文件清理。

转换后还检查 CFA 单通道马赛克、Compression=7、ColorMatrix1、AsShotNeutral、曝光、方向和尺寸。普通 TIFF EXIF 的 FNumber 与 ExposureTime 有理数按分子/分母读取；非法曝光、非整数或越界方向、非物理场景白点和非有限色温均明确失败。选定路由失败不会改走另一条。

### 真实慢快门样片的曝光元数据变化

`DSC_0308.NEF` 的 ExposureTime 为 `10/13` 秒，源文件没有 ShutterSpeedValue；Adobe DNG 的 ExposureTime 为 `8/10` 秒，ShutterSpeedValue 为 `321928/1000000` APEX。两者不是相等的有理数，不能直接把它们视为同一个数值或扩大曝光比较容差。

实际逐像素比较确认，两者的完整 5520×8288 uint16 马赛克相同，CFA 和黑电平相同，马赛克 SHA-256 均为 `99cc179bfb0df3f586cc6e97406479dbb05d8196631094b3534c98e2a3e70dae`。这证明转换器改变了曝光时间元数据表示，而没有改变本次捕获的 RAW 样本；不据此推断真实物理快门时间一定为 0.8 秒。

仅在曝光时间不一致时额外执行这项完整马赛克核对，并要求转换后的 ExposureTime 与 APEX 在其原始有理数分母定义的半个量化步内一致。如果源文件本身有 APEX，也检查其自洽性。任一像素、布局、CFA、黑电平不同，或缺少/不一致的 APEX，仍明确失败。光圈和 ISO 比较规则没有放宽。

M9 测光始终使用原 NEF 的 ExposureTime、FNumber、ISO、EV100；0308 修复后仍使用 `10/13` 秒、EV100 `5.362217815913079`。处理记录同时保留转换后的 0.8 秒、两边原始有理数、APEX 和马赛克校验证据。真实 DNG RAW 解码已验证返回 8256×5504；完整网络处理的重试结果由成片验收报告记录。
