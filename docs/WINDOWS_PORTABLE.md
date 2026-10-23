# Windows NVIDIA 免安装候选包

分发采用独立 Python 3.12.4 x64 目录、自带 Tcl/Tk 和锁定 wheel 的方式；不复制开发环境 `.venv`，不安装或注册系统组件。程序入口为 `启动程序.cmd`，完整环境自检为 `环境诊断.cmd`。运行时不执行 pip，也不下载模型。

首个修订候选版本为 `0.1.0-candidate.2`。它仍是本地候选包：第二台电脑离线验收和第三方再分发权利未确认。实际结果见包内 `验收报告.md`；不要把自检、同机对照或 ZIP 校验当作跨设备验收。

## 构建

在项目根目录使用 Windows x64 Python 3.12。先建立新的时间戳构建目录，冻结源码、工作区差异、依赖、输入和模型/调色资源哈希。原有工作区和旧候选不覆盖。

1. 根据 `packaging/windows/runtime-archives.json` 下载官方 Python 嵌入式 ZIP、对应 Tcl/Tk MSI、Microsoft VC++ 运行库 EXE 到独立 `downloads` 目录。核对锁定 SHA-256；微软下载别名更新后必须重新审查版本，不能直接绕过校验。
2. 将 `packaging/windows/requirements-runtime.lock` 对应的 Python 3.12 Windows x64 wheels 放到 `downloads/wheels`。Torch 从 `https://download.pytorch.org/whl/cu128` 获取，其余从 PyPI 获取；保留下载来源与哈希。
3. 运行构建命令，将下列示例路径替换为本次新目录：

```powershell
.\.venv\Scripts\python.exe -B tools/build_windows_portable.py build `
  --output "E:\m9output\本次构建\发布候选 空格" `
  --downloads "E:\m9output\本次构建\downloads" `
  --version 0.1.0-candidate.2
```

构建器按 `source-manifest.json` 的文件清单复制源码、权重和资源；MSI/EXE 只当作只读文件容器解包，不执行安装动作。依赖通过 pip 从本地 wheelhouse 安装。包内 `_pth` 固定模块搜索位置，GUI 和原生解码子进程均使用包内解释器。

构建器排除生成的开发命令入口和仅供编译扩展使用的 `torch/lib/*.lib`，保留推理 DLL、Python 代码、元数据和第三方声明。包内不支持编译 PyTorch 自定义扩展。固定版本、源码哈希和构建输入保存在 `RELEASE.json`、`FILE_MANIFEST.json` 及构建目录的 `build-inputs.json`。

完成测试后，将脱敏的 `验收报告.md` 放入候选目录，再执行封包：

```powershell
.\.venv\Scripts\python.exe -B tools/build_windows_portable.py seal "E:\m9output\本次构建\发布候选 空格\LeicaM9ForNikon-0.1.0-candidate.2-win-x64-nvidia"
```

`seal` 生成完整文件清单、内部 SHA256SUMS、ZIP64 压缩包及外部 ZIP SHA-256；已有 ZIP 会拒绝覆盖。封包后还须检验 ZIP CRC、解压后哈希和启动。

## 入口与错误

- 只有 `m9.portable` 设置 CUDA 默认值并限制发布界面的设备选项；现有 `m9.gui`、API、CLI 默认值维持兼容。
- 发布入口清除外部模型、后端及 Python/Tcl 路径覆盖，固定使用包内模型。允许 `M9_DNG_CONVERTER` 指向用户另装的 Adobe 转换器。
- 启动时核对关键文件和依赖版本，导入 RAW/图像模块并进行 CUDA 张量检查；诊断另外核对全包哈希、Tk 和三个网络的合成输入。
- 日志写入 `%LOCALAPPDATA%\LeicaM9ForNikon\logs`，包含启动异常和 GUI 队列消息。日志可能含照片路径，应在反馈前检查。
- 缺少模型、CUDA 不可用、显存不足、解码或转换失败均显示错误；不会自动改设备、模型、输入方式或分辨率。

## 验证与边界

```powershell
$env:M9_RUN_STRICT_SPEED_CUDA = "1"
.\.venv\Scripts\python.exe -B -m pytest tests -q
```

实拍工具 `tools/validate_windows_portable.py` 提供 `matrix`、`arm`、`compare`、`gui` 子命令。`matrix` 用冻结源码和包内解释器逐一运行相同输入；`compare` 核对 JPEG 字节、像素、尺寸、ICC、EXIF和原片哈希。`gui` 使用真实 Tk 回调和照片验证批量、失败重试、停止、关闭及重名防覆盖，同时检查资源清理；该自动化不等同视觉验收。

Z6 III 的 DNG 路径在本轮本机 Adobe 转换器上出现 `Cannot init cloud download cache`，冻结源码环境也同样失败；对应实拍验收未完成。Z7 II 与原生 NEF 路径单独记录，不使用替代路径掩盖错误。

对外试用前，须完成未装 Python/CUDA Toolkit 的第二台 NVIDIA 电脑断网、普通用户、真实照片验收，并确认原程序、模型、调校资源和运行库的适用分发条款。最低驱动、内存和显存要求依据真实设备测定；本机通过不能推广为所有 NVIDIA 显卡支持。
