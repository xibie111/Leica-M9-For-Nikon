# NEF GPU 执行候选验证

日期：2026-10-18–19。本轮冻结基线为 `d1f31a7d14dfb20a1d0bc8c8d7748fda3e3a5826`，源码、权重、脚本与原始结果保存在 `E:/m9output/gpu_execution_20261018/`。目标是在保留原精度、算式和成片字节的前提下减少执行与传输开销。**Graph、filter、combo 均拒绝进入默认管线，产品已恢复基线，本轮没有新增默认优化。** 下文的 graph/传输入口是已归档实验实现，不是当前产品 API。

## 基线与测量口径

五次独立同配置 A 批量耗时为 **30.5832、29.7362、29.4807、29.3986、29.5398 s**，中位数 **29.5398 s**，`max/min-1` 的跨度为 **4.03%**，未触发预定的超过 10% 时追加一组 5A 的规则。全部结果保留，不能把低于当前控制波动的差额当作已建立的收益。

每个 worker 先处理真实 1628，再关闭该 session；计时批次重新建立共享 session，依次处理 0192/0398/1567/1628。使用 native/original/CUDA、tele、EV=0、D50、JSON off，整批 wall 包含成片发布及 session 关闭。首张模型加载、graph 预热与 capture 均计入，文件/权重哈希在计时前读取，因此“cold”不表示存储冷缓存。观察器统一为 002、2 秒间隔、后台全进程扫描关闭；没有 profiler、产品函数替换或额外同步。

基线证据：`baseline_windows_20261018T151837_383016Z/baseline_results.json`；完整指纹见 `evidence/baseline_hashes.json`。基线五个 arm 的完整输出一致，关闭后 CUDA allocated/reserved 均为 0。后续性能比较仍须核对首张、整批、资源峰值及同期遥测，观察器也有非零开销。

固定硬件为 RTX 4060 Laptop GPU（8188 MiB）、i7-13620H（10 核/16 线程），PyTorch 2.8.0+cu128；matmul TF32 关闭、cuDNN TF32 保持开启。初始五个 A 的整批遥测采样中位数如下，均为 P0；整批包含 RAW、CPU 与 JPEG 阶段，不能把该中位占用解释为网络运行时占用。

| A | GPU SM MHz | 温度 °C | 功耗 W | GPU 占用 % | CPU 报告占用 % |
|---|---:|---:|---:|---:|---:|
| 1 | 2625 | 58 | 20.84 | 2 | 4.0875 |
| 2 | 2625 | 58 | 26.20 | 2 | 2.2813 |
| 3 | 2625 | 57 | 21.43 | 1 | 2.6000 |
| 4 | 2625 | 57 | 24.83 | 2 | 2.1875 |
| 5 | 2625 | 57 | 22.91 | 2 | 3.6594 |

五个 A 的进程 import 中位数为 1.4006 s，cold 1628 单张为 5.3677 s（其中模型加载 1.2636 s）；进程已运行、会话重新加载的 0192 首张为 9.3606 s（批次模型加载 1.1933 s）。这些阶段口径分别保留，不能将不同照片的 cold/first 差额认作启动成本。初始批次峰值 RSS 为 3087699968 bytes，CUDA allocated/reserved 为 556702720/1059061760 bytes；逐 arm min/median/max 和后续候选的相同字段均见 `evidence/all_arm_telemetry_001.json`。CPU 报告值未经独立校准，限制见末节。

## Colorfix CUDA Graph 实验：拒绝进入默认

仅对 batch 1、NHWC `(1,544,544,6)` 的 colorfix 网络计算 capture/replay。独立持久 GPU 输入以原 NHWC 分配，再使用原 NCHW view 和 stride；图输入/输出地址不会被 high/low 的共享 byte 输入缓存替换。FP32、模型、权重、TF32 设置和原网络算式均不改变，legacy `run` API 保持原路径。

首次调用先在侧 stream 执行三次预热，current stream 等待后 capture；图内只有 `net(t)`。输入归一化、host/device 拷贝、非有限值检查和独立乘/加、ties-to-even round、clip 与 uint8 转换均在图外。返回数组拥有独立存储。capture/replay 失败清空 scratch 并明确报错，不静默改走 eager；KeyboardInterrupt/SystemExit 清理后保留原异常类型。

graph 和缓冲区归 backend/session 所有，`release_scratch()` 先释放图输出、输入、graph 和通用输入缓存，再由 session 清理网络、GC 与 CUDA cache。内部 `use_colorfix_graph` 仅用于候选/控制比较，不增加用户开关；`_colorfix_capture_count`、`_colorfix_setup_seconds` 是无额外同步的诊断字段，不能用 replay-only 时间代替首张及整批结果。

实现依据：[PyTorch 2.8 CUDA Graphs](https://docs.pytorch.org/docs/2.8/notes/cuda.html#cuda-graphs)、[graph 内存管理](https://docs.pytorch.org/docs/2.8/notes/cuda.html#graph-memory-management)。固定地址、静态形状与控制流、侧 stream 预热、保留图输入/输出生命周期均按该语义处理；没有将 `.item()`、CPU 读取或数据相关分支放入 capture。

短测真实 ColorFixNet 的两组输入 eager/graph **float32 位模式和最终 uint8 字节均一致**，并覆盖 stride、固定地址、high/low 形状切换、返回值独立及非有限值拒绝。graph/既有严格 byte 路径/session 共 **42 passed（3.92 s）**，另有实际预热流程的 CPU 模拟中断清理 **2 passed（1.38 s）**。证据为 `evidence/graph_short_tests_20261018T152353_384762Z.json`、`graph_interrupt_tests_001.json`；这证明短测契约，不证明完整成片性能收益。

完整五对加末尾 A 的 **44 张批量 + 11 张 cold JPEG** 均严格一致，每个新 session capture 一次，关闭后 CUDA allocated/reserved 均为 0。不过 `(A-B)/A` 的五个配对耗时差额为 **+0.0197%、+1.0447%、−0.1062%、−1.0712%、−4.8145%**，中位数 **−0.1062%**，未超过六个 A 的 **10.0899%** 跨度，也没有 style 阶段收益。cold 首张中位数由 5.3028 增至 5.4979 s，process warm / session cold 首张由 9.2120 增至 9.3204 s；批次首张 setup 为 0.0939–0.1005 s，fresh-process cold setup 为 0.1109–0.1532 s。

峰值 CUDA allocated 从 **556702720 增至 575763968 bytes**，reserved 从 **1059061760 增至 1570766848 bytes**，RSS 峰值也略增。因此保留一致性结论，拒绝把该实验作为默认优化。实现和测试已归档到 E: 证据，仓库已恢复基线，冻结实验快照为 `variant_graph_001`；拒绝记录见 `evidence/graph_summary_001.json`，全部配对与尾部控制见 `paired_graph_20261018T154044_494936Z/paired_results.json`。输出一致不能抵消首张成本及额外资源，也不构成速度收益证明。

## LeicaFilter GPU 传输实验：拒绝进入默认

候选保留原 LUT 与 vignette 两套独立条带边界：LUT 预算 `2^22` 像素、vignette 预算 `2^20` 像素，各自按原宽度计算行数。LUT 结果留在 GPU；不足一个 vignette 条带的余行用独立 clone carry 衔接，carry 小于一个 vignette 条带，避免 view 持有整个旧 LUT block。只将完成的 vignette 结果传回 CPU，不上传整幅照片作为常驻输入。

LUT 累加顺序、float32 中间结果、全图坐标、noise/vignette 公式和最终 clip 均沿用原实现；不为了合并传输而把两套边界改成同一组。实验入口为内部 `_gpu_transport`，原 CPU、shading off 和未启用候选路径保留。两项 CUDA 短测通过（**2 passed、6 deselected，3.03 s**），见 `evidence/filter_cuda_tests_001.json`；短测本身不证明端到端收益，完整测量如下。

完整五对加末尾 A 的 **44 张批量 + 11 张 cold JPEG** 严格一致，关闭后 CUDA allocated/reserved 均为 0。中间 GPU→CPU→GPU 往返确已减少，filter 阶段的批量耗时中位数由 **2.1441 降至 1.8442 s**，RSS、CUDA allocated/reserved 峰值也下降；cold 首张中位数由 5.2814 降至 5.2165 s，process warm / session cold 首张由 9.0826 降至 8.9556 s。

不过配对整批差额中位数仅 **+1.74093%**，未超过本轮六个 A 的 **2.95563%** 跨度，更未超过预定噪声参考 `max(初始5A跨度, 本轮6A跨度)` = **4.02927%**。因此记录阶段与资源改善，仍按相同门槛拒绝进入默认，不能将搬运减少直接等同稳定的端到端收益。全部样本及五个配对差额见 `evidence/filter_summary_001.json`，完整记录见 `paired_filter_20261018T154941_923502Z/paired_results.json`。

## Combo 与最终选择

combo 同样完成五对加末尾 A，**44 张批量 + 11 张 cold JPEG** 严格一致，session 关闭后 CUDA allocated/reserved 均为 0。配对差额中位数为 **+5.96637%**，但六个 A 的耗时范围达 **28.82065–50.32541 s**、跨度 **74.61580%**，其中 A3 = 50.3254 s。异常 A3 和其他全部样本均保留，不剔除、不据此重算较好看的收益；该组不能建立稳定收益，combo 也拒绝进入默认。汇总为 `evidence/combo_summary_001.json`，原始结果为 `paired_combo_20261018T155747_464024Z/paired_results.json`；同期遥测完整保留，异常原因仍不能唯一归因。

| 实验 | 配对差额中位数 `(A-B)/A` | 噪声参考 | 默认采纳 |
|---|---:|---:|---|
| graph | −0.1062% | 10.0899% | 拒绝 |
| filter | +1.7409% | 4.0293% | 拒绝 |
| combo | +5.9664% | 74.6158% | 拒绝 |

这些是完整保留样本的观察差额，不是可承诺的提速。最终 `evidence/select_default.py` 选择 baseline；三份产品文件及相关既有测试恢复至 d1 基线，两份新增实验测试从仓库移除。实验实现/测试归档于 `archived_candidates_001/`，三份冻结 variant 和全部原始结果继续保留；操作指纹与选择记录见 `evidence/default_selection_001.json`。不会把没有胜出的实验 flag、缓冲区或路径留在默认产品中。

## 45MP 完整成片验证

baseline、graph、filter、combo 分别在四个独立进程中建立新共享 session，运行 native/DNG 两路，共 **8 份 5504×8256 Display P3 JPEG**。本轮重新生成两份 baseline 参考；六份候选分别按相同 route 对照，完整 JPEG、压缩数据、RGB、ICC、EXIF payload/值/实际 TIFF 类型与管线语义全部严格一致。全部 JSON off、源文件哈希不变；这项顺序运行仅验正确性，不作速度推断。

每个 variant 两路共加载一次 session，low/high/colorfix 各一次模型缓存 miss；high 调用 2 次、colorfix `run_u8` 调用 374 次（每 route 187 个 tile，不把 graph 的 Python 预热/capture 次数当 tile 数）。graph/combo capture 一次，release 后图和输入/输出 scratch 均清空；四个 session 关闭后 CUDA allocated/reserved 全为 0。证据为 `acceptance45mp_{baseline,graph,filter,combo}_001/acceptance.json` 四份记录。

## GUI、遥测与最终回归

combo 与最终仓库 baseline 两套 GUI 均通过：各完成正常轮 **3/3**，下一轮 **1/2** 在当前照片完成后停止/关闭，另一项保持 pending。各四张 JPEG 与本轮 d1 参考的完整文件、压缩数据、RGB、ICC、EXIF 类型/值和管线语义严格一致。每轮模型加载一次，三网各一次 miss；combo 每轮 capture 一次并先释放 scratch，再 closed（CUDA allocated/reserved=0）、finished，窗口在 worker 退出后销毁。没有 callback 错误或 Tcl 警告；两轮最大心跳间隙分别为 combo 52.7/49.5 ms、baseline 59.1/45.4 ms。Tk 根窗口 withdraw，但使用实际控件、事件循环和默认 processor；这证明功能/生命周期，不是人工视觉 QA。

GUI 记录为 `acceptance_gui_{combo,baseline}_001/acceptance.json`。45MP 与 GUI 合计 **16 份新 JPEG**，汇总见 `evidence/acceptance_summary_001.json`；`acceptance_process_audit_001.json` 确认该验收创建的 Python 进程全部退出。首次 45MP baseline 的 shell console 中文解码乱码被原样保留，直接写出的 UTF-8 `acceptance.log` 和 JSON 完整；后续仅给 child 设置 `PYTHONIOENCODING=utf-8`。该问题不影响成片，也没有被隐去。

性能阶段合计 **38 个完整 arm、190 张 JPEG（152 batch + 38 cold）**，跨组逐源成片指纹一致、runtime 配置相同，全部 arm 的 cold/batch 关闭后 CUDA allocated/reserved 为 0，见 `evidence/all_candidates_summary_final.json`；逐 arm 遥测见 `all_arm_telemetry_001.json`。combo A3 与尾部 A 的 frontend 为 17.3811/6.5406 s、raw 为 7.2585/4.4307 s、jpeg 为 7.3382/4.2789 s，多个阶段共同变慢；两者整卡温度中位数同为 57°C。A3 系统 CPU 报告中位数约 0.669% 异常低，未经独立校准，不能用来裁决物理空闲、调度饥饿或特定 CPU 原因。

遥测是 WDDM 整卡、2 秒采样，利用率/功耗平均窗口并非阶段隔离；CIM 是名义频率，Windows 进程 CPU delta 未校准，命令子进程 CPU 也未计全。观察器 body 最大 wall 延迟 **1.3559 s** 等慢样本完整保留，不声称零开销。具体原因仍为 **UNKNOWN**，不据这些关联指认 GPU 候选、有效核数、过度订阅或唯一的热/电源/后台负载原因。

恢复基线后的最终回归以 `M9_RUN_STRICT_SPEED_CUDA=1` 一次运行，**528 passed（15.95 s，无 skip/fail；进程 wall 16.839 s）**，包含 522 项常规及 6 项 CUDA。三份产品的 py_compile 与 git diff-check 均 exit 0，`m9/tests` diff 为空，39 个冻结文件指纹均与 baseline 一致。实际记录为 `final_regression_001/regression.json`；未将同一组测试重复计为两次回归。全部原始慢样本、实验快照、归档实现与验收证据继续保留，默认产品保持 d1。

最终代码、文档与证据索引的 SHA256 收据为 `evidence/final_receipt_001.json`。
