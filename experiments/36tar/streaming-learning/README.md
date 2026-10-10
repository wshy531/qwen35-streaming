# 36tar：流式视频架构学习与实验交接

这是截至2026-10-10的独立实验工程，位于个人分支 `36tar/s4-mixed-state`。外层仓库原有文件保持原样；本目录保留已验证的 `code/`、`scripts/`、`runs/` 布局，方便双方阅读、复现与决定下一项改造。没有上传权重、真实视频、虚拟环境或凭据。

先读 [实验总纲](实验总纲.md)，再读 [架构设计入口](架构设计.md)。总纲是唯一进度主记录；其余方案、报告和历史快照不替代最新状态。旧记录中的本机绝对路径是实验来源，不是新机器上的运行路径。

## 阅读入口

| 内容 | 入口 |
|---|---|
| 合作仓库结构与计划解释 | [架构解读](合作仓库架构与计划解读.md) |
| 机制与未解决问题 | [通俗复盘](流式视频架构学习复盘与未解决问题.md) |
| 当前语言计算块验证 | [32/64交错、多长度结果](语言计算块交错扩展结果.md) |
| 视觉一致性与性能背景 | [视觉扩展与性能复盘](视觉扩展与性能探索复盘.md) |
| 有界增量缓存 | [HybridSequence](code/qwen35-streaming-s4/streaming/hybrid.py) |
| 成对帧和隔离问答 | [OnlinePairSession](code/qwen35-streaming-s4/streaming/online.py) |
| 触发、容量与reset演示 | [PairDemo](code/qwen35-streaming-s4/streaming/demo.py) |
| 模型实际改动的审阅差异 | [相对原main的patch](runs/s4b_chunk_interleaved_delivery_check_20261010_02/tracked_changes.patch) |
| 最新严格验证完整证据 | [GPU报告](runs/s4b_chunk_interleaved_gpu_20261010_01/interleaved_summary.json) / [审计](runs/s4b_chunk_interleaved_delivery_check_20261010_02/verification.json) |
| 素材来源及署名 | [ATTRIBUTION](datasets/natural_commons_v1/ATTRIBUTION.md) |

`code/qwen35-streaming` 保留原基线，`code/qwen35-streaming-s4` 是增量实验实现。S4对多Token输入续接卷积与递归状态，另外管理普通注意力KV；问答在fork分支运行，主历史不受回答影响。容量超限需reset或显式拒绝，尚未实现无限记忆、缓存裁剪或训练。递归参考、固定时间单元视觉与32计算块是显式实验候选，不能当作原默认路径已修复。

## 已有证据与边界

固定FP32真实权重混合状态回归、共享特征控制、有限23帧/7问答同步重放已经完成。六组视觉候选完整/成对特征逐位一致；32块最新五组、每组六次交错计时全部满足原严格logits和四类缓存门槛，完整语言耗时减少17.4%–31.2%，分段累计减少26.8%–33.8%。这是本机RTX4060 Laptop、FP32 IEEE/math、torch后备路径的有限测试，不是完整视频问答端到端提速或模型能力提升。

BF16数值失败、原模型的早期画面/顺序/数量错误和审计失败均保留在报告及 `runs/` 中。固定时间单元视觉候选与原整段视觉仍存在数值差。尚无集群部署、SFT、记忆森林、Token压缩或长期摄像头输入的证据。

## 无权重的最小CPU检查

在本目录建立独立环境；Python 3.12是本机验证版本。`requirements-baseline.lock.txt`记录原本机CUDA环境，CPU环境按核心依赖安装，不直接安装其中的cu126条目，也不据此估计GPU速度：

```powershell
python -m venv .venv
.venv/Scripts/python.exe -m pip install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cpu
.venv/Scripts/python.exe -m pip install -r code/qwen35-streaming-s4/requirements.txt
.venv/Scripts/python.exe -m pip install av==16.1.0 psutil==7.2.2
Set-Location code/qwen35-streaming-s4
../../.venv/Scripts/python.exe -m unittest tests.test_local_model tests.test_mixed_state tests.test_recurrent_reference tests.test_stream_window tests.test_stream_session -v
```

从 `code/qwen35-streaming-s4` 到本目录环境的正确相对路径是 `../../.venv`。Linux替换为 `../../.venv/bin/python`。上述测试使用随机微型参数、合成画面，不下载官方权重，也不证明视频理解能力。完整测试集中的online/demo测试另外需要 `models/Qwen3.5-0.8B` 下的官方处理器资产。

## 复现实验

回到本目录（不是外层仓库根目录），按 [基线复现说明](基线复现说明.md) 准备固定revision的模型和 [自然视频说明](自然视频结果与复现.md) 中已署名的素材。本机实测GPU依赖为torch 2.10.0+cu126、transformers 5.3.0；不要覆盖同学的环境。历史 `configs/prototype.json` 是LLaVA旧草案，不能直接执行。

若要复现最新32/64实验，在登记本机预算后，使用两个新的输出目录：

```powershell
.venv/Scripts/python.exe scripts/run_chunk_interleaved.py --experiment interleaved --stage preflight --out runs/my_chunk_preflight_01
.venv/Scripts/python.exe scripts/run_chunk_interleaved.py --experiment interleaved --stage gpu --preflight runs/my_chunk_preflight_01 --out runs/my_chunk_gpu_01
```

先读 [冻结方案](语言计算块交错扩展方案.md)：一次载入、2视觉、314语言、0生成、600秒，载入前至少5.5GiB空闲显存和3GiB可用RAM，额外留存特征限8MiB。只支持当前显式torch后备候选，保留64默认；不得绕过CPU门禁。GPU启动器还核个人分支 `36tar/s4-mixed-state`。其他分支或新方案须先审阅接口约定，不能借用旧机器上的CPU批准。

归档的proof中含旧资产路径、代码SHA和输入指纹，**不能直接用作新机器/新源码的GPU预检**。先产生新的CPU证明，保留成功和失败输出；Git或不同操作系统若改变换行字节，也应重新冻结源码证明。`UPLOAD_SOURCE_MANIFEST.json`记录此次导出内容的原始SHA；历史source_tree/provenance表示各次运行时的版本。

后续优先考虑统一显式32/64选项，再做短问答生成与完整延迟回归；长期输入、能力评测、BF16、压缩/森林和训练分别按证据决定。合并通过PR审阅，不直接推送main。
