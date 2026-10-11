# 36tar：流式视频架构学习与实验交接

这是36tar的独立实验工程，本轮开发分支为 `36tar/general-video-demo`。外层仓库原有文件保持原样；本目录保留 `code/`、`scripts/`、`runs/` 布局，方便双方阅读、复现与决定下一项改造。权重、真实视频、虚拟环境和凭据不入库。

[原PR #1](https://github.com/wshy531/qwen35-streaming/pull/1) 已由协作方合并；本轮修改通过新的个人分支和PR审阅，main不直推。
[本轮草稿PR #2](https://github.com/wshy531/qwen35-streaming/pull/2) 提供通用入口、85项CPU及一次真实权重运行的成功/失败证据，尚未合并。

先读 [实验总纲](实验总纲.md)，再读 [架构设计入口](架构设计.md)。总纲是唯一进度主记录；其余方案、报告和历史快照不替代最新状态。旧记录中的本机绝对路径是实验来源，不是新机器上的运行路径。

## 阅读入口

| 内容 | 入口 |
|---|---|
| 当前通用视频入口 | [运行与约定](#通用视频演示入口) / [设计与预算](通用视频演示设计与验收.md) |
| 本轮CPU与真实权重结果 | [通用视频结果及失败回答](通用视频演示结果.md) |
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

**唯一活动实现是本目录的 `code/qwen35-streaming-s4`**，新通用入口为 `scripts/run_video_demo.py` → `scripts.video_demo`。`code/qwen35-streaming` 只读原基线；`runs/**/script_snapshot.py` 等是运行时证据，不是开发入口；旧根启动器和 `scripts.pair_demo` 是历史固定实验复现工具。双方修改活动实现，并使用新的run_id，避免编辑不同副本。首次上传清单 `UPLOAD_SOURCE_MANIFEST.json` 是1c8141b之前的导出快照，不能当作后续修改的当前校验清单；每次新预检重新生成source_hashes/proof。

S4对多Token输入续接卷积与递归状态，另外管理普通注意力KV；问答在fork分支运行，主历史不受回答影响。容量超限需reset或显式拒绝，尚未实现无限记忆、缓存裁剪或训练。递归参考、固定时间单元视觉与32计算块是显式实验候选，不能当作原默认路径已修复。

## 通用视频演示入口

在本目录运行，准备本机固定revision的模型资产（含DOWNLOAD_MANIFEST.json）和任意本地视频。以下路径需改成自己的；两个stage使用同一视频、问题和参数，新输出目录不能已存在：

```powershell
$common = @('--model-dir', 'E:/VIT Model/models/Qwen3.5-0.8B',
            '--video', 'E:/VIT Model/datasets/natural_commons_v1/clean_hands.webm',
            '--question', 'What are the hands doing?', '--fps', '1', '--duration', '8',
            '--max-frames', '8', '--answer-tokens', '16', '--qa-every', '4',
            '--max-tokens', '256', '--wall-seconds', '180')
.venv/Scripts/python.exe scripts/run_video_demo.py --stage preflight @common --out runs/my_video_cpu_01
.venv/Scripts/python.exe scripts/run_video_demo.py --stage gpu @common --preflight runs/my_video_cpu_01 --ui-port 8765 --out runs/my_video_gpu_01
```

运行中打开 `http://127.0.0.1:8765/` 查看当前画面和最近回答；结束后服务关闭，打开输出的 `viewer.html` 查看末画面及最近8条回答。页面显示当前接收视频时间、实际回答对应时间、记忆从何时开始、重置次数和历史Token数。`events.jsonl` 保存全部有限事件，`summary.json` 保存显存峰值、全程重放时间和包括问答/解码的append耗时；不要只看语言阶段耗时。

默认保持比例居中补黑边，视频写入不计算词表logits，问答只投影末位置；回归证据接口仍可输出完整logits。支持任意长度视频，按 `--duration` 与 `--max-frames` 明确截断，实际样本数可以小于预算，不要求23帧；奇数末帧在EOF/限制结束时补对。每次问题自包含；仅一帧也允许EOF回答，但两帧输入可能理解错误，不能用流程通过冒充准确率。

GPU默认允许任意非main/master、非detached的协作分支，不绑定36tar；可重复 `--allow-branch your/name` 收紧名单。CPU证明绑定源码及启动器、问题/参数、视频SHA和全部模型资产；改变任一项须重新预检。CPU阶段回答来自随机微型权重，页面会明确标注，不证明视频理解。所有运行在已声明显存/RAM/forward/生成/墙钟预算内；不适合绕过门禁做无限输入。

当前是有限、同步顺序文件重放；没有按墙钟播放、摄像头、丢帧队列或跨reset记忆。先做这些工程验证及多视频能力比较，再评估BF16、32/64块与长期记忆。使用新入口时无需调用绑定个人分支的历史启动器。

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

接续优先级由最新总纲决定：通用连续效果 → 墙钟/摄像头与丢帧 → 多视频公平对照 → 全程延迟/长期资源 → BF16、32/64和长期状态。合并通过PR审阅，不直接推送main。
