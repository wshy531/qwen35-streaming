# qwen35-streaming

基于 Qwen3.5-0.8B 官方权重和原始模型实现的流式视频理解实验。

当前阶段：已经建立原始基线、视频抽帧/时间戳、增量混合状态续接和可视化问答原型，正在验证有界历史证据的保存、选择与读取。未进行 SFT，尚未实现无限历史或默认缓存裁剪。

## 当前进度（2026-10-14）

[PR #1](https://github.com/wshy531/qwen35-streaming/pull/1) 和 [PR #2](https://github.com/wshy531/qwen35-streaming/pull/2) 已合入 `main`；本次说明以主分支提交 `ba22776` 为基准。

- **文件视频流式原型已可用**：顺序采样，成对编码新增帧，续接注意力 KV、卷积和递归状态；问答在独立缓存分支生成，不写回视频主历史。
- **运行与展示已接入**：支持文件快速重放、按视频 PTS 墙钟播放、单最新帧槽及丢帧统计，网页显示画面、回答、时间和 reset。达到主状态容量限额时重置，并明确显示新的记忆起点。
- **已有真实权重验证**：保留同输入原/流式对照、文件来源和约两分钟的重复短片压力结果。后者接入 112 个采样帧、36 次问答、19 次 reset，没有丢帧；它是有限重复播放，不能代表自然长视频或无限期稳定。
- **历史记忆仍在实验阶段**：有界 RGB 帧库、投影特征、事件代表、ToMe 式合并/森林和严格时间查询是显式可选路径或独立诊断；没有统一升级为默认方案，也没有证明总体识别能力提升。

活动实现与运行说明见 [流式实验 README](experiments/36tar/streaming-learning/README.md)；阶段记录见 [实验总纲](experiments/36tar/streaming-learning/实验总纲.md)，候选结果见 [后续实验索引](experiments/36tar/streaming-learning/FOLLOWUP_EXPERIMENTS.md)、[森林效果分析](experiments/36tar/streaming-learning/FOREST_RUNTIME_ANALYSIS.md) 和 [第二素材验证](experiments/36tar/streaming-learning/SECOND_FILE_VALIDATION.md)。实验总纲中的旧路径和阶段描述是交付快照，不是本机目录约定。

协作方低帧率断流修复和 UCF101 验证见 [PR #3](https://github.com/wshy531/qwen35-streaming/pull/3) 及 `wshy/fix-live-sampling-timeout` 分支。PR #3 已合入 `36tar/general-video-demo`；该修复的 45 项回归通过，三视频/六上下文原与流式生成 Token 一致、问答主缓存保持不变；俯卧撑/剃须识别正确，篮球未识别出扣篮。**当前 `main` 尚未包含这次修复和证据**，仍需从实验分支另行整合，不能将其作为主分支新增验收。

## 仓库结构

```text
qwen35_local/                  根目录原始模型/配置基线
scripts/smoke_test.py           根目录官方权重文字/单图验证
tests/                         根目录源码迁移对照测试
experiments/36tar/streaming-learning/
  code/qwen35-streaming/        实验中的只读原始基线
  code/qwen35-streaming-s4/     当前活动流式实现
  scripts/run_video_demo.py    通用视频演示启动器
  scripts/run_live_demo.py     独立文件/可选摄像头来源启动器
  diagnostics/                 显式候选和独立对照实验
  runs/                        固定运行记录与小型文本证据
third_party/                   上游许可证、提交及源码哈希
requirements.txt               根目录固定核心依赖
THIRD_PARTY_NOTICES.md          来源和修改说明
```

源码来自 Transformers **v5.3.0**，提交
`aad13b87ed59f2afcfaebc985f403301887a35fc`。公共加载、生成、处理器等功能由
Transformers 提供；模型及视觉/语言子模块来自本仓库。没有复制整个 Transformers。
模型权重单独下载，不提交到 GitHub。

流式开发应修改 `experiments/36tar/streaming-learning/code/qwen35-streaming-s4`，从其中的本地模型类加载；根目录 `qwen35_local` 保留作原始参考。`runs/**/source_snapshot` 是证据，不是开发入口。

## Windows / RTX 4060 环境

建议 Python 3.11。以下命令在仓库根目录执行，无需激活虚拟环境。
CUDA wheel 需要兼容的 NVIDIA 驱动；用 `nvidia-smi` 查看驱动与显存。

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu126
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip install av==16.1.0 psutil==7.2.2 numpy==2.4.6
.\.venv\Scripts\python.exe -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

CUDA 检查应输出 `True`。如果已经在这个虚拟环境中装过 CPU 版 PyTorch，
先用同一 CUDA 安装命令加 `--force-reinstall` 替换它。
Linux 下将 Python 路径替换为 `.venv/bin/python`。

仅做 CPU 代码验证时，将上面的 `cu126` 换成 `cpu`，不要据此估计 GPU 性能。
初始验证不依赖 FlashAttention、FLA 等额外编译扩展；后续按需要单独引入。

本机已配置的开发环境位于 `D:\AI_project\qwen35-streaming\.venv`：Python 3.11.7、torch 2.10.0+cu126、torchvision 0.25.0+cu126、Transformers 5.3.0、PyAV 16.1.0。RTX 4060 Laptop 已通过实际 CUDA 运算和真实模型运行；独立 UCF101 FP32 测试的 allocated 显存峰值约 3.28 GiB，不是所有配置的显存保证。模型目录为 `D:\AI_project\models\Qwen3.5-0.8B`；其他机器需要自行准备环境与权重。

实验锁文件来自协作者的 Python 3.12 环境，其中 NumPy 2.5.2 不适用于 Python 3.11；本机使用 2.4.6。不要在 Python 3.11 上直接安装整份历史锁文件。

## 验证本地实现（无需下载权重）

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

测试使用同一组随机小模型参数，比较上游与本地实现的文字、合成图像/视频
logits、增量文本缓存，并检查从上游保存的检查点加载到本地类。
这些测试只检查源码迁移正确性，不能代表 0.8B 官方权重的理解能力、显存或速度。

初始化验证（2026-10-09）：Windows / Python 3.11.7 / PyTorch 2.10.0 CPU，
上述 3 项 unittest 全部通过。这是根目录原始实现的历史迁移验证；当前流式实现及真实权重实验位于实验目录，不能用这 3 项代替流式验收。当前本机环境已安装 CUDA PyTorch 和官方权重。

## 使用官方权重（首次运行会下载）

```powershell
.\.venv\Scripts\python.exe -m scripts.smoke_test
.\.venv\Scripts\python.exe -m scripts.smoke_test --image "D:\images\example.jpg" --prompt "请描述这张图片。"
```

脚本默认 CUDA、非思考模式、最多生成 64 tokens；图像缩至不超过 336×336，
处理器再按模型要求处理。它打印并校验模型、视觉和语言子模块的源码路径，
三者必须指向 `qwen35_local`。运行后显示模型 revision 和峰值已分配显存。
默认官方权重 revision 固定为 `2fc06364715b967f1860aea9cf38778875588b17`。
可用 `--model` 指定本地权重目录，或用 `--revision <模型提交SHA>` 覆盖权重版本。
下载缓存由 Hugging Face 管理，也可设置 `HF_HOME`。

## 运行流式视频演示

以下在仓库根目录的 PowerShell 中运行。先进入自己的开发分支；GPU 演示入口会拒绝 `main`、`master` 和 detached HEAD。模型资产须包含经过校验的 `DOWNLOAD_MANIFEST.json`，视频不随仓库上传；将路径换成本机文件。

```powershell
git switch -c my/video-demo
$RepoRoot = (Get-Location).Path
$Python = Join-Path $RepoRoot '.venv\Scripts\python.exe'
Set-Location (Join-Path $RepoRoot 'experiments\36tar\streaming-learning')
$common = @('--model-dir', 'D:/AI_project/models/Qwen3.5-0.8B',
            '--video', 'D:/videos/example.mp4',
            '--question', 'Describe what is happening in the video.',
            '--fps', '1', '--duration', '8', '--max-frames', '8',
            '--answer-tokens', '16', '--first-qa-frames', '4', '--qa-every', '4',
            '--max-tokens', '288', '--wall-seconds', '180')
& $Python scripts/run_video_demo.py --stage preflight @common --out runs/my_video_cpu_01
& $Python scripts/run_video_demo.py --stage gpu @common --preflight runs/my_video_cpu_01 --ui-port 8765 --out runs/my_video_gpu_01
```

CPU 预检通过后再运行 GPU；预检使用随机微型模型，不代表理解能力。两阶段必须使用相同输入和参数，输出目录每次使用新名称。源码、视频或模型资产改变后需要重新预检；GPU 还会检查系统 RAM、空闲显存和计算预算。

运行时打开 `http://127.0.0.1:8765/` 查看画面与回答；结束后查看输出目录中的 `viewer.html`、`summary.json` 和 `events.jsonl`。默认 `--playback fast`，按源时间播放可用 `--playback clock`，修改后须重新生成预检。

独立来源、重复播放和可选历史查询的参数见 [详细运行说明](experiments/36tar/streaming-learning/README.md)。严格时间特征查询是独立候选，见 [文件入口及验收](experiments/36tar/streaming-learning/TIME_FILE_ENTRY.md)。

## 当前限制

- 主混合状态 reset 会丢弃其历史；可选外部证据另外限额和淘汰，不能当作无限记忆或默认 KV 裁剪。
- 取回正确的帧、正常 EOS 和状态隔离通过，不代表回答正确。猫/鸟误认、动作遗漏及无动物画面仍答狗等失败保留；当前没有正式视频基准准确率或训练外泛化结论。
- 多数验证采用固定 FP32 路径，历史 BF16 差异仍保留；不能从安装环境支持 BF16 推断续接数值验证已通过。
- 物理摄像头尚未实测，端到端提速、自然长视频和长期服务稳定性尚未确证；没有进行 SFT 或更换官方权重。

## 修改方式

- 根据工作目标使用根目录基线或活动流式实现的 `qwen35_local`，不使用 AutoModel 自动选择上游实现。
- 修改缓存可从 `Qwen3_5DynamicCache` 入手；注意同时存在 attention KV 和递归状态。
- 当前模型/配置文件独立维护，不运行上游 modular 代码生成器覆盖它们。
- 未修改时应与固定版本上游保持一致；架构改动后保留原始基线并补充针对性测试。
- 不直接修改 `.venv/Lib/site-packages`。不要单独升级 Transformers 而不复核本地代码。

## 工作计划

- [x] 建立官方权重基线并保存固定样例的能力与失败记录
- [x] 加入视频读取、抽帧、时间戳和滑窗问答
- [x] 实现文件视频增量续接、问答隔离、有界 reset 和可视化演示
- [x] 完成有限墙钟播放、过载及重复短片的延迟/显存/状态验证
- [x] 开始外部记忆、事件代表、视觉 Token 合并和时间查询的独立对照
- [ ] 将低帧率来源修复和独立真实数据验证整合到当前主线
- [ ] 用更多独立素材区分基座识别、证据读取和压缩损失
- [ ] 根据对照证据决定 KV/递归状态及记忆候选是否升级默认
- [ ] 验证自然长视频、实际摄像头和更长运行的稳定性
- [ ] 构建流式数据并进行 SFT

## 来源

- [官方权重](https://huggingface.co/Qwen/Qwen3.5-0.8B)
- [原始实现](https://github.com/huggingface/transformers/tree/v5.3.0/src/transformers/models/qwen3_5)
- [第三方许可证及修改记录](THIRD_PARTY_NOTICES.md)
