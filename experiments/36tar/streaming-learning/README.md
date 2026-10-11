# 36tar：流式视频架构学习与实验交接

这是36tar的独立实验工程，本轮开发分支为 `36tar/general-video-demo`。外层仓库原有文件保持原样；本目录保留 `code/`、`scripts/`、`runs/` 布局，方便双方阅读、复现与决定下一项改造。权重、真实视频、虚拟环境和凭据不入库。

[原PR #1](https://github.com/wshy531/qwen35-streaming/pull/1) 已由协作方合并；本轮修改通过新的个人分支和PR审阅，main不直推。
[本轮草稿PR #2](https://github.com/wshy531/qwen35-streaming/pull/2) 包含通用入口、按PTS墙钟播放和同输入原模型公平对照。最新149项CPU/17来源契约通过；文件GPU结果见本段报告，未打开摄像头；第六段显式短片循环播放约120秒、112帧/36回答/19reset、无丢帧，allocated峰值3.270GiB；此前第五段12例真实采样起点对照已完成，洗手方向稳定、猫细节谨慎复核；新增12例洗脸误答图像/时间诊断两臂生成一致，单图变化有影响、所试时间处理未修复原错误。此前14例咖啡/洗脸及先后错误继续保留。未合并。

先读 [实验总纲](实验总纲.md)，再读 [架构设计入口](架构设计.md)。总纲是唯一进度主记录；其余方案、报告和历史快照不替代最新状态。旧记录中的本机绝对路径是实验来源，不是新机器上的运行路径。

## 阅读入口

| 内容 | 入口 |
|---|---|
| 当前通用视频入口 | [运行与约定](#通用视频演示入口) / [设计与预算](通用视频演示设计与验收.md) |
| 本轮CPU与真实权重结果 | [通用视频结果及失败回答](通用视频演示结果.md) |
| 墙钟播放、单槽丢帧与时间观测 | [预登记预算](墙钟播放与丢帧验收方案.md) / [正常和过载结果](墙钟播放与丢帧结果.md) / [播放实现](code/qwen35-streaming-s4/streaming/playback.py) |
| 两素材14例同输入原/流式诊断 | [方案与预算](多视频同输入公平对照方案.md) / [逐例真实答案](多视频同输入公平对照结果.md) / [独立输入核验](code/qwen35-streaming-s4/streaming/fair_inputs.py) |
| 四帧洗脸误答：图片与时间文字诊断 | [方案](洗脸误答单因素定位方案.md) / [全部真实答案](洗脸误答单因素定位结果.md) / [固定12例](configs/action_ablation_cases.json) |
| 实时来源CPU软件接口 | [方案](实时输入接口验收方案.md) / [实际结果与待验边界](实时输入接口结果.md) / [采集进程](code/qwen35-streaming-s4/scripts/capture_worker.py) |
| 两分钟显式循环播放与资源/reset | [预算](两分钟循环播放验收方案.md) / [全部回答及边界](两分钟循环播放结果.md) / [资源分组比较](runs/repeat_playback_delivery_20261011_01/comparison.json) |
| 两素材整秒/半秒真实采样对照 | [预登记](多视频采样稳定性方案.md) / [全部真实答案](多视频采样稳定性结果.md) / [12例配置](configs/sampling_stability_cases.json) |
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
            '--max-frames', '8', '--answer-tokens', '16', '--first-qa-frames', '1', '--qa-every', '4',
            '--max-tokens', '256', '--wall-seconds', '180')
.venv/Scripts/python.exe scripts/run_video_demo.py --stage preflight @common --out runs/my_video_cpu_01
.venv/Scripts/python.exe scripts/run_video_demo.py --stage gpu @common --preflight runs/my_video_cpu_01 --ui-port 8765 --out runs/my_video_gpu_01
```

运行中打开 `http://127.0.0.1:8765/` 查看当前画面和最近回答；结束后服务关闭，打开输出的 `viewer.html` 查看末画面及最近8条回答。页面显示当前接收视频时间、实际回答对应时间、记忆从何时开始、重置次数和历史Token数。`events.jsonl` 保存全部有限事件，`summary.json` 保存显存峰值、全程重放时间和包括问答/解码的append耗时；不要只看语言阶段耗时。

默认保持比例居中补黑边，视频写入不计算词表logits，问答只投影末位置；回归证据接口仍可输出完整logits。支持任意长度视频，按 `--duration` 与 `--max-frames` 明确截断，实际样本数可以小于预算，不要求23帧；奇数末帧在EOF/限制结束时补对。每次问题自包含。默认首问需要4帧；上面的 `--first-qa-frames 1` 显式复现前轮早问配置，允许单帧EOF问答。两帧输入曾误答咖啡，增加首问帧数尚未证明能普遍修复能力。

GPU默认允许任意非main/master、非detached的协作分支，不绑定36tar；可重复 `--allow-branch your/name` 收紧名单。CPU证明绑定源码及启动器、问题/参数、视频SHA和全部模型资产；改变任一项须重新预检。CPU阶段回答来自随机微型权重，页面会明确标注，不证明视频理解。所有运行在已声明显存/RAM/forward/生成/墙钟预算内；不适合绕过门禁做无限输入。

默认 `--playback fast` 按顺序尽快重放。另可选择下述有限墙钟播放，尚无摄像头、跨reset记忆或长期运行证明。使用新入口时无需调用绑定个人分支的历史启动器。

## 按实际视频时间播放与丢帧

```powershell
$clock = @('--model-dir', 'E:/VIT Model/models/Qwen3.5-0.8B',
           '--video', 'E:/VIT Model/datasets/natural_commons_v1/clean_hands.webm',
           '--question', 'What are the hands doing?', '--playback', 'clock',
           '--fps', '1', '--duration', '12', '--max-frames', '12',
           '--answer-tokens', '16', '--first-qa-frames', '4', '--qa-every', '4',
           '--max-tokens', '256', '--wall-seconds', '90')
.venv/Scripts/python.exe scripts/run_video_demo.py --stage preflight @clock --out runs/my_clock_cpu_01
.venv/Scripts/python.exe scripts/run_video_demo.py --stage gpu @clock --preflight runs/my_clock_cpu_01 --ui-port 8765 --out runs/my_clock_gpu_01
```

读帧线程按解码得到的真实PTS发布采样画面，模型线程独占全部混合状态。待处理槽容量为1：推理来不及时，新画面替换尚未领取的旧画面，并累计丢弃样本数；正在推理的帧不撤回。这里的丢帧统计针对 `--fps` 选择的样本，不是视频的全部原始帧。模型仍成对编码，但过载时一对画面的实际时间间隔可能变大；报告记录最大间隔。

页面分别显示源画面位置、接入模型流程的位置、回答实际依据的时间、答案出现的墙钟时间和滞后，以及记忆起点和reset。接入流程的单帧可能尚在等待配对，不能当作已完成视觉编码。CPU预检仍快速遍历全部样本，不按随机微型模型速度模拟GPU丢帧；真实运行核对完整源指纹、参数、源码和资产，实际接入/丢帧日程保存在 `events.jsonl`。

正常实测12个样本全部接入，3次回答滞后0.551–0.633秒。另加 `--consumer-delay-seconds 1.5` 可独立预检和运行显式过载条件：本轮12样本中8接入、4丢弃，2次回答滞后2.370–3.568秒。这个参数是压力注入，不能据此估算正常速度；过载末答错误地说成洗脸。两轮allocated峰值均约3.266GiB，槽内待处理画面最多144KiB；后者不是进程总内存。见 [完整结果及边界](墙钟播放与丢帧结果.md)。

## 独立文件读帧入口（当前使用）

当前用户选择视频文件，暂不启用摄像头。先用真实文件和随机微型模型生成当前接口proof，再运行真实GPU；文件SHA/完整选定源指纹必须同一，按原视频PTS播放，忙时容量1最新帧槽可替换旧帧。完整CPU149/17来源契约通过；本段真实结果与载入前失败见[文件来源报告](文件实时来源结果.md)，预算见[验收方案](文件实时来源验收方案.md)。

```powershell
$fileLive = @('--input-kind', 'file', '--video', 'E:/VIT Model/datasets/natural_commons_v1/cat_play.webm',
              '--model-dir', 'E:/VIT Model/models/Qwen3.5-0.8B', '--fps', '1', '--duration', '9',
              '--max-frames', '12', '--question', 'What is the cat doing?', '--answer-tokens', '16',
              '--max-tokens', '288', '--first-qa-frames', '4', '--qa-every', '2', '--wall-seconds', '90')
.venv/Scripts/python.exe scripts/run_live_demo.py --stage preflight @fileLive --out runs/my_file_cpu_01
.venv/Scripts/python.exe scripts/run_live_demo.py --stage gpu @fileLive --preflight runs/my_file_cpu_01 --ui-port 8765 --out runs/my_file_gpu_01
```

本机真实GPU重试已完成：9帧/2问答/1reset/0丢帧，allocated峰值3.270GiB；两答滞后均0.609秒，主状态逐位不变、源指纹与当前CPUproof同一。首次载入前RAM拒绝保留；本次不打开摄像头，不能推断动作准确率或长期自然视频稳定。

素材文件需自行按既有署名说明准备，不在Git包中。每次使用新的输出目录；资源不足会在载入前拒绝。文件模式不需要capture-check，不打开摄像头。

## 实时摄像头软件入口（本段仅CPU验收）

146完整CPU/14新来源契约通过，合成源真实子进程12帧/4QA/1reset，问答主状态逐位不变；断流、卡住、取消和父退出清理通过。没有打开真实设备，也未新增GPU。预检命令不打开摄像头：

```powershell
$live = @('--model-dir', 'E:/VIT Model/models/Qwen3.5-0.8B', '--device', 'Integrated Camera',
          '--capture-size', '640x480', '--capture-fps', '15', '--fps', '1',
          '--duration', '12', '--max-frames', '12', '--question', 'Describe what is happening now.',
          '--answer-tokens', '16', '--max-tokens', '288', '--first-qa-frames', '4', '--qa-every', '2', '--wall-seconds', '90')
.venv/Scripts/python.exe scripts/run_live_demo.py --stage preflight @live --out runs/my_live_cpu_01
```

以下摄像头入口仅可选，当前用户选择暂不启用。用户另行选定拍摄范围后才能执行；两步会实际打开摄像头，不当本段已运行结果：

```powershell
.venv/Scripts/python.exe scripts/run_live_demo.py --stage capture-check @live --preflight runs/my_live_cpu_01 --out runs/my_camera_probe_01
.venv/Scripts/python.exe scripts/run_live_demo.py --stage gpu @live --preflight runs/my_live_cpu_01 --capture-check runs/my_camera_probe_01 --ui-port 8765 --out runs/my_live_gpu_01
```

设备检查仅3帧、不载入模型/不保存RGB；GPU要求新CPUproof、同配置设备probe和原RAM/显存门禁。采集子进程不加载Torch，最大RGB256×256×3，native输入buffer8MiB、只视频不录音，父端最新帧槽容量1；阻塞超时可终止采集进程。会话时间依据接收后的monotonic时刻，原设备PTS另存，页面scope明确非传感器曝光到答案延迟。当前无实际驱动断流/摄像头GPU/场景质量证据，不能据软件测试宣布这些通过。所有历史proof是源快照，改过源码后必须重新预检。

## 两分钟显式重复压力输入

这是已有短片的重复工程测试，来源PTS与合成会话时间分开记录，不当自然长片或跨reset记忆。循环期间不主动重置，容量策略仍自动清空历史；空隙不复制补帧。单次播放默认参数保持。重新预检后可显式启用：

```powershell
$repeat = @('--model-dir', 'E:/VIT Model/models/Qwen3.5-0.8B',
            '--video', 'E:/VIT Model/datasets/natural_commons_v1/clean_hands.webm',
            '--question', 'What are the hands doing?', '--playback', 'clock',
            '--fps', '1', '--duration', '120', '--max-frames', '120',
            '--repeat-count', '5', '--repeat-period', '25', '--answer-tokens', '16',
            '--first-qa-frames', '4', '--qa-every', '2', '--max-tokens', '288', '--wall-seconds', '240')
.venv/Scripts/python.exe scripts/run_video_demo.py --stage preflight @repeat --out runs/my_repeat_cpu_01
.venv/Scripts/python.exe scripts/run_video_demo.py --stage gpu @repeat --preflight runs/my_repeat_cpu_01 --ui-port 8765 --out runs/my_repeat_gpu_01
```

本机112源样本全部接入，36问答/19reset、无丢帧。实际播放119.766秒，含载入122.445秒。主缓存每问逐位不变，19个新记忆起点均与保留成对帧一致。峰值allocated3.270GiB/reserved3.605GiB；去掉前两epoch、按事件种类与同Token长度比较，allocated极差0、RSS最大极差5.03MiB。前/中/后40秒回答滞后中位0.669/0.7065/0.7325秒，没有排队累积到多秒；这不证明无限期稳定。

GPU summary.json包含同步后allocated/reserved/RSS标量曲线；events.jsonl保留来源循环/实际PTS、会话时间和全部答案。网页显示这些时间和当前记忆起点，最近回答只保留8条。循环衔接附近仍有倒咖啡等误答，理解能力未因资源通过而修复，摄像头/真实长视频未测。首次完整CPU沙箱HTTP拒绝保留；02的132CPU及29接口通过，新GPU只有一次。此轮改过入口源码，所有历史proof仅为历史证据，另机/改变参数必须重新预检。

## 两素材采样起点稳定性

固定洗手0/4/8秒、猫0/2/4秒，每一起点比较phase0/0.5，共12个四真实帧输入。原模型与流式独立视觉，12例生成IDs一致；洗手6例方向稳定，猫姿态/动作文字随画面变化，玩球细节未充分证实。6对有5对文字不同，不计作失败率；旧洗脸反例保留，默认采样/时间/状态核心不改。124CPU/8新接口/12微型输入通过，流式12QA主状态逐位不变；峰值allocated约3.266GiB，仍无长期稳定或摄像头证明。

在本目录用新run_id复现，两GPU命令顺序执行：

```powershell
$sampling = @('--model-dir', 'E:/VIT Model/models/Qwen3.5-0.8B',
              '--video-dir', 'E:/VIT Model/datasets/natural_commons_v1', '--wall-seconds', '240')
.venv/Scripts/python.exe scripts/run_sampling_stability.py --stage preflight --arm both @sampling --out runs/my_sampling_cpu_01
.venv/Scripts/python.exe scripts/run_sampling_stability.py --stage gpu --arm original @sampling --preflight runs/my_sampling_cpu_01 --out runs/my_sampling_original_01
.venv/Scripts/python.exe scripts/run_sampling_stability.py --stage gpu --arm streaming @sampling --preflight runs/my_sampling_cpu_01 --out runs/my_sampling_streaming_01
```

每臂1载入/12答/最多192生成/240秒，原12视觉204语言、流式24视觉240语言，3GiB RAM/5.5GiB显存门禁。目标时间之后第一真实帧、取第四帧即止；不重复补帧或伪造时间。改变源码/素材/参数需新CPUproof。详见[逐例结果](多视频采样稳定性结果.md)和[机器比较](runs/sampling_stability_delivery_20261011_01/comparison.json)。

## 固定四帧的洗脸误答定位

最新12例在原未改模型与流式臂重新运行，所有生成IDs相同。稀疏[7,8,10,11]画面仍误答洗脸；保持时间文字/位置不变，只换第1图7→8或第2图8→9后，两边均回答洗手/揉皂。稀疏画面不变的6个时间标签干预仍洗脸；归零/均匀标签只改变部分措辞，不作为修复。四格图像×时间交叉支持本固定例对图像组合敏感，不能推广成全部动作能力或内部机制结论。

116完整CPU/8新增接口/12微型输入控制通过。独立两臂allocated峰值3.2430/3.2659GiB，流式12QA主状态逐位不变。首GPU因RAM不足在载入权重前拒绝，释放内存后原门槛重试完成；失败记录保留。模型核心和默认时间协议未改。详见[逐例结果与解释](洗脸误答单因素定位结果.md)及[机器可读比较](runs/action_ablation_delivery_20261011_01/comparison.json)。

在本目录用新run_id复现，两GPU命令顺序执行；默认只读此前成功的两臂文本报告来核实际输入锚点：

```powershell
$action = @('--model-dir', 'E:/VIT Model/models/Qwen3.5-0.8B',
            '--video', 'E:/VIT Model/datasets/natural_commons_v1/clean_hands.webm', '--wall-seconds', '240')
.venv/Scripts/python.exe scripts/run_action_ablation.py --stage preflight --arm both @action --out runs/my_action_cpu_01
.venv/Scripts/python.exe scripts/run_action_ablation.py --stage gpu --arm original @action --preflight runs/my_action_cpu_01 --out runs/my_action_original_01
.venv/Scripts/python.exe scripts/run_action_ablation.py --stage gpu --arm streaming @action --preflight runs/my_action_cpu_01 --out runs/my_action_streaming_01
```

每臂1载入/12答/最多192生成/240秒，原臂12视觉/204语言，流式24视觉/240语言，载入前至少5.5GiB显存及3GiB RAM。固定配置是诊断集，含重复图片/合成时间，不是实时播放入口；来源PTS与标签分开记录。实际核对6类Tensor，生成相同不作为准确率/数值等价判据。使用新机器须重新预检，不能直接借用归档proof。

## 已有证据与边界

此前公平对照在同两素材及明确拼接14例上重新运行两臂，生成Token全部一致，流式14次QA主状态逐位不变。原模型也在两帧回答咖啡、稀疏四帧回答洗脸；固定选择题均5/8，先后1/2。108CPU/11新接口及14微型输入控制通过；首个GPU被JSON坐标tuple/list比较拒绝，新增往返回归后重新预检/重跑，原失败保留。有限生成一致不是内部数值/长期等价，更不是14/14理解正确。

从本目录使用新的run_id复现公平对照（需本机两个已署名视频及固定模型）：

```powershell
$fair = @('--model-dir', 'E:/VIT Model/models/Qwen3.5-0.8B',
          '--video-dir', 'E:/VIT Model/datasets/natural_commons_v1', '--wall-seconds', '240')
.venv/Scripts/python.exe scripts/run_fair_video.py --stage preflight --arm both @fair --out runs/my_fair_cpu_01
.venv/Scripts/python.exe scripts/run_fair_video.py --stage gpu --arm original @fair --preflight runs/my_fair_cpu_01 --out runs/my_fair_original_01
.venv/Scripts/python.exe scripts/run_fair_video.py --stage gpu --arm streaming @fair --preflight runs/my_fair_cpu_01 --out runs/my_fair_streaming_01
```

两个GPU命令顺序执行。原模型臂14视觉/238语言，流式臂27视觉/280语言；各1载入、14答、最多224生成/240秒，载入前GPU5.5GiB/RAM3GiB门禁。默认读取本目录已有咖啡/洗脸历史文本记录来固定实际输入，源码/方案/配置/素材/历史/资产改变必须重新预检。原模型没有共享流式视觉特征；固定14例定义在 [配置](configs/fair_video_cases.json)，默认示例是诊断集，通用演示仍可输入任意视频和问题。

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
../../.venv/Scripts/python.exe -m unittest tests.test_local_model tests.test_mixed_state tests.test_recurrent_reference tests.test_stream_window tests.test_stream_session tests.test_action_inputs tests.test_sampling_inputs -v
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

接续优先级由最新总纲决定：更长墙钟输入/全程资源与reset记忆边界 → 摄像头 → BF16、32/64和长期状态。墙钟与单槽丢帧已有有限证据，模型能力错误仍需定位。合并通过PR审阅，不直接推送main。
