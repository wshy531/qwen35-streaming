# 后续记忆候选实验：复现入口与固定证据索引

本文件是截至本次交付的实验索引，不是新的综合汇报或持续进度主记录。持续主记录为用户本机实验总纲；本胶囊中的总纲/架构文件是阶段快照。默认主实现仍在 `code/qwen35-streaming-s4`，以下四个 `diagnostics` 模块均为显式候选，未替换默认入口、权重或混合状态核心。

| 独立模块 | 用途与固定结果 | CPU / 真权重证据 |
|---|---|---|
| protocol_readout_v1 | 同8真实帧对：原/当前整段/特征读取首logits与完整生成同一，误认同样存在 | 8输入数值控制；2载入24QA；一次RAM门禁拒绝保留 |
| visual_detail_v1 | 两次缩放/直接小图/原像素大图；同48Token8答不变，192Token未救回关键误认 | 24输入数值控制；1载入24QA |
| representative_v1 | 最新对与开组代表分开：first/last/在线中心，同192Token上限而实际用量分报 | 6契约/60随机QA；1载入51QA；受控重复动物窄主体7/12、4/12、7/12 |
| representative_holdout_v1 | 不重叠采样帧，真实主续接、容量reset、中途问答后继续，复用一次视觉投影 | 3新组合契约/70随机QA；1载入58QA；中心未比first语义更好 |

最近四段实际5载入、88视觉、157QA、1278语言forward、1127生成。正常EOS不是答对，随机权重CPU输出不是能力证据。原/当前同答只限固定输入，不证明任意长视频完全等价。中心选择仅是旧代表与新对的在线候选比较，不是全历史精确medoid、完整StreamForest、语义检索或训练。

## 必须保留的边界

- first/centroid最高实际794368B，last596800B；统一192Token上限不代表同实际内存，候选比旧144Token银行允许更多载荷。
- 留出58QA是29配对，27同生成、2只改变dog标点；5次代表ID不同。样本来自旧视频，只有采样帧与77校准/85评估不重叠，不能称新来源泛化。
- 犬片头和尾部有无动物的字卡，不能根据文件名给dog标签。原答中的 `animal_clip_subject_check` 明示不是逐帧真值，**不得用它汇报帧级准确率**；事前逐对人工检查在 `runs/representative_holdout_cpu_20261013_01/inspection_rubric.json`。
- 无动物卡片答犬、实际犬答horse、取液答coffee、猫答犬、纸巾答擦镜子等失败保留。代表保存/取回正确不代表识别正确。
- 候选主写入/查询不会重编码未来对；源RGB可有限预解码，尚未到达的视觉投影不能进入算法。当前只有文件顺序采样，不是新增摄像头/墙钟实时证明。
- 首代表预检曾在导入前失败：失败后、修复前保存的源码快照已明确标注。后续watchdog在子进程导入前保存bootstrap。80core与旧参考未改；历史失败、RAM拒绝和旧源码版本保留。

## 从个人实验胶囊目录运行

先准备固定revision的本地模型与原素材，模型/视频不在Git中。来源署名沿用胶囊既有素材文档，本轮附原始来源manifest。修改机器路径、源码、配置、素材、题目或分支后必须重新CPU预检，不复用别人的proof。默认CPU前置检查无需GPU；真权重GPU仍受RAM3GiB/GPUfree5.5GiB/转换2GiB和360秒watchdog限制。

```powershell
$python = 'E:/VIT Model/.venv/Scripts/python.exe'
$common = @('--model-dir', 'E:/VIT Model/models/Qwen3.5-0.8B',
            '--reference-run', 'runs/representative_gpu_20261013_01',
            '--branch', '36tar/general-video-demo')
& $python diagnostics/representative_holdout_v1/run_probe.py --stage preflight @common --out runs/my_holdout_cpu_NEW
& $python diagnostics/representative_holdout_v1/run_probe.py --stage gpu @common --preflight runs/my_holdout_cpu_NEW --out runs/my_holdout_gpu_NEW
```

`--branch`换成真实个人checkout分支。`cases.json`和原输入注册表中的绝对媒体路径也要换成本机路径，随后重新预检。模型路径由参数指定；不下载新权重、不改同学文件。

其余入口参数可执行各 `probe.py --help` 查看；`run_probe.py` 包装器另外要求out/stage。完整预算与复现说明见各方案。protocol_readout使用 `--arm both/original/current`；visual_detail使用 `--reference-cpu/--reference-gpu`；代表首轮参考 `runs/event_memory_gpu_20261013_01`，冻结阈值只读校准；留出参考上一代表结果。out必须是新目录，CPU→GPU参数必须相同。

## 原始证据

每组CPU/GPU的proof、预算、源/模型/输入SHA、原答、状态、watchdog及source_snapshot分别存于 `runs`，独立审计目录名为 `*_audit_*`。未复制JPG/PNG/视频/权重/环境/凭据；照片检查保留像素身份与事前rubric。失败源码版本与当前版本不能混用。个人胶囊属性保留冻结字节，避免跨机器换行转换使证据SHA变化。
