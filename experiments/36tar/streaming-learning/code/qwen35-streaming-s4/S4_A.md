# S4-A：混合状态续接实验快照

进度主记录在项目根目录`实验总纲.md`；本文仅说明2026-10-09这次实验和接口边界。

基于main提交`61e8de673fe9abb7281f803c1d1c94dc65f6a064`，个人分支`36tar/s4-mixed-state`。原模型/配置来源和许可证仍见`THIRD_PARTY_NOTICES.md`、`third_party/upstream.json`。

## 改动及用途

原GatedDeltaNet只在单Token续接时使用历史状态，多Token会重建卷积状态并忽略递归initial_state。本分支让多Token也接入历史卷积与递归状态，保留原单Token路径，不新增参数、不训练。

`streaming.hybrid.HybridSequence`是单样本、无padding、显式位置的推理接口。append只提交新增Token或已编码embedding；fork共享模型权重并独立复制KV、卷积、递归四类真实状态；reset清空四类状态、Token/RoPE计数和模型rope_deltas。容量上限是提前拒绝，不是KV裁剪。显式embedding必须提供四轴位置，第一轴为已见Token连续编号，另外三轴为原视频位置。

它还没有接收原始帧、处理跨成对帧patch、构造真正聊天提问或生成答案的完整协议。推理中途异常的回滚/并发/训练也不在本接口保证范围内。

## 真实结果及限制

- 原3项适配测试和新增12项混合状态测试全部通过；FP32随机微型模型的2→2/1→3 logits最大误差从约0.00397/0.00430降到5.96e-8，所有最终状态在预登记容差内。
- 官方Qwen3.5-0.8B/BF16/SDPA：7组整段/分块对照只有video_last_single通过数值门槛。其余6组失败，最大logits差1.15625，fork后缀也数值失败。最后argmax全部相同不能替代通过。
- fork主状态逐位不变，reset与新实例logits/四种状态逐位相同，超容量在forward前拒绝且状态不变。这些状态管理结果通过。
- CPU事后算子定位发现BF16 L2归一化随切片形状产生0.00390625差异，先FP32归一化的随机算子对照把最终状态差缩至约1.79e-7。这不是GPU根因证明，也不是已修复GPU模型。官方第一层CPU小误差不能代表整模型。

完整数据在项目runs/s4_*目录；没有真实答案/视频能力提升、在线视觉、无限历史或集群部署证据。

后续固定4组GPU定位排除了“只改q/k归一化为FP32即可修复”假设。逐Token递归参考在真实首层CPU上逐位一致，于是添加默认关闭的`text_config.streaming_recurrent_reference`实验开关；29项CPU通过，但原10组真实GPU回归仍失败，最高logits差1.375，仍只有末尾单Token组通过。开关保留用于复现研究，未启用为默认。详细统计/原整段输出偏移见项目根目录`逐层精度诊断与递归参考结果.md`，不能把首层CPU或默认关闭的候选写成已修复。

进一步四组整语言模型精度诊断：默认块递归路径的20/98Token FP32 IEEE+SDPA math整段/分块，logits和四类最终状态均通过严格参考容差（最大logits差约5.63e-5）。原BF16完全复现失败，仅禁BF16低精度中间归约也失败；同输入重放显示首层输出投影有独立的形状相关数值差。模型默认精度未改，该轮尚未做视频/fork/reset/容量FP32真实权重回归，不能据文字两组称S4或BF16通过。精度控制/trace代码是独立诊断模块；详见项目根目录`高精度整模型诊断方案.md`、`高精度整模型诊断结果.md`。

随后独立`scripts.fp32_regression`完成原10项FP32真实权重参考：七种文字/预编码视频分块、fork与主续接、reset及容量拒绝通过，max logits约1.05e-4，严格allclose与四类状态通过；视觉也FP32但只编码一次，两路径共用特征。BF16默认失败不改。后续CPU只验证当前4帧的成对像素/Token拼接和因果位置，不是官方在线视觉或答案证据。详细方案/结果见项目根目录`FP32混合状态回归方案.md`、`FP32混合状态回归结果.md`；S4-B真正在线成对帧/问答仍待实现，设计见`在线成对帧与问答协议方案.md`。

S4-B随后新增`streaming.online`同步成对帧原型和`scripts.online_experiment`独立整段/真实问答参照。首轮CPU12项通过，真实GPU8项仅5项通过：四帧主前缀/问题继承KV/奇帧EOF仍超严格容差；在线与整段答案同为CAT/ONE且主状态不变，不能称语义或数值全部通过。聊天EOS随后修订，当前13项CPU通过，修订未运行GPU。三个新文件的初版精确快照保留；详见项目根目录`在线成对帧问答结果与复现.md`、`在线问答失败定位与后续控制.md`。旧基线/模型默认精度未改，下一步先共享实际成对特征控制，不直接重跑旧预算或放宽容差。

## CPU复现

安装`requirements-baseline.lock.txt`中的固定依赖；其中torch为本次Windows/CUDA构建，其他平台选择对应官方wheel，不把本地锁当作集群已安装。

在本仓库目录执行，输出必须为未存在的新目录：

```powershell
python -m scripts.chunk_experiment --stage cpu --out C:/path/to/new_run
python -m scripts.chunk_experiment --stage diagnostic --out C:/path/to/new_diagnostic
python -m scripts.precision_diagnostic_cpu --model-dir C:/path/to/Qwen3.5-0.8B --out C:/path/to/new_precision
```

最后一个命令只读官方第一层约20.11MiB权重及所需词向量，不加载整套权重到CPU。模型目录须包含固定revision配置/processor、safetensors与项目DOWNLOAD_MANIFEST.json；权重不随代码或交接包上传。

官方GPU复现使用项目根目录`scripts/run_mixed_state.py`，必须引用通过的CPU及匹配的preflight，核对自己的分支、输入/源码/权重manifest哈希，执行10组/≤256Token/0答案生成/600秒。失败按原门槛保留，不反复换容差使结果通过。

S4-B共享实际特征固定控制 scripts.shared_features_control：在线视觉输出/完整嵌入/四轴位置逐位共享，5个真实FP32严格组通过；两个独立整段视觉偏移控制仍失败并复现旧数值误差。此固定例支持语言续接/问答隔离/EOF，不追溯修改首轮5/8或BF16失败。EOS在两回答止于248046，语义仍CAT/ONE错误。方案与结果见项目根共享视觉特征控制方案.md/共享视觉特征控制结果与复现.md；本次无训练/推送。

S4-B scripts.protocol_capability_control核对同画面2/4帧×Q1/Q2：原make_window_inputs与当前完整协议逐Tensor相同；只读原模型固定8例BF16默认SDPA与FP32 IEEE/math均2帧主体CAT/4帧HANDS、数量ONE。原模型也有这些错误，不能用旧4帧主体正例与新2帧主体/数量题证明改造导致语义退化。8例是执行完成而非准确率8/8；旧BF16续接和整段视觉偏移失败保留。详见项目根同画面输入与能力控制方案.md/同画面输入与能力控制结果.md；后续在线同四帧Q1/触发配置，未训练/推送。

2026-10-10：scripts.four_frame_online完成同四帧截止Q1主体公平对照，在线/独立共享实际特征参照均HANDS及IDs[72141,50,248046]，主前缀/问答严格、重复问答与reset重放逐位一致4组通过。首GPU因新诊断BatchFeature.to改变CPU模板导致设备cat错误保留，仅修诊断独立device副本并重做CPU/真实GPU；模型/在线接口未改。该片段重复不是新准确率，原两帧CAT/数量ONE/BF16及视觉偏移失败保留。详见项目根在线四帧主体公平对照方案.md/在线四帧主体公平对照结果.md；后续明确触发/容量的同步演示待设计，未训练/推送。

2026-10-10 finite synchronous demo: new streaming/demo.py policy waits for four real frames per epoch, then asks every two. CPU nine contracts and real 23-frame clean_hands replay: 12 new-pair vision calls, 7 isolated QA, 3 pre-encoding capacity resets preserving both current frame IDs, 44 language calls/21 generated tokens; all seven answers HANDS with chat EOS. One explicit EOF duplicate counts only one real frame and earns no extra QA. Four main cache fields remain bitwise unchanged per QA; reader/hooks/main cache cleanup checked. FP32 IEEE/math, finite sequential file replay only; no live camera, long-run or new-video generalization claim. Old BF16/numerical failures and original core sources retained. Project-level Chinese plan/result and experiment master document record the exact scope and provenance.

2026-10-10 visual counterfactuals: new scripts/scene_pair_control.py tests four controlled six-frame early/late CAT/HANDS splices and two tail-only controls. CPU full protocol plus 8 strict shared-input language references passed. Real FP32: current-subject 4/4 at four frames and 4/4 at six; FIRST 3/4, paired same-terminal early-subject criterion only 1/2. hands_to_cat FIRST wrongly CAT in both online and shared-input full-language reference; preserved. All 8 strict logits/four-state/IDs controls and 16 online main-isolation checks passed; 14 vision/102 language/58 generated tokens, one load. Original source PTS/IDs and derived experiment PTS/IDs are separate. This is not natural continuous event footage, original-unmodified-model evidence, live inference or reliable long-term memory. Old executable sources and failures retained; project experiment master records next steps.

2026-10-10原模型FIRST补充：独立原未改模型完整六帧视觉编码四例完成。BF16/FP32两设置在hands_to_cat均错误CAT，cat_to_cat均正确CAT，IDs均[27280,248046]与旧在线同一。原反例无需S4改造即可出现，不代表一般准确率或所有中间计算相同。原模型只读，新增独立original_first_control诊断；两记录接口失败快照保留。详见项目根原模型早期主体对照结果.md，进度主记录仍为实验总纲。

2026-10-10早期四帧原FIRST控制：截去旧六帧末两帧，原BF16/FP32均早期手HANDS、猫CAT，四例正确。原六帧手→猫错误CAT保留；前缀可识别，但截短改变长度和末尾内容，不能归为唯一末帧机制或S4缓存故障。新prefix_first_control独立诊断复用旧助手，原模型不改。固定事实见项目根原模型早期四帧对照结果.md，进度仅实验总纲。

2026-10-10原六帧等长末尾FIRST：同前四手、同六帧长度和非像素5项Tensor，只末两像素不同。原BF16/FP32均手→手HANDS正确、手→猫CAT错误，IDs与旧在线对应相同。支持本固定例的末尾内容影响，不能定位唯一末帧/注意力/视觉/递归机制；原模型未改。实际接口6项Tensor，逐键证明保留；根原模型六帧等长末尾对照结果.md保存事实，进度仅实验总纲。
