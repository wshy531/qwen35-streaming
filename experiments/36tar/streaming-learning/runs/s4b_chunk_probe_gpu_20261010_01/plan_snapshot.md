# 后备语言计算块64→32：单一候选预登记

2026-10-10。分阶段计时显示短调用开销大；本地torch_chunk_gated_delta_rule源码固定chunk_size=64、补齐后63步三角递推，18个GDN层每次多Token调用均用该后备函数。源码线索不是实测耗时归因，本实验不预先声称能提速。

仅在临时上下文将18层的chunk_gated_delta_rule callable包裹为固定32，原权重/源码/默认64不变，单Token递归路径不变。计算块是算法执行粒度，不是丢弃历史、KV压缩或视频滑窗。拒绝非torch后备实现，退出/异常还原全部callable。CPU微型契约和真实config预算核验先行；强制0调用预算测试还原。

同一真实前四洗手帧/192×256/115Token/候选实际视觉特征：64与32每种1轮预热+3正式轮，整段一次与提示+第一对+第二对三次增量分别测同步墙钟/CUDA事件，原始值全留。两个预热轮比较32完整对64完整的全部logits/四种缓存；每种增量末对对本模式完整；问句及预定单Token分支对64分支，主状态逐位不变。数值门槛保持原严格logits max≤0.01/mean≤0.0001/RMS≤0.0003、allclose atol/rtol1e-4，四种缓存同allclose。失败就保留，不扩大容差或只看argmax。

预算：模型1次，本机8GiB、FP32 IEEE/math SDPA；1视觉（2时间单元候选100内部Linear）、36语言（8轮×4主调用+2模式各2分支调用）、≤612被包装多TokenGDN调用（34×18）、0生成，600秒。预计allocated≤5.5GiB，载入前free≥5.5GiB/RAM≥3GiB、转FP32前≥2GiB。额外留存特征≤8MiB，语言结果/参考/比较计入总峰值；仅保留64参考及当轮，结束释放。无新依赖/集群/训练，不设为默认。

启动器scripts/run_chunk_probe.py --experiment probe --stage preflight/gpu；冻结CPU/源/资产/方案/启动器SHA。若通过，只把32保留为独立性能候选；更广输入、BF16、跨环境和长期状态仍需验证。
