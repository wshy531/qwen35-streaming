# PR #5 / #6 本机统一整合（2026-10-16）

用户要求在本机处理两份PR，无需协作者更新。统一交付使用 [PR #6](https://github.com/wshy531/qwen35-streaming/pull/6)，完整包含 [PR #5](https://github.com/wshy531/qwen35-streaming/pull/5) 的提交 `80b1292` 与原收尾 `0d864af`。采用合并提交保留双方历史，不强推或删除协作者分支；PR #5 在确认远端包含后关闭。main 仍待用户审核合并。

## 整合结论

低帧率修复在两分支相同。默认 `code/` 与原 #6 逐文件一致，三个外置模型资产测试修正和此前真实权重验收均保留。唯一 Git 冲突是 `.gitattributes`，取双方规则并集，并为本轮 Windows 原始日志保留 CRLF 字节。

#5 新增的十组工作是历史读取、描述桥接、问法/输出格式、缺证据拒答、目标时间取证和视觉通道诊断。它们保留为独立候选和固定研究证据，没有替换默认推理、记忆策略或权重。方向可以继续研究，但不能据此声称已经改善通用视频质量：

- 直接与原始特征读取在固定同输入上生成相同；允许 none 的问法却会全部答 none。
- 描述后再回答有局部纠错，也新增错误和格式失败；新静态素材上没有显示优于直接读取。
- 缺证据时仍会猜测，关系与选项顺序敏感；固定时间取证后的局部问法在熟悉样本有改善，尚非独立泛化结果。
- 最新视觉加描述与相同视觉任务控制均为4/6，辅助描述净纠错0；重复犬问题使分数偏重，不能声称整体获益。

上述结论来自协作者的原报告/原答和本轮字节核验；本轮没有重跑这些GPU、下载其素材或进行新模型训练。

## 发现并修复的问题

恢复工具原先使用完整 `gzip.decompress()` 后才检查4MiB上限，无法兑现解压读取上限。现在限制外层文件8MiB、解压读取最多4MiB+1，并检查声明大小/哈希；拒绝重复路径、文件与目录冲突，在全部条目验证后才写新目录。恢复不自动执行脚本。已加入8项专项测试，包括十个真实包逐字节恢复、UTF-8/CRLF、不执行脚本、越界路径、损坏身份、输出目录保护与实际读取上限。

压缩包中的 `visual_channel_cpu_20261015_01/source_snapshot/probe.py` 有语法错误，属于作者明确记录的首次失败，保留原字节；最终源码及后续成功版本语法正常。不能修写失败快照来制造全绿。新审计首次误以为全部历史收据都使用 `default_promoted` 字段；grounding实际使用 `next_candidate_activated`。保留[审计01失败](runs/pr5_integration_delivery_20261016_01/verification.json)和源码快照，修正模式兼容后[审计02通过](runs/pr5_integration_delivery_20261016_02/verification.json)。

## 本轮新验证

- [完整CPU回归195项](runs/pr5_integration_cpu_20261016_01/summary.json)：根3、实验原基线12、活动S4 180；失败/错误/跳过0。
- [时间帧7项](runs/pr5_integration_contracts_20261016_01/temporal/summary.json)与[时间文件8项](runs/pr5_integration_contracts_20261016_01/time_file/summary.json)重新通过。
- [恢复工具8项](runs/pr5_integration_tools_20261016_01/summary.json)通过；合计 **218 次成功CPU测试执行**，不把同轮重复运行计入总数。
- [档案/源码整合审计](runs/pr5_integration_delivery_20261016_02/verification.json)：10包、488文件大小/SHA逐项一致，277份内嵌JSON解析通过；106个有效Python AST通过，1个已登记历史失败快照保留。各CPU proof及GPU摘要哈希均与原发布收据对应。除文档、属性和恢复工具等明确调整，#5的168文件保持原字节。
- [默认路径历史验收身份复核](runs/pr5_integration_default_identity_20261016_01/verification.json)：当前原始模型/流式/来源/启动器哈希与0.8B已验记录匹配；不当作新真实运行。本轮新增GPU载入 **0**，既有实测见[0.8B收尾报告](CLOSEOUT_08B.md)。

## 十组原始诊断入口

- [同帧读取与无动物回答](READOUT_GROUNDING_RESULT.md)
- [描述桥接](EVIDENCE_BRIDGE_RESULT.md)、[新静态素材](READOUT_HOLDOUT_RESULT.md)、[困难与顺序问题](READOUT_CHALLENGE_RESULT.md)
- [文本读取格式/语义分离](TEXT_READER_RESULT.md)、[关系和缺证据控制](READER_CONTROLS_RESULT.md)
- [按时间目标读取](TARGET_READOUT_RESULT.md)、[局部问题/描述包装](LOCAL_READOUT_RESULT.md)、[新视频迁移](READOUT_TRANSFER_RESULT.md)
- [保留视觉并添加辅助描述](VISUAL_CHANNEL_RESULT.md)

以上目录的完整源码/配置/原答位于对应 `runs/*/full_text.json.gz-base64.json`，原文本无损保存。新机器可通过 `python tools/restore_text_bundle.py <包路径> --out <不存在的目录>` 恢复查看；旧独立脚本含作者路径、分支及素材身份约束，运行前须适配本机路径并重新CPU预检，不能将恢复成功称为可以直接复用旧GPU proof。默认通用入口仍按README运行。

## 后续边界

两个PR统一后只审核#6，不需要另一人处理冲突。历史读取候选不升级默认；正式质量、摄像头/自然长时、BF16与性能等限制保持。下一阶段2B仍须核配置/权重、原模型基线、同输入状态隔离和实际显存/质量；本轮不启动2B。
