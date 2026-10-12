# 可见纸巾事实与双事实判定：适应性预登记

2026-10-12一小时窗口。在可见流水题六例语义一致、但Yes/No大小写使旧strict0/6，以及中性动作数字1/6之后，本轮是适应性诊断，不作新留出/通用准确率。固定相同旧新六对两帧，只问hands taking or using paper towel。取纸/用纸两例期望YES，其余NO，模型新回答前冻结；旧source/标签/question/proof和错误不改。

评分保留完整trim大小写敏感strict；另预登记归一化为trim后lowercase完整yes/no，其他字符串拒绝/不确定，不截前缀。新paper题和旧water题同一帧RGB/PTS/grid后，按两个视觉事实派生：water yes、paper no→1；water no、paper yes→2；两者no→3；同时yes或无法解析→不确定。优先避免把矛盾硬判类。此规则在本轮paper答案前固定，旧water事后语义解释不改其strict0/6记录。

CPU4epoch契约/6独立whole-pair/原修改微型fresh prefill/两素材SHA，新方案config/旧water summary SHA绑定。原模型GPU最多一次载入6视觉102语言96生成16Token EOS/总≤288/90秒，RAM3GiB/显存5.5GiB/转换2GiB不降。没有流式/持续视频确认，不把独立两帧事实路线当连续问答修复。六已知对中有一例可见水流YES正对照/两纸巾YES正对照，归一化成绩必须同时报告strict/拒绝/错误原文。

11:23:24前停止；完成后仅便携核验、审计、通俗报告、个人PR与一个小型备份，无训练/摄像头/集群/压缩森林/后台工作。持续进度仅总纲。
