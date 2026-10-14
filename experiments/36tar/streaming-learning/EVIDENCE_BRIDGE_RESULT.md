# 把看图与作答分开：两阶段证据读取候选

2026-10-15 固定结果。持续进度以《实验总纲.md》为准；这不是综合汇报的更新。

这次有一个值得继续研究的现象：同样四组画面，先让模型中性描述，再在全新文字上下文中判断，结果能区分有动物和没有动物。不过，**严格按事先要求的输出，候选仍未通过：它应该输出动物名称或 none，却只回答 Yes / No。** 视觉描述也有不可靠细节。因此保留为诊断候选，不接入默认入口。

## 为什么需要它

上一段，直接问“画面里是什么动物”，狗和猫能答对，纯文字片头、片尾却也被答成狗。加上“没有动物可以回答 none”后，四组又全答 none。这提示我们：最终回答可能受提问方式影响，不能仅凭一次答错就认为历史特征已经损坏。

这次只改变读取的组织方式，让过程更容易检查：

```text
成功写入的一对真实画面 → 完整48个视觉Token → 独立视觉问答：描述看到了什么
                                               ↓ 模型实际生成的描述
                                        全新纯文字问答：是否提到动物

另一路诊断：人工事前描述 → 全新纯文字问答
```

第二步没有图像，也没有第一步的缓存。它只能依据传入的描述。人工描述只检查第二步能否理解简单文本，不是系统自行看图的成果。

## 我们改了什么

新增独立实验 `experiments/evidence_bridge_v1`。第一步复用已有完整特征读取器，问题固定为：

```text
Describe the visible content of these video frames in one short sentence.
```

第二步创建新的 `HybridSequence(max_tokens=384)`，只输入以下文本，最多生成12个Token：

```text
A description of video frames follows.
Description: {模型实际生成的描述}
Question: Does this description explicitly mention an animal? If yes, name the animal. Otherwise answer none. If unclear, answer uncertain.
```

两个阶段共用权重，但不共用问答缓存，不把描述或答案写回视频主状态。已有混合状态续接、森林、压缩、模型层数和权重都未修改。这是一个**读取流程候选**，不是新训练出来的识别模块。

没有可用描述时记录 uncertain 并跳过第二次模型调用。CPU01后发现该拒绝分支未写入逐行日志，已补齐并以新CPU02核最终来源；真实运行没有遇到空描述。旧CPU01保留，不覆盖。

## 实际效果：四例全保留

沿用熟悉的四对小画布输入，无压缩，每对48个视觉Token。狗来自20/22秒，猫来自0/约1秒，片头来自0/2秒，片尾来自40/42秒；秒数分别属于各自源视频。

| 实际画面 | 第一步真实生成的完整描述 | 第二步原答 | 人工正确描述的第二步原答 |
|---|---|---|---|
| 浅色狗 | A white dog is eating grass in front of a tower. | Yes | Yes |
| 黑白猫 | A black and white cat is lying on the ground. | Yes | Yes |
| 无动物片头 | The video shows a white background with the text "Pablo MIV" in a stylized font. | No | No |
| 无动物片尾 | A video title screen with a black background and text in red, yellow, and white. | No | No |

可以确认的局部现象是：第一步对四例的动物主体/文字场景描述有区分；第二步输出也随描述区分有无动物。人工对照得到同样四个短答案，说明这里第二步至少能作简单有无判断。

但要同时看三个问题：

1. **严格输出没有通过。** 事前要求 dog、cat、none 或 uncertain，实际全是 Yes / No，严格标签0/4。事后把它们解读为有无动物可得4/4，但这只是辅助观察，不能替换预登记结果，也没有偷偷把 Yes 转成 dog 或 cat。
2. **描述不完全可信。** 两张狗画面不足以确认“吃草”；片头可见 MIX，而模型转写成 MIV；片尾被叫成 title screen 也不精确。猫的动作/姿态本轮不评分。不能把主体正确扩大为动作、文字和每个细节都正确。
3. **没有泛化或公平速度证据。** 四对都是以前用过的画面，人工文字又有意设计得简单；本轮未重跑直接读取基线，也没有测试森林压缩节点、事件先后或新视频。因此不能宣布比原模型更聪明、压缩损失解决或端到端更快。

## 验证和资源

最终CPU02通过三组契约：纯文字分支不调用视觉编码器；新缓存重复读取一致；主状态、银行、取回节点不变，空/过长/非文字描述拒绝。随机微型模型的完整重放只验流程，不评分。

真实GPU105为一次串行载入，4次视觉前向、102次语言前向、12次问答、82个生成Token，均EOS结束。峰值allocated约3.2312GiB，reserved约3.3730GiB。包含载入和replay的记录耗时12.42秒，不含全部启动/预扫描，不作实时速度。

事前预算为1次载入、4视觉、至多12问答/192生成/232语言、360秒；主状态256Token、视觉问答768、纯文字384。RAM≥3GiB、GPU空闲≥5.5GiB、FP32转换余量≥2GiB门槛保持。缩小RGB包/外部特征的限额不能代表整进程内存，原图解码、模型、主缓存、logits暂存另计。

独立审计02确认152个源文件、模型资产、视频SHA、CPU/GPU像素身份、实际描述传递、日志与summary一致、完整12答和调用预算。首次审计把资产列表误当字典而失败，修正审计器后重新检查；没有为此重跑GPU。失败收据保留。

## 下一步建议与边界

先不增加森林层数、压缩强度或训练。若继续这个候选，最有价值的是先明确它只承担“有无动物”这个窄任务，再用事前标注且未参与本轮的新画面对比直接读取与两阶段读取。明确任务后才能处理输出接口；本轮不会事后改提示或解析规则来把失败变成通过。

只有未压缩旧证据在独立样例上可读，才把同一读取流程放回未压缩/压缩、单独旧节点/混合近期画面的四组对照。这里仍需分别检验“保存了什么”和“如何读取”，不能用一段流畅描述证明旧事件已经被可靠记住。

复现入口仍在本地主工作区；源配置与原始小文本按原字节归档，恢复工具只解包、不执行。另一台机器须先适配本地路径并生成新CPU证明，不能直接沿用本机GPU准入。

```powershell
& 'E:\VIT Model\.venv\Scripts\python.exe' 'E:\VIT Model\experiments\evidence_bridge_v1\run.py' --stage preflight --out 'E:\VIT Model\runs\NEW_CPU_DIR'
# 看实际接触图、冻结inspection_rubric，再在资源门槛满足时：
& 'E:\VIT Model\.venv\Scripts\python.exe' 'E:\VIT Model\experiments\evidence_bridge_v1\run.py' --stage gpu --preflight 'E:\VIT Model\runs\NEW_CPU_DIR' --out 'E:\VIT Model\runs\NEW_GPU_DIR'
```
