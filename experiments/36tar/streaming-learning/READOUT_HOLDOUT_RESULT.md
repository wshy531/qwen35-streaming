# 换新画面以后：两阶段读取是否更有用

2026-10-15固定结果。持续进度只看《实验总纲.md》，不更新旧综合汇报。

**这次没有发现两阶段读取比直接读取更好的证据。** 四张新来源照片中，狗和兔都有动物，喷泉与书架没有可见动物；两种方式全部作出正确有无判断。两阶段流程继续作为诊断工具保留，不设为默认，也没有改模型权重或森林。

## 为什么做这次对照

前轮在熟悉画面上，先描述再回答能够输出Yes/Yes/No/No，但事前要求的是动物名称或none，所以严格没有通过。不能事后把评分改成有无动物，就称它成功。

这次事前把目标明确为一个更窄的问题：**有没有动物，两条路径都要求Yes/No。** 原始回答先去首尾空白、转小写，并允许去掉一个句末英文句号；其他输出不自动改写。两个动物正例、两个无动物负例分开计，全部保留。这个任务比辨别物种、动作和事件先后简单。

## 我们改了什么

新建独立 `experiments/readout_holdout_v1`，旧源码与默认入口不动。四张公共照片是本轮新下载的新来源；保留作者、许可、原文件大小及SHA1/SHA256。这里的“新”指没有用于此前项目的读取实验，不保证模型预训练时没见过。

每张照片EXIF转正后等比缩到最长边256，再由原处理器适配层补边到256×192。为了使用现有成对帧接口，每图复制成两张完全相同的输入，显式设合成时间0/1秒、编号0/1。**它们是静态图片控制，不是真实视频，不提供动作、连续输入或长期记忆证据。** 四个来源是四个样本，复制出来的八帧不能按八个独立样本计算。

每例只做一次视觉编码，保留完整48个视觉Token，主状态重置后分别读取同一份外部特征：

```text
同一完整视觉特征 ─→ 独立问答：有没有动物？ ─→ 直接回答
              └→ 独立问答：中性描述画面 ─→ 新纯文字上下文：描述是否提到动物？
```

两臂回答与描述各最多生成24Token。案例间交替先后顺序。纯文字步骤不读图片、素材名、案例ID、真值或前一步缓存；只接收模型真实生成的描述。所有问答都检查主状态、银行和节点未变。

冻结问题如下：

```text
直接：Is any animal visible in these frames? Answer Yes or No.

描述：Describe the visible content of these video frames in one short sentence.

文字：Does this description explicitly mention an animal? Answer Yes or No.
Description: {实际生成的描述}
```

比较的是两套读取流程。两题措辞与中间表示不同，不是某一个词、某个网络层的单因素因果实验。直接臂复用已有完整特征读取器，本轮没有再跑官方原始模型路径；前轮原/完整特征的局部一致证据不能推广成所有输入都等价。

## 全部真实结果

| 新画面与真值 | 直接读取原答 | 模型完整描述 | 两阶段原答 |
|---|---|---|---|
| 金毛狗：Yes | Yes | A golden retriever dog is sitting on the ground. | Yes |
| 兔子：Yes | Yes | A white rabbit with black ears is sitting on a bed of hay. | Yes |
| 喷泉广场：No | No. | A view of a city street with a fountain in the foreground and a tall building in the background. | No |
| 图书馆书架：No | No. | A library with shelves filled with books. | No |

按事前规则，两臂都正例2/2、负例2/2，四组判断一致，没有只有候选答对的样本。直接臂的句末句号属于事前允许的格式，不是看完答案再放宽。

这支持一个小结论：在这四张清楚的新照片上，两条路径都能完成有无动物判断。它不支持两阶段更强、泛化准确率100%或历史问答已经解决。样本少且容易，没有统计显著性主张，也没有按成功挑掉失败样本。

中间描述仍需警惕：狗图主要是头部，不能据此确认“坐在地上”；兔子的耳色与“干草床”表述也没有作为独立细节能力评分。主体判断成立不代表整句话每个细节都可靠。描述写得流畅，不能自动当成可信长期记忆。

## 验证和资源情况

CPU01三组契约及12次随机模型完整重放通过。独立CPU审核从原JPEG重建EXIF/缩图/补边，确认输入像素与证明一致，四组canvas均不同于前轮四例；两臂完整特征SHA相同。随机模型的乱码只用于流程核验，不作能力评分。

GPU106第一次尝试在导入后可用RAM约2.99GiB时被3GiB门槛拒绝，没有载入权重。用户表示会释放内存后，实际测得空闲约5.67GiB，再按原预算提交GPU107并成功。两次目录/状态/日志均保留；不能把106算成一次真实模型运行。

本轮实际一次真实载入，4视觉前向、95语言前向、12调用、75生成Token，全部EOS结束；峰allocated约3.2311GiB、reserved约3.4023GiB。载入加replay记录9.45秒，不含全部启动/预扫描，不是实时延迟或两臂公平速度测量。两阶段每例多一个模型调用，但没有据本轮数据宣称具体慢多少。

预算为1次实际载入、4视觉、至多12调用/288生成/332语言/360秒。主256Token、视觉问答768、文字上下文384；RAM3GiB/GPU空闲5.5GiB/FP32转换余量2GiB门槛均保持。原图解码、模型、主缓存、logits暂存另计，缩小帧的限额不能代表整进程内存。

独立真实运行审核核156个源、模型资产、照片SHA、CPU/GPU输入身份、三资源门槛、描述传递、同特征对照、12条JSONL与summary、状态隔离及预算。原图下载时书架曾遇HTTP429，在准备代码后仅有限重试原URL成功；沙箱网络首次拒绝与限流属于素材准备记录，不是模型失败。

## 这对下一步有什么用

最重要的判断是：**暂时没有理由为了简单有无动物题，给默认路径多加一层描述生成。** 描述桥有助于看见模型在说什么，但本轮没有显示新增正确回答，且可能带入细节幻觉。

停止继续围绕这四图调提示。下一段更贴近原项目目标的是回到一个明确历史问题：同一份可读旧证据，分别测试未压缩旧内容、压缩旧内容、未压缩旧内容加近期画面、压缩旧内容加相同近期画面。这样才能继续区分压缩损失和近期干扰。先在原视频真实帧对上确认问题可答，再冻结对照；静态有无动物通过不能替代事件记忆验证。

## 素材与复现

四图只留本地，不进入代码仓库/交接ZIP，来源元数据随小文本证据保存：

- [Retriever 01.JPG](https://commons.wikimedia.org/wiki/File:Retriever_01.JPG)，HTO，CC0。
- [Pet rabbit](https://commons.wikimedia.org/wiki/File:Pet_rabbit_%F0%9F%90%B0.jpg)，Santhosh annakili，CC0。
- [Fountain at San Jose City Hall](https://commons.wikimedia.org/wiki/File:Fountain_at_San_Jose_City_Hall.jpg)，Don DeBold，CC BY-SA 2.0。
- [Booksm.jpg](https://commons.wikimedia.org/wiki/File:Booksm.jpg)，Bellavista1957，CC BY-SA 3.0。

本机CPU/GPU输出分别为 `runs/readout_holdout_cpu_20261015_01` 与 `runs/readout_holdout_gpu_20261015_02`；失败载入保留在01 GPU目录。独立审核在 `runs/readout_holdout_cpu_audit_20261015_01` 与 `runs/readout_holdout_audit_20261015_01`。复现使用独立run.py新输出目录，先CPU、人工检查并冻结rubric，再GPU；另一台机器须适配本地路径和重做准入，不能复用本机CPU证明。
