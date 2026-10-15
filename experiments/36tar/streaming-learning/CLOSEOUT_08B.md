# Qwen3.5-0.8B 工程收尾（2026-10-15）

2026-10-16 更新：已在本机整合 PR #5 并新增218项CPU整合验证；详见[统一整合报告](PR_INTEGRATION_20261016.md)。本文以下真实权重数据仍为10月15日原运行，不冒称整合后重跑。

本轮在 `wshy/closeout-08b-20261015` 完成现有 0.8B 文件视频流式原型的集成、回归和真实权重复验。基准为 main `6ed58af`，整合 [PR #3](https://github.com/wshy531/qwen35-streaming/pull/3) 已进入实验分支的低帧率修复；本分支合并后才进入 main。0.8B 可作为后续 2B 的工程对照版本，下一阶段尚未开始。

## 交付范围

- 活动实现仍唯一 `code/qwen35-streaming-s4`，原始模型基线保留在根目录和 `code/qwen35-streaming`。
- 采集进程在有意等待下次采样时发送限频元数据，避免正常 5 秒采样间隔被 3 秒 idle 门槛误判。进度不充当 RGB 帧，不写模型历史；实际堵塞读取仍由断流/超时机制处理。
- 三个 CPU 契约测试支持 `QWEN35_MODEL_DIR` 指向外置模型资产，默认路径和测试断言保持。
- 上传复现脚本、文本原答、版本/输入/资产哈希和成功/失败记录；权重、视频、虚拟环境、缓存及含帧画面的 HTML 留在本机。
- 不改变已有模型权重、状态算法、默认记忆策略、GDN 块大小或精度方案。此次是工程验收，没有训练。

## 本轮通过的验收

1. **完整 CPU 回归 195 项**：根基线 3、实验原基线 12、S4 活动实现 180，失败/错误/跳过均为 0。另补当前时间帧记忆 7、严格时间文件入口 8，共 **210 次成功测试执行**。预检重复执行的契约不另计；随机微型模型只验证接口和数值机制。
2. **真实 UCF101 采集对照**：固定三视频 × 0.2/1 FPS，共六组旧/新配对、12 个子进程。修复前五组完成，剃须 0.2 FPS 复现 idle 超时；修复后六组全部完成，实际帧/PTS/像素指纹、队列上限和退出均通过。
3. **原模型/流式质量对照**：固定三视频的 early/full 六上下文，1 FPS、192×256、FP32 IEEE/math、greedy、关闭 thinking；两个独立串行模型进程。六组生成 Token 全部一致，六次流式问答主混合缓存逐位不变，两臂共 12 次 EOS、130 个生成 Token。俯卧撑与剃须主体动作符合视频，篮球仍描述运球/防守，未识别扣篮；原始误答保留。
4. **通用墙钟播放入口**：剃须原视频 8 帧、1 次问答、1 次容量 reset、0 丢帧，答案 `A man is shaving his face.`。reset 后剩余 epoch 未达到首问四帧门槛，因此实际仅一次问答。主缓存与计数不被问答改写，来源进程退出。
5. **独立稀疏文件入口**：0.2 FPS、默认 idle 3 秒，实际 PTS 0/5 秒两帧、1 次问答、0 reset、0 丢帧；34 条限频进度通知，不误报断流，答案同上，采集/来源子进程均退出。最大单槽 RGB 147456 字节。

本轮共 **4 次真实模型载入、14 次真实问答、146 个生成 Token**。RTX 4060 Laptop 8 GiB 上 allocated 峰值约 3.274 GiB；reserved 最大约 3.598 GiB。二者不等于驱动总显存或后续任意配置需求。稀疏入口 GPU 首次因 RAM 不足在载入前拒绝；释放内存后，保持原 3 GiB RAM / 5.5 GiB 空闲 GPU 门槛通过。

机器为 Windows / Python 3.11.7 / torch 2.10.0+cu126 / Transformers 5.3.0 / PyAV 16.1.0 / NumPy 2.4.6。官方权重 revision `2fc06364715b967f1860aea9cf38778875588b17`，10 个模型资产按清单校验。UCF101 镜像固定 revision `e88622e14fa09fa328b1a97e795232f6189eda41`，来源和 SHA 见 [数据清单](runs/closeout_08b_capture_20261015_01/dataset_manifest.json)。

## 证据入口

- [独立汇总审计](runs/closeout_08b_delivery_20261015_03/verification.json) 与 [145 个文本证据的大小/SHA 清单](runs/closeout_08b_delivery_20261015_03/evidence_manifest.json)：核对根基线、实验基线、当前 S4、采集修复、质量对照和两个默认入口的源哈希；首次审计含 451 个 Python AST，提交前另核所有暂存 Python。
- [195 项完整回归](runs/closeout_08b_cpu_20261015_02/summary.json)、[7 项时间记忆](runs/closeout_08b_real_20261015_01/temporal/summary.json)、[8 项时间文件](runs/closeout_08b_time_file_cpu_20261015_02/summary.json)。各目录附原日志。
- [采集六组对照](runs/closeout_08b_capture_20261015_01/summary.json)、[质量预检 proof](runs/closeout_08b_quality_cpu_20261015_01/proof.json)、[完整真实答案/生成 Token](runs/closeout_08b_quality_gpu_20261015_01/summary.json)。
- [通用入口摘要](runs/closeout_08b_generic_gpu_20261015_01/summary.json) 与 [事件](runs/closeout_08b_generic_gpu_20261015_01/events.jsonl)；[稀疏来源摘要](runs/closeout_08b_sparse_file_gpu_20261015_03/summary.json) 与 [事件](runs/closeout_08b_sparse_file_gpu_20261015_03/events.jsonl)。

本机两个 GPU 输出目录另有 `viewer.html` 可直接查看画面和回答；网页含解码帧，不提交。移动过稀疏 CPU02/GPU02 的输出目录以统一布局，原字节未改；其中嵌入的旧绝对路径仍表示实际运行来源。

## 保留的失败与修正

- [首轮完整 CPU](runs/closeout_08b_cpu_20261015_01/summary.json)：三个 setUpClass 使用仓库内仅有元数据的模型路径，改为支持既有外置资产环境变量后，另开 CPU02 全量通过。
- [首轮真实检查控制器](runs/closeout_08b_real_20261015_01/summary.json) 仍为失败：时间文件收尾运行器漏注册正常入口的模块别名，8 项中 2 项报错；只修运行器导入，在新目录重跑 8 项通过。稀疏来源 `max-frames=3` 被既有 CLI 最小值 4 拒绝，改为 4，实际仍只采两帧。
- [GPU RAM 拒绝](runs/closeout_08b_sparse_file_gpu_20261015_02/resource_precheck.json) 和 [错误信息](runs/closeout_08b_sparse_file_gpu_20261015_02/traceback.txt)：载入前仅约 2.08 GiB 可用 RAM，未降门槛，用户释放内存后另开 GPU03 成功。
- [审计01](runs/closeout_08b_delivery_20261015_01/verification.json) 与 [审计02](runs/closeout_08b_delivery_20261015_02/verification.json)：新审计器先后误用采集源码路径基准、未解析 `@launcher` 别名；失败脚本快照保留，修审计路径后03通过。未改变实验原答或重跑已通过的模型。

## 当前可运行流程与复现

环境/权重安装见 [根 README](../../../README.md)。每次使用个人分支、固定依赖/资产/视频 SHA，先 CPU proof 再 GPU，参数一致且输出目录不存在；源码或输入改变需重新预检。以下从仓库根目录执行，路径按机器修改。控制器四次真实载入串行执行，沿用 [预登记预算](diagnostics/closeout_08b_v1/方案.md)；已有 run 目录不可覆盖。

```powershell
$Python = 'D:/AI_project/qwen35-streaming/.venv/Scripts/python.exe'
$env:QWEN35_MODEL_DIR = 'D:/AI_project/models/Qwen3.5-0.8B'
$env:QWEN35_CLOSEOUT_DATA_DIR = 'D:/AI_project/data/ucf101_capture_validation'
$Experiment = Join-Path (Get-Location).Path 'experiments/36tar/streaming-learning'
Set-Location $Experiment
& $Python diagnostics/closeout_08b_v1/checks.py runs/my_closeout_cpu_01
# 仅上一命令成功后运行；将分阶段结果写入 my_closeout_real_01_* 新目录。
if ($LASTEXITCODE -eq 0) {
    & $Python diagnostics/closeout_08b_v1/real_checks.py runs/my_closeout_real_01 --cpu-run runs/my_closeout_cpu_01
}
```

控制器在本轮结束后增加了新目录前缀/`--cpu-run` 参数，避免复现时引用作者的固定日期目录；语法与参数解析已核验，未为这项路径改动重复四次 GPU。原运行器版本冻结在 `runs/closeout_08b_real_20261015_01/source_snapshot`。单独复现某阶段可用 `diagnostics/capture_dataset_v1/validate.py`、`diagnostics/ucf101_quality_v1/run.py` 及两个通用启动器，参数见本轮各 run 的 `arguments.json` / `proof.json`。只核验当前归档证据时执行 `python diagnostics/closeout_08b_v1/audit.py runs/my_closeout_audit_01`，不载入模型；归档审计固定引用本次日期目录。

## 验收边界和下一阶段

默认文件流式链路、容量 reset、主状态隔离、播放/来源及本轮真实采样对照已完成工程收尾。已有可选 RGB 历史、时间检索、投影压缩、森林/事件代表候选保留各自历史方案和正负结果；本轮完整回归与额外时间契约覆盖当前实现，不声称逐个重跑历史所有 GPU 候选。候选没有整体升级默认。

尚无正式 UCF101 准确率、泛化提升、自然长视频长期稳定、物理摄像头、端到端提速或 BF16 全路径等价证明；EOS/Token 一致不能替代理解质量。现有误识别、截断和性能负收益继续保留。2B 阶段应新建开发分支，先核对其官方配置/资产与当前实现的兼容性，再做原模型基线、同输入增量/状态隔离及实际显存/质量对照；通过后再讨论 KV cache、记忆候选和 SFT。本次没有下载或运行 2B。
