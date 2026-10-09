# qwen35-streaming

基于 Qwen3.5-0.8B 官方权重和原始模型实现的流式视频理解实验。

当前阶段：独立维护 Qwen3.5 模型/配置代码，固定 Transformers 依赖，建立原始基线。
**当前尚未实现流式视频输入、缓存裁剪或 SFT。**

## 仓库结构

```text
qwen35_local/         本地模型和配置实现（后续在这里修改）
scripts/smoke_test.py 官方权重加载、文字/单图推理
tests/               小型随机模型对照测试，不下载权重
third_party/         上游许可证、提交及源码 SHA-256 记录
requirements.txt     固定核心依赖版本
THIRD_PARTY_NOTICES.md 来源和修改说明
```

源码来自 Transformers **v5.3.0**，提交
`aad13b87ed59f2afcfaebc985f403301887a35fc`。公共加载、生成、处理器等功能由
Transformers 提供；模型及视觉/语言子模块来自本仓库。没有复制整个 Transformers。
模型权重单独下载，不提交到 GitHub。

## Windows / RTX 4060 环境

建议 Python 3.11。以下命令在仓库根目录执行，无需激活虚拟环境。
CUDA wheel 需要兼容的 NVIDIA 驱动；用 `nvidia-smi` 查看驱动与显存。

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu126
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

CUDA 检查应输出 `True`。如果已经在这个虚拟环境中装过 CPU 版 PyTorch，
先用同一 CUDA 安装命令加 `--force-reinstall` 替换它。
Linux 下将 Python 路径替换为 `.venv/bin/python`。

仅做 CPU 代码验证时，将上面的 `cu126` 换成 `cpu`，不要据此估计 GPU 性能。
初始验证不依赖 FlashAttention、FLA 等额外编译扩展；后续按需要单独引入。

## 验证本地实现（无需下载权重）

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

测试使用同一组随机小模型参数，比较上游与本地实现的文字、合成图像/视频
logits、增量文本缓存，并检查从上游保存的检查点加载到本地类。
这些测试只检查源码迁移正确性，不能代表 0.8B 官方权重的理解能力、显存或速度。

初始化验证（2026-10-09）：Windows / Python 3.11.7 / PyTorch 2.10.0 CPU，
上述 3 项 unittest 全部通过。当前本地 `.venv` 是 CPU 验证环境；在 4060 上
运行前按上面的说明替换为 CUDA 版 torch 和 torchvision。
官方 0.8B 权重尚未下载，真实图片/视频质量和 GPU 推理尚未验证。

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

## 修改方式

- 从 `qwen35_local` 导入本地模型类，不使用 AutoModel 自动选择上游实现。
- 修改缓存可从 `Qwen3_5DynamicCache` 入手；注意同时存在 attention KV 和递归状态。
- 当前模型/配置文件独立维护，不运行上游 modular 代码生成器覆盖它们。
- 未修改时应与固定版本上游保持一致；架构改动后保留原始基线并补充针对性测试。
- 不直接修改 `.venv/Lib/site-packages`。不要单独升级 Transformers 而不复核本地代码。

## 工作计划

- [ ] 验证官方权重的文字、图像和短视频理解效果
- [ ] 加入视频读取、抽帧、时间戳和滑窗问答
- [ ] 加入持续输入和可视化演示
- [ ] 评测响应延迟、显存和长期运行稳定性
- [ ] 探索增量推理、KV cache 与递归状态优化
- [ ] 构建流式数据并进行 SFT

## 来源

- [官方权重](https://huggingface.co/Qwen/Qwen3.5-0.8B)
- [原始实现](https://github.com/huggingface/transformers/tree/v5.3.0/src/transformers/models/qwen3_5)
- [第三方许可证及修改记录](THIRD_PARTY_NOTICES.md)
