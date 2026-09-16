# FastVideo Wan2.2 Causal Ascend

这是一个面向 8×Ascend 910B 的轻量补丁仓库，用于把 FastVideo 中的
Wan2.2 TI2V-5B 转换为按 latent block 自回归生成的 causal 视频骨干。

训练分为两阶段：

1. Causal SFT：把双向 Wan DiT 适配为 block-causal student。
2. Self-Forcing DMD2：修复 student 读取自身生成历史时的分布偏移和误差累积。

当前推荐方案是 4 步蒸馏：保留 causal SFT 初始化，DMD2 使用
`[1000, 750, 500, 250]`，并在每个序列上随机选择一个去噪位置保留梯度；
student 更新先用全程 no-grad 的 rollout 计算 teacher/critic target，释放打分
峰值后再重算一个随机 causal block 反向，因此恢复所有 block 的无偏覆盖，同时
避免 student 图与打分模型峰值叠加。之前的 50 步配置仍保留用于对照实验。

## 固定环境

| 项目 | 版本 |
| --- | --- |
| 硬件 | 8×Ascend 910B 64 GB |
| 基础镜像 | `quay.io/ascend/triton:3.2.1-cann9.0.0-torch_npu2.7.1.post4-910b-ubuntu22.04-py3.11` |
| CANN | 9.0.0 |
| PyTorch | 2.7.1 |
| torch_npu | 2.7.1.post4 |
| Python | 3.11 |
| FastVideo revision | `7bb76b5ec99807a66aa3047b901f15019abe0f00` |

全部 attention 使用 Dense/TORCH_SDPA，不使用 VSA。

## 安装

在固定 FastVideo revision 的源码树上运行：

```bash
bash fastvideo-ascend-910b-wan22-causal-20260910/install.sh \
  /workspace/FastVideo

bash fastvideo-ascend-910b-wan22-causal-20260910/verify.sh \
  /workspace/FastVideo
```

也可以使用归档：

```bash
sha256sum -c fastvideo-ascend-910b-wan22-causal-20260910.tar.gz.sha256
tar -xzf fastvideo-ascend-910b-wan22-causal-20260910.tar.gz
```

完整的模型检查、数据预处理、Causal SFT、4 步 Self-Forcing DMD2、
checkpoint 导出和 causal 推理命令，安装后见：

```text
docs/getting_started/ascend_910b_wan22_causal_distillation.md
```

## 范围与验证状态

- 推荐先以 49 个 RGB 帧（13 latent frames）训练 50～100 步观察，再升到
  81 个 RGB 帧（21 latent frames）；两者均以三帧 latent block 分块。
- block 之间单向因果，block 内保持双向注意力。
- Parquet dataloader 显式选择当前平台的 pinned-memory device，避免 Ascend
  首次 `iter(dataloader)` 时错误初始化 CUDA。
- checkpoint 导出读取训练时保存的 resolved config，保留本地模型绝对路径，
  避免离线集群错误访问 Hugging Face Hub。
- DMD2 的 critic/student 轮次完全分离，避免第 5 步同时持有两套反向图导致 OOM；
  导出时同时记录 4 步时间表，推理自动复用。
- DMD2 导出只实例化 student 并且只从 DCP 读取 student transformer，不再在
  单卡上同时构建 5B student、teacher、critic 和两套优化器。
- 当前不包含动作、键盘或相机控制，因此不是完整可交互世界模型。
- 已完成 Python/Shell/YAML 静态验证、补丁哈希校验、干净源码安装及重复安装验证。
- 尚未在本仓库打包机器上完成真实 910B 张量测试或数值质量验证。

## License

Apache License 2.0。补丁对应文件仍遵循 FastVideo 上游许可。
