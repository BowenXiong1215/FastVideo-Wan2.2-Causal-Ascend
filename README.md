# FastVideo Wan2.2 Causal Ascend

这是一个面向 8×Ascend 910B 的轻量补丁仓库，用于把 FastVideo 中的
Wan2.2 TI2V-5B 转换为按 latent block 自回归生成的 causal 视频骨干。

训练分为两阶段：

1. Causal SFT：把双向 Wan DiT 适配为 block-causal student。
2. Self-Forcing DMD2：修复 student 读取自身生成历史时的分布偏移和误差累积。

这不是少步蒸馏。训练和部署保留 50 个去噪位置；DMD2 训练使用
`last_step_only: false` 随机覆盖完整 50 点时间表，推理仍执行全部 50 点。

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

完整的模型检查、数据预处理、Causal SFT、Self-Forcing DMD2、checkpoint
导出和 50 步 causal 推理命令，安装后见：

```text
docs/getting_started/ascend_910b_wan22_causal_distillation.md
```

## 范围与验证状态

- 121 个 RGB 帧对应 31 个 latent frames，以 `3+3+…+3+1` 分块。
- block 之间单向因果，block 内保持双向注意力。
- 当前不包含动作、键盘或相机控制，因此不是完整可交互世界模型。
- 已完成 Python/Shell/YAML 静态验证、补丁哈希校验、干净源码安装及重复安装验证。
- 尚未在本仓库打包机器上完成真实 910B 张量测试或数值质量验证。

## License

Apache License 2.0。补丁对应文件仍遵循 FastVideo 上游许可。
