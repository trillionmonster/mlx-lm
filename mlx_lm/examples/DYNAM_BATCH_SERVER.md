# MLX LM 示例

本目录包含使用 `mlx-lm` 构建的高级示例。

## 动态批处理服务器 (Dynamic Batch Server)

`dynamic_batch_server.py` 是一个基于 Python `http.server` 实现的轻量级推理服务器。它利用 `mlx-lm` 的 `BatchGenerator` 实现了动态批处理（Dynamic Batching），能够在处理并发请求时显著提高吞吐量。

该服务器提供了一个兼容 OpenAI API 的接口 (`/v1/chat/completions`)，支持流式 (streaming) 和非流式响应。

### 特性

- **动态批处理**: 自动将多个并发请求合并为一个批次进行推理，最大化 GPU 利用率。
- **OpenAI 兼容**: 支持标准的 OpenAI 聊天补全 API。
- **流式输出**: 支持 Server-Sent Events (SSE) 流式输出。

### 用法

启动服务器：

```bash
python dynamic_batch_server.py --model mlx-community/Llama-3.2-3B-Instruct-4bit --port 8080
```

### 参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--model` | 模型路径或 Hugging Face 仓库 ID (必填) | - |
| `--adapter-path` | LoRA 适配器路径 (可选) | None |
| `--host` | 绑定的主机地址 | 127.0.0.1 |
| `--port` | 监听端口 | 8080 |
| `--batch-size` | 最大批处理大小 | 32 |
| `--log-level` | 日志级别 (DEBUG, INFO, WARNING, ERROR) | INFO |

### API 调用示例

你可以使用 `curl` 或 `openai` Python 客户端进行调用：

```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "default",
    "messages": [{"role": "user", "content": "Hello!"}],
    "stream": true
  }'
```

---

## 基准测试客户端 (Benchmark Client)

`benchmark_client.py` 是一个用于压力测试和性能评估的工具。它可以模拟多个并发用户向服务器发送请求，并计算关键的性能指标。

### 依赖

需要安装 `openai` 库：

```bash
pip install openai
```

### 用法

运行基准测试：

```bash
python benchmark_client.py --concurrency 8 --requests 50 --stream --model mlx-community/Llama-3.2-3B-Instruct-4bit
```

### 参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--base-url` | API 基础 URL | http://localhost:8080/v1 |
| `--api-key` | API 密钥 (服务器未验证时可随意填写) | EMPTY |
| `--model` | 模型名称 | default_model |
| `--concurrency` | 并发线程数 (模拟用户数) | 4 |
| `--requests` | 发送的总请求数 | 20 |
| `--stream` | 是否使用流式请求 (推荐开启以计算 TTFT) | False |
| `--min-delay` | 请求之间的最小随机延迟 (秒) | 0.1 |
| `--max-delay` | 请求之间的最大随机延迟 (秒) | 2.0 |
| `--min-tokens` | 生成的最小 token 数 (通过 max_tokens 控制) | 10 |
| `--max-tokens` | 生成的最大 token 数 | 512 |

### 输出指标解释

测试完成后，客户端会输出以下统计信息：

- **Successful requests**: 成功完成的请求数量。
- **Total tokens generated**: 所有请求生成的总 token 数量。
- **Total duration**: 测试总耗时。
- **Throughput (TPS)**: 系统吞吐量 (Tokens Per Second)。
- **Avg Latency**: 平均请求延迟 (从发送到收到完整响应)。
- **Avg TTFT (Time To First Token)**: 平均首字延迟 (仅在流式模式下准确)。
- **Avg ITL (Inter-Token Latency)**: 平均 token 间延迟 (生成的流畅度)。
