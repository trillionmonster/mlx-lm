# Thanks To [MLM-LM](https://github.com/ml-explore/mlx-lm) BUT batch dynamic server is really Useful

##  just have a Try, 4~5 speed up in batch=8
`
pip install git+https://github.com/trillionmonster/mlx-lm-batch-server.git
`

# MLX LM Examples

This directory contains advanced examples built using `mlx-lm`.

## Dynamic Batch Server

`mlx_lm.batch_server` is a lightweight inference server implemented based on Python's `http.server`. It utilizes `mlx-lm`'s `BatchGenerator` to implement dynamic batching, significantly improving throughput when handling concurrent requests.

The server provides an OpenAI API compatible interface (`/v1/chat/completions`), supporting both streaming and non-streaming responses.

### Features

- **Dynamic Batching**: Automatically merges multiple concurrent requests into a single batch for inference, maximizing GPU utilization.
- **Prompt Caching**: Leverages Radix Tree based caching to reuse KV cache across requests with shared prefixes (e.g. system prompts, multi-turn chat history), significantly reducing Time To First Token (TTFT).
- **OpenAI Compatible**: Supports the standard OpenAI Chat Completions API.
- **Streaming Output**: Supports Server-Sent Events (SSE) streaming output.

### Usage

Start the server:

```bash
# Basic usage
mlx_lm.batch_server --model mlx-community/Llama-3.2-3B-Instruct-4bit --port 8080

# With Prompt Caching enabled (Recommended for multi-turn chat)
mlx_lm.batch_server --model mlx-community/Llama-3.2-3B-Instruct-4bit --use-prompt-cache --radix-cache-size 2147483648
```

### Arguments

| Argument | Description | Default |
|------|------|--------|
| `--model` | Model path or Hugging Face repo ID (Required) | - |
| `--adapter-path` | LoRA adapter path (Optional) | None |
| `--host` | Host address to bind | 127.0.0.1 |
| `--port` | Port to listen on | 8080 |
| `--batch-size` | Maximum batch size | 32 |
| `--max-tokens` | Default maximum number of tokens to generate | 512 |
| `--trust-remote-code` | Enable trusting remote code for tokenizer | False |
| `--log-level` | Log level (DEBUG, INFO, WARNING, ERROR) | INFO |
| `--use-prompt-cache` | Enable prompt caching | False |
| `--radix-cache-size` | Size of the Radix Cache in bytes | 1073741824 (1GB) |
| `--chat-template` | Specify a chat template for the tokenizer | "" |
| `--use-default-chat-template` | Use the default chat template | False |
| `--chat-template-args` | JSON formatted string of arguments for chat template | "{}" |
| `--temp` | Default sampling temperature | 0.0 |
| `--top-p` | Default nucleus sampling top-p | 1.0 |
| `--top-k` | Default top-k sampling | 0 |
| `--min-p` | Default min-p sampling | 0.0 |

### API Call Example

You can use `curl` or the `openai` Python client to make calls:

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

## Benchmark Client

`benchmark_client.py` is a tool for stress testing and performance evaluation. It can simulate multiple concurrent users sending requests to the server and calculate key performance metrics.

### Dependencies

Requires the `openai` library:

```bash
pip install openai
```

### Usage

Run the benchmark:

```bash
python mlx_lm/examples/benchmark_client.py --concurrency 8 --requests 50 --stream --model mlx-community/Llama-3.2-3B-Instruct-4bit
```

### Arguments

| Argument | Description | Default |
|------|------|--------|
| `--base-url` | API Base URL | http://localhost:8080/v1 |
| `--api-key` | API Key (Can be anything if server doesn't verify) | EMPTY |
| `--model` | Model name | default_model |
| `--concurrency` | Number of concurrent threads (simulated users) | 4 |
| `--requests` | Total number of requests to send | 20 |
| `--stream` | Whether to use streaming requests (Recommended for calculating TTFT) | False |
| `--min-delay` | Minimum random delay between requests (seconds) | 0.1 |
| `--max-delay` | Maximum random delay between requests (seconds) | 2.0 |
| `--min-tokens` | Minimum tokens to generate (controlled via max_tokens) | 10 |
| `--max-tokens` | Maximum tokens to generate | 512 |

### Output Metrics Explanation

After the test completes, the client outputs the following statistics:

- **Successful requests**: Number of requests successfully completed.
- **Total tokens generated**: Total number of tokens generated across all requests.
- **Total duration**: Total duration of the test.
- **Throughput (TPS)**: System throughput (Tokens Per Second).
- **Avg Latency**: Average request latency (from sending to receiving full response).
- **Avg TTFT (Time To First Token)**: Average Time To First Token (Accurate only in streaming mode).
- **Avg ITL (Inter-Token Latency)**: Average Inter-Token Latency (Generation smoothness).

---

## Prompt Cache Test

`test_prompt_cache.py` is a specialized test script designed to verify the effectiveness of the Prompt Cache mechanism.

### Usage

```bash
python mlx_lm/examples/test_prompt_cache.py --model mlx-community/Llama-3.2-3B-Instruct-4bit --concurrency 4
```

### How it works

The test runs 3 rounds of requests for each concurrent thread:
1.  **Round 1**: Sends a unique long prompt (~1500 tokens). This populates the cache.
2.  **Round 2**: Sends the same prompt as Round 1 plus a suffix. This should hit the cache and have a significantly lower TTFT.
3.  **Round 3**: Sends the same prompt as Round 1 plus a different suffix. This should also hit the cache.

The script calculates the speedup ratio between Round 1 and Round 2 to validate acceleration.
