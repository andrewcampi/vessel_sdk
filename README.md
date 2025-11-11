# Vessel SDK

A Python client library for the Vessel API that provides a simple interface for embeddings and chat completions with automatic batch processing.

## Installation

```bash
pip install -r requirements.txt
```

For development installation:
```bash
pip install -e .
```

## Features

- **Automatic Batch Processing**: The SDK automatically handles batch processing for multiple inputs
- **Rate Limiting**: Built-in rate limiting to comply with API restrictions
- **Simple Interface**: Clean, intuitive API that mirrors the OpenAI SDK
- **Comprehensive Metrics**: Get detailed timing and throughput information
- **Error Handling**: Robust error handling with informative error messages

## Quick Start

```python
from vessel import Vessel
import os
from dotenv import load_dotenv

load_dotenv()

# Initialize the client
client = Vessel(
    base_url=os.getenv("VESSEL_BASE_URL"),
    api_key=os.getenv("VESSEL_API_KEY")
)

# List available models
response = client.models.list()
for model in response.data:
    print(f"{model.name}: ${model.input_cost} per 1M tokens")

# Get account information
account = client.accounts.retrieve()
print(f"Credits: ${account.data.credits}")

# Create a single embedding
response = client.embeddings.create(
    model="vessel-embedding-nano",
    input="Hello, world!",
    verbose=True
)

if response.success:
    print(f"Embedding dimension: {len(response.data[0].embedding)}")
    print(f"Throughput: {response.throughput:.2f} tokens/sec")

# Create batch embeddings
response = client.embeddings.create(
    model="vessel-embedding-nano",
    input=["Hello", "World", "How are you?"],
    verbose=True
)

# Single chat completion
response = client.chat.completions.create(
    model="vessel-llm-nano",
    messages=[
        {"role": "user", "content": "Hello!"}
    ],
    verbose=True
)

if response.success:
    print(response.data[0].message.content)

# Batch chat completions
response = client.chat.completions.create(
    model="vessel-llm-nano",
    messages=[
        [{"role": "user", "content": "Hello!"}],
        [{"role": "user", "content": "What is the capital of France?"}]
    ],
    verbose=True
)
```

## API Reference

### `Vessel(base_url, api_key)`
Main client for the Vessel API.

**Parameters:**
- `base_url` (str): The base URL for the Vessel API
- `api_key` (str): Your API key

### `client.models.list()`
List all available models.

**Returns:** `ModelsListResponse` with a `data` field containing a list of models.

### `client.accounts.retrieve()`
Get account information.

**Returns:** `AccountRetrieveResponse` with account details including email and credits.

### `client.embeddings.create(model, input, verbose=False)`
Create embeddings for text input.

**Parameters:**
- `model` (str): Model name (e.g., "vessel-embedding-nano")
- `input` (str | List[str]): Text or list of texts to embed
- `verbose` (bool): Print progress information

**Returns:** `EmbeddingsResponse` with:
- `success` (bool): Whether the request succeeded
- `data` (List[EmbeddingData]): The embedding vectors
- `input_tokens` (int): Number of input tokens processed
- `throughput` (float): Processing speed in tokens/second
- `processing_time` (float): Time taken in seconds
- `error` (str | None): Error message if failed

### `client.chat.completions.create(model, messages, verbose=False, **kwargs)`
Create chat completions.

**Parameters:**
- `model` (str): Model name (e.g., "vessel-llm-nano")
- `messages` (List[Dict] | List[List[Dict]]): Messages or list of message lists for batch
- `verbose` (bool): Print progress information
- `**kwargs`: Additional parameters (e.g., max_tokens, temperature)

**Returns:** `ChatCompletionsResponse` with:
- `success` (bool): Whether the request succeeded
- `data` (List[ChatChoice]): The chat responses
- `input_tokens` (int): Number of input tokens
- `output_tokens` (int): Number of output tokens
- `throughput` (float): Processing speed in tokens/second
- `processing_time` (float): Time taken in seconds
- `error` (str | None): Error message if failed

## Rate Limiting

The SDK automatically handles rate limiting with a 10-second cooldown between API requests. This happens transparently in the background.

## Batch Processing

When you pass a list of inputs to `embeddings.create()` or a list of message lists to `chat.completions.create()`, the SDK automatically:

1. Creates batch tasks in JSONL format
2. Uploads the batch file
3. Creates and monitors the batch job
4. Downloads and parses the results
5. Returns structured data with metrics

This all happens transparently - you just pass your data and get results back!
