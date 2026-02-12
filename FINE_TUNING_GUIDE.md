# Fine-Tuning GPT-4o with Your Exported Training Data

A step-by-step guide for exporting annotated conversation data from the Annotation Platform and using it to fine-tune an OpenAI GPT model.

---

## Table of Contents

1. [Before You Start](#1-before-you-start)
2. [Export Your Training Data](#2-export-your-training-data)
3. [Understand the Exported JSONL Format](#3-understand-the-exported-jsonl-format)
4. [Validate Your File Locally](#4-validate-your-file-locally)
5. [Set Up Your Python Environment](#5-set-up-your-python-environment)
6. [Upload Your Training File to OpenAI](#6-upload-your-training-file-to-openai)
7. [Create the Fine-Tuning Job](#7-create-the-fine-tuning-job)
8. [Monitor Training Progress](#8-monitor-training-progress)
9. [Test Your Fine-Tuned Model](#9-test-your-fine-tuned-model)
10. [Deploy to Production](#10-deploy-to-production)
11. [Pricing Reference](#11-pricing-reference)
12. [Troubleshooting](#12-troubleshooting)

---

## 1. Before You Start

### Prerequisites

- **Approved conversations**: At least 10 conversations must have status "approved" in the platform (OpenAI's minimum). The platform recommends 50+ for meaningful results.
- **OpenAI account**: You need an API account at [platform.openai.com](https://platform.openai.com) with billing enabled.
- **API key**: Generate one at [platform.openai.com/api-keys](https://platform.openai.com/api-keys).
- **Usage tier**: Fine-tuning GPT-4o requires **Tier 4 or 5**. GPT-4o-mini and GPT-4.1 family models are available from **Tier 1+**. Check your tier at [platform.openai.com/settings/organization/limits](https://platform.openai.com/settings/organization/limits).
- **Python 3.8+** installed on your machine.

### Which Model Should You Fine-Tune?

As of early 2026, these models support supervised fine-tuning:

| Model | Best For | Training Cost (per 1M tokens) |
|-------|----------|-------------------------------|
| `gpt-4.1-2025-04-14` | Best quality, complex tool use | $25.00 |
| `gpt-4.1-mini-2025-04-14` | Good balance of quality and cost | $5.00 |
| `gpt-4.1-nano-2025-04-14` | Cheapest, fast, simpler tasks | $1.50 |
| `gpt-4o-2024-08-06` | Multimodal, vision fine-tuning | $25.00 |
| `gpt-4o-mini-2024-07-18` | Budget-friendly GPT-4o | $3.00 |

**Recommendation for restaurant voice AI**: Start with `gpt-4.1-mini-2025-04-14` — it's significantly cheaper to train than GPT-4.1 full, handles tool calling well, and you can always upgrade later if quality isn't sufficient.

---

## 2. Export Your Training Data

### Step 2.1: Log in as Admin

Navigate to `http://localhost:8000/login/` and log in with your admin credentials.

### Step 2.2: Go to the Export Page

Navigate to `http://localhost:8000/admin-panel/export/`.

You'll see:
- The count of approved conversations available for export
- Estimated token count and training cost
- Configuration options

### Step 2.3: Configure Export Options

| Option | What it does | Recommended setting |
|--------|-------------|---------------------|
| **Include system prompt** | Adds the active system prompt as the first `"system"` message in each example. This teaches the model its identity and behavior. | **ON** (checked) |
| **Include tools** | Includes the `"tools"` array with function definitions (create_order, check_availability, etc.). Required if your conversations contain tool calls. | **ON** (checked) |
| **Include RAG context** | Injects knowledge base chunks into user messages as `\n\nContext:\n...` blocks. Teaches the model to use retrieved context when answering. | **ON** if your agent uses a knowledge base |
| **Tool calls only** | Only export conversations that contain at least one tool call. | OFF for general training, ON if you only want to improve tool usage |
| **Filter by agent** | Export only conversations from a specific agent. | Use if you have multiple agents and want a model per agent |
| **Filter by tag** | Export only conversations with a specific tag. | Use for curated subsets |

### Step 2.4: Preview Before Downloading

Click **"Preview First 3 Examples"** at the bottom of the page. This renders the first 3 JSONL lines with syntax highlighting so you can verify the format looks correct.

Things to check in the preview:
- The `"system"` message contains your full system prompt
- User and assistant messages alternate correctly
- Tool calls have proper `"arguments"` as JSON strings
- Tool responses follow their corresponding tool calls
- RAG context blocks (if enabled) appear in user messages
- Weight values appear on assistant messages where you set them

### Step 2.5: Download

You have two download options:

**Option A: Single JSONL file**
Click **"Export JSONL"** to download a single file like `finetune_2026-02-12_49examples.jsonl`. Use this if you plan to let OpenAI auto-split or if you want to manage the split yourself.

**Option B: Train/Val ZIP (recommended)**
Click **"Export Train/Val Split (ZIP)"** to download a ZIP containing:
- `train.jsonl` — 80% of examples (randomly shuffled)
- `val.jsonl` — 20% of examples

Using a validation file lets you monitor for overfitting during training. **This is the recommended option.**

### Step 2.6: Note the Stats

Before closing the export page, note down:
- **Number of examples** (e.g., 49)
- **Estimated token count** (e.g., 22,986)
- **Estimated training cost** (e.g., $1.72 at $25/1M tokens, 3 epochs)

These will help you verify the upload and anticipate costs.

---

## 3. Understand the Exported JSONL Format

Each line of the JSONL file is a single training example — one complete conversation. Here's what a full example looks like:

### Simple Conversation (no tools)

```json
{
  "messages": [
    {
      "role": "system",
      "content": "You are a friendly phone ordering assistant for Tony's Pizzeria..."
    },
    {
      "role": "user",
      "content": "Hi, I'd like to order a large pepperoni pizza"
    },
    {
      "role": "assistant",
      "content": "Great choice! A large pepperoni pizza. Would you like anything else?",
      "weight": 1
    },
    {
      "role": "user",
      "content": "No, that's it. My name is Alex."
    },
    {
      "role": "assistant",
      "content": "Perfect! One large pepperoni pizza for Alex. Can I get your phone number for the order?",
      "weight": 1
    }
  ],
  "parallel_tool_calls": false
}
```

### Conversation with Tool Calls

```json
{
  "messages": [
    {
      "role": "system",
      "content": "You are a friendly phone ordering assistant..."
    },
    {
      "role": "user",
      "content": "I'd like to order a large pepperoni pizza for pickup"
    },
    {
      "role": "assistant",
      "content": "Let me place that order for you!",
      "weight": 1,
      "tool_calls": [
        {
          "id": "call_001",
          "type": "function",
          "function": {
            "name": "create_order",
            "arguments": "{\"customerName\": \"Alex\", \"customerPhone\": \"+15551234567\", \"items\": [{\"itemName\": \"Large Pepperoni Pizza\", \"quantity\": 1}]}"
          }
        }
      ]
    },
    {
      "role": "tool",
      "tool_call_id": "call_001",
      "content": "{\"success\": true, \"orderId\": \"ORD-123\", \"estimatedTime\": \"20 minutes\"}"
    },
    {
      "role": "assistant",
      "content": "Your order has been placed! Order number ORD-123. It'll be ready in about 20 minutes.",
      "weight": 1
    }
  ],
  "tools": [
    {
      "type": "function",
      "function": {
        "name": "create_order",
        "description": "Create a pickup order for the customer",
        "parameters": {
          "type": "object",
          "properties": {
            "customerName": {"type": "string", "description": "Customer's name"},
            "customerPhone": {"type": "string", "description": "Customer's phone number"},
            "items": {"type": "array", "items": {"type": "object", "properties": {"itemName": {"type": "string"}, "quantity": {"type": "integer"}}}}
          },
          "required": ["customerName", "customerPhone", "items"]
        }
      }
    }
  ],
  "parallel_tool_calls": false
}
```

### Conversation with RAG Context

When "Include RAG context" is enabled, knowledge base chunks appear in user messages:

```json
{
  "role": "user",
  "content": "What pizzas do you have?\n\nContext:\nPIZZA MENU\nLarge Pepperoni Pizza - $14.99\nMedium Margherita Pizza - $9.99\nHawaiian Pizza - $12.99\nBBQ Chicken Pizza - $13.99"
}
```

The `\n\nContext:\n` block is injected into the user message that precedes the agent turn that used the RAG retrieval. This matches how your model will see context at inference time.

### Key Fields Explained

| Field | Description |
|-------|-------------|
| `messages` | Array of message objects making up the conversation |
| `messages[].role` | `"system"`, `"user"`, `"assistant"`, or `"tool"` |
| `messages[].content` | The text content of the message |
| `messages[].weight` | `0` = don't train on this response, `1` = train on it. Only on assistant messages. |
| `messages[].tool_calls` | Array of function calls the assistant made (only on assistant messages) |
| `tools` | Array of function/tool definitions available to the model |
| `parallel_tool_calls` | Always `false` — our agent makes one tool call at a time |

### The Weight System

The `weight` field on assistant messages controls what the model learns:

- **`weight: 1`** — The model is trained to produce this response. Use for good, correct responses.
- **`weight: 0`** — The model sees this message as context but is NOT trained to reproduce it. Used for:
  - Initial greetings before the customer speaks (auto-assigned by the platform)
  - Responses you manually marked as W:0 during annotation
- **No weight field** — Treated as `weight: 1` by OpenAI (default behavior)

---

## 4. Validate Your File Locally

Before uploading to OpenAI, validate the file to catch issues early. This saves you from failed uploads and wasted time.

### Quick Validation Script

Create a file called `validate_jsonl.py`:

```python
"""Validate a JSONL file before uploading to OpenAI for fine-tuning."""
import json
import sys
from collections import Counter

def validate_file(filepath):
    errors = []
    warnings = []
    stats = Counter()

    with open(filepath, 'r') as f:
        lines = f.readlines()

    if not lines:
        print("ERROR: File is empty")
        return

    print(f"File: {filepath}")
    print(f"Total examples: {len(lines)}")
    print()

    for i, line in enumerate(lines, 1):
        line = line.strip()
        if not line:
            continue

        # Parse JSON
        try:
            example = json.loads(line)
        except json.JSONDecodeError as e:
            errors.append(f"Line {i}: Invalid JSON - {e}")
            continue

        messages = example.get("messages", [])
        if not messages:
            errors.append(f"Line {i}: No messages array")
            continue

        # Count roles
        roles = [m.get("role") for m in messages]
        stats["total_examples"] += 1
        stats["total_messages"] += len(messages)

        has_system = "system" in roles
        has_user = "user" in roles
        has_assistant = "assistant" in roles
        has_tools = "tool" in roles

        if has_system:
            stats["with_system_prompt"] += 1
        if has_tools:
            stats["with_tool_calls"] += 1

        # Validate structure
        if not has_user:
            errors.append(f"Line {i}: Missing user message")
        if not has_assistant:
            errors.append(f"Line {i}: Missing assistant message")
        if messages[-1].get("role") != "assistant":
            errors.append(f"Line {i}: Last message must be assistant (got '{messages[-1].get('role')}')")

        # Validate tool call/response pairing
        pending_ids = set()
        for msg in messages:
            if "tool_calls" in msg:
                for tc in msg["tool_calls"]:
                    tc_id = tc.get("id")
                    if tc_id:
                        pending_ids.add(tc_id)
                    # Validate arguments are valid JSON
                    args_str = tc.get("function", {}).get("arguments", "")
                    try:
                        json.loads(args_str)
                    except (json.JSONDecodeError, TypeError):
                        errors.append(f"Line {i}: Invalid JSON in tool_call arguments for {tc.get('function', {}).get('name')}")
            elif msg.get("role") == "tool":
                tc_id = msg.get("tool_call_id")
                if tc_id in pending_ids:
                    pending_ids.discard(tc_id)
                else:
                    errors.append(f"Line {i}: Orphaned tool response (tool_call_id '{tc_id}' not found)")

        if pending_ids:
            errors.append(f"Line {i}: Unmatched tool_call_ids: {pending_ids}")

        # Check for empty content
        for j, msg in enumerate(messages):
            if msg.get("role") in ("user", "system"):
                if not msg.get("content", "").strip():
                    errors.append(f"Line {i}, message {j}: Empty {msg['role']} content")

        # Check weights
        for msg in messages:
            if msg.get("role") == "assistant" and "weight" in msg:
                w = msg["weight"]
                if w not in (0, 1):
                    errors.append(f"Line {i}: Invalid weight {w} (must be 0 or 1)")

    # Check minimum examples
    if stats["total_examples"] < 10:
        errors.append(f"Only {stats['total_examples']} examples. OpenAI requires at least 10.")
    elif stats["total_examples"] < 50:
        warnings.append(f"Only {stats['total_examples']} examples. OpenAI recommends 50+ for good results.")

    # Print results
    print("--- Stats ---")
    print(f"  Valid examples:       {stats['total_examples']}")
    print(f"  Total messages:       {stats['total_messages']}")
    print(f"  With system prompt:   {stats['with_system_prompt']}")
    print(f"  With tool calls:      {stats['with_tool_calls']}")
    print()

    if errors:
        print(f"--- ERRORS ({len(errors)}) ---")
        for e in errors:
            print(f"  [ERROR] {e}")
        print()

    if warnings:
        print(f"--- WARNINGS ({len(warnings)}) ---")
        for w in warnings:
            print(f"  [WARN] {w}")
        print()

    if not errors:
        print("PASSED - File is valid for OpenAI fine-tuning upload.")
    else:
        print("FAILED - Fix errors before uploading.")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python validate_jsonl.py <path_to_file.jsonl>")
        sys.exit(1)
    validate_file(sys.argv[1])
```

### Run It

```bash
python validate_jsonl.py train.jsonl
```

Expected output:

```
File: train.jsonl
Total examples: 39

--- Stats ---
  Valid examples:       39
  Total messages:       312
  With system prompt:   39
  With tool calls:      24

PASSED - File is valid for OpenAI fine-tuning upload.
```

If you downloaded the Train/Val ZIP, validate both files:

```bash
python validate_jsonl.py train.jsonl
python validate_jsonl.py val.jsonl
```

---

## 5. Set Up Your Python Environment

### Step 5.1: Create a Virtual Environment

```bash
mkdir fine-tuning && cd fine-tuning
python3 -m venv venv
source venv/bin/activate   # Linux/Mac
# venv\Scripts\activate    # Windows
```

### Step 5.2: Install the OpenAI SDK

```bash
pip install openai
```

### Step 5.3: Set Your API Key

**Option A: Environment variable (recommended)**

```bash
export OPENAI_API_KEY="sk-proj-your-key-here"
```

Add this to your `~/.bashrc` or `~/.zshrc` to make it permanent.

**Option B: In your script**

```python
from openai import OpenAI
client = OpenAI(api_key="sk-proj-your-key-here")
```

> **Security note:** Never commit API keys to git. Use environment variables or a `.env` file that's in your `.gitignore`.

### Step 5.4: Verify the Connection

```python
from openai import OpenAI

client = OpenAI()  # reads OPENAI_API_KEY from environment

# Quick test
models = client.models.list()
print(f"Connected. {len(models.data)} models available.")
```

---

## 6. Upload Your Training File to OpenAI

### Step 6.1: Upload the File

If you downloaded the **Train/Val ZIP**, unzip it first and upload both files. If you downloaded a **single JSONL**, upload just that one.

```python
from openai import OpenAI

client = OpenAI()

# Upload training file
train_file = client.files.create(
    file=open("train.jsonl", "rb"),
    purpose="fine-tune",
)
print(f"Training file uploaded: {train_file.id}")
print(f"  Filename: {train_file.filename}")
print(f"  Size: {train_file.bytes} bytes")
print(f"  Status: {train_file.status}")

# Upload validation file (if you have one)
val_file = client.files.create(
    file=open("val.jsonl", "rb"),
    purpose="fine-tune",
)
print(f"\nValidation file uploaded: {val_file.id}")
print(f"  Filename: {val_file.filename}")
print(f"  Size: {val_file.bytes} bytes")
print(f"  Status: {val_file.status}")
```

Expected output:

```
Training file uploaded: file-abc123def456
  Filename: train.jsonl
  Size: 148392 bytes
  Status: processed

Validation file uploaded: file-xyz789ghi012
  Filename: val.jsonl
  Size: 37098 bytes
  Status: processed
```

### Step 6.2: Save the File IDs

**Write these down.** You'll need them in the next step.

```
Training file ID:   file-abc123def456
Validation file ID: file-xyz789ghi012
```

You can also retrieve them later:

```python
# List all uploaded files
files = client.files.list()
for f in files.data:
    if f.purpose == "fine-tune":
        print(f"{f.id}: {f.filename} ({f.bytes} bytes) - {f.status}")
```

### Step 6.3: Wait for Processing

After upload, OpenAI processes and validates the file. This usually takes 10-30 seconds. The status will change from `"uploaded"` to `"processed"`.

```python
import time

# Poll until processed
while True:
    file_status = client.files.retrieve(train_file.id)
    if file_status.status == "processed":
        print("File processed and ready!")
        break
    elif file_status.status == "error":
        print(f"File processing failed: {file_status.status_details}")
        break
    print(f"Status: {file_status.status}... waiting 5s")
    time.sleep(5)
```

---

## 7. Create the Fine-Tuning Job

### Step 7.1: Create the Job

```python
from openai import OpenAI

client = OpenAI()

job = client.fine_tuning.jobs.create(
    # The base model to fine-tune
    model="gpt-4.1-mini-2025-04-14",

    # Your uploaded file IDs (from Step 6)
    training_file="file-abc123def456",         # replace with your ID
    validation_file="file-xyz789ghi012",       # replace with your ID (optional)

    # A suffix to identify your model (max 18 chars)
    suffix="tonys-pizzeria",

    # Hyperparameters (optional — "auto" is usually best to start)
    hyperparameters={
        "n_epochs": "auto",                    # auto picks 3-10 based on dataset size
        "batch_size": "auto",
        "learning_rate_multiplier": "auto",
    },
)

print(f"Fine-tuning job created!")
print(f"  Job ID: {job.id}")
print(f"  Status: {job.status}")
print(f"  Model:  {job.model}")
```

Expected output:

```
Fine-tuning job created!
  Job ID: ftjob-abc123xyz789
  Status: validating_files
  Model:  gpt-4.1-mini-2025-04-14
```

### Step 7.2: Understand the Parameters

| Parameter | What it does | Recommended |
|-----------|-------------|-------------|
| `model` | The base model to fine-tune. | `gpt-4.1-mini-2025-04-14` for cost-effective training |
| `training_file` | File ID of your training JSONL. | Required |
| `validation_file` | File ID of your validation JSONL. | Recommended to detect overfitting |
| `suffix` | Appended to your model name (e.g., `ft:gpt-4.1-mini:org::tonys-pizzeria`). Max 18 characters. | Use something descriptive |
| `n_epochs` | Number of full passes through the training data. More epochs = more training. | Start with `"auto"`. If underfitting, try 5-10. |
| `batch_size` | Examples processed per gradient update. | `"auto"` (usually 1-4 for small datasets) |
| `learning_rate_multiplier` | Scales the learning rate. Higher = faster learning but risk of instability. | `"auto"`. Try 1.5-2.0 if model isn't learning enough. |

### What Happens After You Create the Job

The job goes through these statuses:

```
validating_files → queued → running → succeeded
                                   ↘ failed
```

1. **validating_files** — OpenAI validates your JSONL format (1-2 minutes)
2. **queued** — Waiting for GPU capacity (minutes to hours, depends on demand)
3. **running** — Model is training (minutes to hours, depends on dataset size)
4. **succeeded** — Done! Your model is ready to use.

---

## 8. Monitor Training Progress

### Option A: Dashboard (easiest)

Go to [platform.openai.com/finetune](https://platform.openai.com/finetune) and click on your job. You'll see:

- Real-time training loss curve
- Validation loss (if you provided a validation file)
- Estimated completion time
- Checkpoints at each epoch

### Option B: Python Script

```python
from openai import OpenAI
import time

client = OpenAI()

job_id = "ftjob-abc123xyz789"  # your job ID from Step 7

while True:
    job = client.fine_tuning.jobs.retrieve(job_id)

    print(f"Status: {job.status}", end="")
    if job.estimated_finish:
        print(f" | ETA: {job.estimated_finish}s", end="")
    if job.trained_tokens:
        print(f" | Trained tokens: {job.trained_tokens}", end="")
    print()

    if job.status == "succeeded":
        print(f"\nTraining complete!")
        print(f"Fine-tuned model: {job.fine_tuned_model}")
        break
    elif job.status == "failed":
        print(f"\nTraining failed!")
        print(f"Error: {job.error}")
        break

    time.sleep(60)  # check every minute
```

### Step 8.3: View Training Events (detailed log)

```python
events = client.fine_tuning.jobs.list_events(job_id, limit=50)
for event in reversed(events.data):
    print(f"[{event.created_at}] {event.message}")
```

Example output:

```
[1707700000] Validating training file...
[1707700010] Training file validated successfully
[1707700020] Validating validation file...
[1707700030] Validation file validated successfully
[1707700100] Fine-tuning job started
[1707700200] Step 10/150: training loss=2.34
[1707700300] Step 20/150: training loss=1.87
[1707700400] Step 30/150: training loss=1.52, validation loss=1.61
...
[1707702000] New checkpoint at step 150 (epoch 3/3)
[1707702100] Fine-tuning job succeeded
[1707702100] Fine-tuned model: ft:gpt-4.1-mini-2025-04-14:your-org::abc123
```

### What to Look For

- **Training loss should decrease** over time. If it plateaus early, you might need more/better data.
- **Validation loss should also decrease**, but if it starts going UP while training loss keeps going down, your model is **overfitting** (memorizing training data rather than learning patterns).
- If overfitting occurs, OpenAI saves **checkpoints at each epoch** — you can use an earlier checkpoint that had lower validation loss.

---

## 9. Test Your Fine-Tuned Model

Once training succeeds, you'll have a model ID like:

```
ft:gpt-4.1-mini-2025-04-14:your-org::abc123
```

### Step 9.1: Simple Test

```python
from openai import OpenAI

client = OpenAI()

# Your fine-tuned model ID (from the training output)
MODEL = "ft:gpt-4.1-mini-2025-04-14:your-org::abc123"

response = client.chat.completions.create(
    model=MODEL,
    messages=[
        {
            "role": "system",
            "content": "You are a friendly phone ordering assistant for Tony's Pizzeria."
        },
        {
            "role": "user",
            "content": "Hi, I'd like to order a large pepperoni pizza for pickup."
        },
    ],
)

print(response.choices[0].message.content)
```

### Step 9.2: Test with Tool Calling

If your training data included tool calls, test that the model uses them correctly:

```python
from openai import OpenAI
import json

client = OpenAI()
MODEL = "ft:gpt-4.1-mini-2025-04-14:your-org::abc123"

# Define the same tools the model was trained on
tools = [
    {
        "type": "function",
        "function": {
            "name": "create_order",
            "description": "Create a pickup order for the customer",
            "parameters": {
                "type": "object",
                "properties": {
                    "customerName": {"type": "string"},
                    "customerPhone": {"type": "string"},
                    "items": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "itemName": {"type": "string"},
                                "quantity": {"type": "integer"},
                            },
                            "required": ["itemName", "quantity"],
                        },
                    },
                },
                "required": ["customerName", "customerPhone", "items"],
            },
        },
    },
]

# Multi-turn conversation
messages = [
    {"role": "system", "content": "You are a phone ordering assistant for Tony's Pizzeria."},
    {"role": "user", "content": "I'd like a large pepperoni and a medium margherita. Name is Alex, number is 555-1234."},
]

response = client.chat.completions.create(
    model=MODEL,
    messages=messages,
    tools=tools,
)

message = response.choices[0].message

# Check if the model made a tool call
if message.tool_calls:
    for tc in message.tool_calls:
        print(f"Tool call: {tc.function.name}")
        print(f"Arguments: {json.loads(tc.function.arguments)}")
else:
    print(f"Response: {message.content}")
```

Expected output:

```
Tool call: create_order
Arguments: {
  "customerName": "Alex",
  "customerPhone": "555-1234",
  "items": [
    {"itemName": "Large Pepperoni Pizza", "quantity": 1},
    {"itemName": "Medium Margherita Pizza", "quantity": 1}
  ]
}
```

### Step 9.3: Compare Against the Base Model

Run the same prompts against the base (non-fine-tuned) model to see the improvement:

```python
# Base model (not fine-tuned)
base_response = client.chat.completions.create(
    model="gpt-4.1-mini-2025-04-14",
    messages=messages,
    tools=tools,
)

# Fine-tuned model
ft_response = client.chat.completions.create(
    model=MODEL,
    messages=messages,
    tools=tools,
)

print("=== Base Model ===")
print(base_response.choices[0].message.content or "Made tool call")

print("\n=== Fine-Tuned Model ===")
print(ft_response.choices[0].message.content or "Made tool call")
```

Things to compare:
- Does the fine-tuned model match your restaurant's tone and style?
- Does it call the right tools at the right time?
- Does it handle menu-specific knowledge better?
- Does it produce the response format you expect?

### Step 9.4: Batch Evaluation (optional but recommended)

For a more rigorous evaluation, run your validation set through both models and compare:

```python
import json

MODEL_BASE = "gpt-4.1-mini-2025-04-14"
MODEL_FT = "ft:gpt-4.1-mini-2025-04-14:your-org::abc123"

# Load validation examples
with open("val.jsonl", "r") as f:
    val_examples = [json.loads(line) for line in f if line.strip()]

results = []
for example in val_examples[:10]:  # test first 10
    messages = example["messages"]
    # Use all messages up to the last assistant message as input
    # (everything before the final response)
    input_messages = []
    for msg in messages:
        if msg["role"] == "assistant" and msg == messages[-1]:
            break
        input_messages.append(msg)

    # Get response from fine-tuned model
    try:
        response = client.chat.completions.create(
            model=MODEL_FT,
            messages=input_messages,
            tools=example.get("tools"),
        )
        result = response.choices[0].message.content or "[tool call]"
    except Exception as e:
        result = f"[error: {e}]"

    expected = messages[-1].get("content", "[tool call]")

    print(f"Input:    {input_messages[-1]['content'][:80]}...")
    print(f"Expected: {expected[:80]}...")
    print(f"Got:      {result[:80]}...")
    print()
```

---

## 10. Deploy to Production

Once you're satisfied with the fine-tuned model's performance, here's how to use it in your application.

### Using the Fine-Tuned Model ID

Your fine-tuned model ID works exactly like any other model ID. Just replace the model name in your existing code:

```python
# Before (base model)
model = "gpt-4.1-mini-2025-04-14"

# After (fine-tuned)
model = "ft:gpt-4.1-mini-2025-04-14:your-org::abc123"
```

Everything else stays the same — the API, tools, message format, streaming, etc.

### In the X1 Restaurant Platform

If you're using this fine-tuned model with the X1 Restaurant platform's ElevenLabs integration, you would update the ElevenLabs agent configuration to use your fine-tuned model as the LLM backend (if ElevenLabs supports custom OpenAI models), or build a direct integration using the OpenAI API for the voice AI tool responses.

### Important Production Notes

1. **Fine-tuned model pricing is higher than base model pricing.** See the [Pricing Reference](#11-pricing-reference) section.
2. **Your fine-tuned model is permanent** — it won't be deleted unless you explicitly delete it.
3. **Rate limits** for fine-tuned models are the same as the base model.
4. **You can create multiple fine-tuned versions** from different training data or hyperparameters and A/B test them.

---

## 11. Pricing Reference

### Training Costs

| Model | Training Cost | Inference Input | Inference Output |
|-------|--------------|-----------------|------------------|
| `gpt-4.1` | $25.00 / 1M tokens | $3.00 / 1M tokens | $12.00 / 1M tokens |
| `gpt-4.1-mini` | $5.00 / 1M tokens | $0.80 / 1M tokens | $3.20 / 1M tokens |
| `gpt-4.1-nano` | $1.50 / 1M tokens | $0.20 / 1M tokens | $0.80 / 1M tokens |
| `gpt-4o` | $25.00 / 1M tokens | $3.75 / 1M tokens | $15.00 / 1M tokens |
| `gpt-4o-mini` | $3.00 / 1M tokens | $0.30 / 1M tokens | $1.20 / 1M tokens |

### How Training Cost is Calculated

```
Total cost = (tokens_in_file × n_epochs) × (cost_per_1M / 1,000,000)
```

**Example:**
- Training file: 50,000 tokens
- Epochs: 3 (default auto)
- Model: gpt-4.1-mini ($5.00 / 1M tokens)
- Cost: (50,000 x 3) / 1,000,000 x $5.00 = **$0.75**

The annotation platform's export page shows this estimate before you download.

### Inference Cost After Fine-Tuning

Using your fine-tuned model in production costs the same as the fine-tuning inference prices shown above. These are separate from training costs — training is a one-time expense, inference is ongoing per-request.

---

## 12. Troubleshooting

### File Upload Errors

| Error | Cause | Fix |
|-------|-------|-----|
| `Invalid file format` | File is not valid JSONL | Run the validation script from Step 4. Each line must be valid JSON. |
| `File is in prompt-completion format` | Using old format instead of chat format | Ensure each line has a `"messages"` array, not `"prompt"`/`"completion"`. |
| `Invalid message role` | Unknown role in messages | Only `"system"`, `"user"`, `"assistant"`, `"tool"` are allowed. |
| `Missing required field` | Message missing content | Every user/system message needs `"content"`. |

### Job Creation Errors

| Error | Cause | Fix |
|-------|-------|-----|
| `File not ready` | File still being processed | Wait 30 seconds and retry. |
| `Too few examples` | Less than 10 training examples | Go back to the annotation platform, approve more conversations, and re-export. |
| `Model not supported` | Wrong model ID | Use exact snapshot ID like `gpt-4.1-mini-2025-04-14`, not just `gpt-4.1-mini`. |

### Training Failures

| Symptom | Cause | Fix |
|---------|-------|-----|
| Training loss not decreasing | Data quality issues | Review your annotations — are the assistant responses actually good examples? |
| Validation loss increasing | Overfitting | Reduce epochs, add more diverse training data, or use an earlier checkpoint. |
| Model ignores system prompt | System prompt not in training data | Make sure "Include system prompt" was checked during export. |
| Model doesn't use tools | Tools not in training data | Make sure "Include tools" was checked during export and conversations with tool calls were included. |
| Model hallucinate menu items | No RAG context in training | Re-export with "Include RAG context" checked so the model learns to use provided context. |

### Cancelling a Job

If something goes wrong mid-training:

```python
client.fine_tuning.jobs.cancel("ftjob-abc123xyz789")
```

You are only charged for tokens processed up to the cancellation point.

### Deleting a Fine-Tuned Model

If you no longer need a model:

```python
client.models.delete("ft:gpt-4.1-mini-2025-04-14:your-org::abc123")
```

This is permanent and cannot be undone.

---

## Quick Reference: Complete Script

Here's a single self-contained script that does everything from upload to testing:

```python
"""
Complete fine-tuning workflow for annotation platform exports.
Usage: python finetune.py train.jsonl [val.jsonl]
"""
import sys
import time
import json
from openai import OpenAI

client = OpenAI()  # reads OPENAI_API_KEY from environment

# --- Configuration ---
BASE_MODEL = "gpt-4.1-mini-2025-04-14"
SUFFIX = "restaurant-ai"  # max 18 chars, identifies your model

def main():
    if len(sys.argv) < 2:
        print("Usage: python finetune.py train.jsonl [val.jsonl]")
        sys.exit(1)

    train_path = sys.argv[1]
    val_path = sys.argv[2] if len(sys.argv) > 2 else None

    # Step 1: Upload files
    print("Uploading training file...")
    train_file = client.files.create(file=open(train_path, "rb"), purpose="fine-tune")
    print(f"  ID: {train_file.id} ({train_file.bytes} bytes)")

    val_file_id = None
    if val_path:
        print("Uploading validation file...")
        val_file = client.files.create(file=open(val_path, "rb"), purpose="fine-tune")
        val_file_id = val_file.id
        print(f"  ID: {val_file.id} ({val_file.bytes} bytes)")

    # Wait for processing
    print("Waiting for file processing...")
    time.sleep(10)

    # Step 2: Create fine-tuning job
    print(f"\nCreating fine-tuning job on {BASE_MODEL}...")
    job_params = {
        "model": BASE_MODEL,
        "training_file": train_file.id,
        "suffix": SUFFIX,
    }
    if val_file_id:
        job_params["validation_file"] = val_file_id

    job = client.fine_tuning.jobs.create(**job_params)
    print(f"  Job ID: {job.id}")
    print(f"  Status: {job.status}")

    # Step 3: Monitor progress
    print("\nMonitoring training (Ctrl+C to stop monitoring)...")
    try:
        while True:
            job = client.fine_tuning.jobs.retrieve(job.id)
            status_line = f"  [{job.status}]"
            if job.trained_tokens:
                status_line += f" tokens: {job.trained_tokens}"
            print(status_line)

            if job.status == "succeeded":
                break
            elif job.status == "failed":
                print(f"\n  ERROR: {job.error}")
                sys.exit(1)

            time.sleep(30)
    except KeyboardInterrupt:
        print(f"\n\nMonitoring stopped. Job {job.id} continues in background.")
        print(f"Check status at: https://platform.openai.com/finetune/{job.id}")
        sys.exit(0)

    # Step 4: Test the model
    model_id = job.fine_tuned_model
    print(f"\n{'='*60}")
    print(f"Fine-tuned model ready: {model_id}")
    print(f"{'='*60}")

    print("\nRunning test...")
    response = client.chat.completions.create(
        model=model_id,
        messages=[
            {"role": "system", "content": "You are a phone ordering assistant for a restaurant."},
            {"role": "user", "content": "Hi, I'd like to place an order for pickup."},
        ],
    )
    print(f"\nTest response: {response.choices[0].message.content}")
    print(f"\nUsage: {response.usage}")

if __name__ == "__main__":
    main()
```

Save as `finetune.py` and run:

```bash
# With train/val split
python finetune.py train.jsonl val.jsonl

# With single file (no validation)
python finetune.py finetune_2026-02-12_49examples.jsonl
```

---

## Summary

| Step | Action | Where |
|------|--------|-------|
| 1 | Approve conversations | Annotation Platform `/admin-panel/review/` |
| 2 | Export JSONL | Annotation Platform `/admin-panel/export/` |
| 3 | Validate locally | `python validate_jsonl.py train.jsonl` |
| 4 | Upload to OpenAI | `client.files.create(...)` |
| 5 | Create fine-tuning job | `client.fine_tuning.jobs.create(...)` |
| 6 | Monitor training | Dashboard or `client.fine_tuning.jobs.retrieve(...)` |
| 7 | Test fine-tuned model | `client.chat.completions.create(model="ft:...", ...)` |
| 8 | Deploy | Replace model ID in your application |
