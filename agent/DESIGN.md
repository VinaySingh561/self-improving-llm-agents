# Phase 3: Local LLM Inference (CPU-Only)

## Overview

The LLM inference wrapper implements a **frozen reasoning oracle** that:
- Runs locally on CPU with quantized models
- Generates deterministically (reproducible)
- Validates JSON output strictly
- Budgets tokens per episode
- Logs all interactions completely

This design ensures:
- **No GPU dependency**: Works on any machine
- **Reproducibility**: Same prompt → Same output (temperature=0.1, no sampling)
- **Transparency**: Every LLM call is logged
- **Safety**: Token and call limits prevent runaway behavior

---

## Architecture

### 1. LLMConfig

Configuration object for model and generation parameters:

```python
config = LLMConfig(
    model_name="mistralai/Mistral-7B-Instruct-v0.1",  # HuggingFace model ID
    quantization_bits=4,              # 4-bit quantization for CPU
    temperature=0.1,                  # Low temp = deterministic
    top_p=0.95,                       # Top-p sampling (disabled if temp=0)
    max_tokens_per_call=256,          # Max tokens per call
    max_calls_per_episode=100,        # Max calls per episode
    device="cpu",                     # CPU only
    device_map="cpu",
    load_in_4bit=True,                # Use 4-bit quantization
    generation_timeout_sec=60.0,
    max_retries=2,                    # Retry on parse errors
    log_all_calls=True,
    save_logs=True,
)
```

**Key parameters:**

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `quantization_bits` | 4 | Fits on CPU memory, acceptable quality loss |
| `temperature` | 0.1 | Deterministic (close to greedy decoding) |
| `top_p` | 0.95 | Ignored when temp ≈ 0 |
| `max_tokens_per_call` | 256 | Fits JSON skill decisions (50-100 tokens typical) |
| `max_calls_per_episode` | 100 | Sufficient for 5000-step episodes |

---

### 2. LLMWrapper

Main wrapper class:

```python
wrapper = LLMWrapper(config=config, log_dir="/path/to/logs")
```

#### Initialization

```python
# Load model on first instantiation
wrapper = LLMWrapper(config=config, log_dir="/tmp/llm_logs")

# Model loading sequence:
# 1. Download from HuggingFace Hub
# 2. Apply 4-bit quantization (BitsAndBytes)
# 3. Move to CPU
# 4. Verify model parameters
```

#### Model Loading

The wrapper supports:
- **Quantized 4-bit** (recommended for CPU): ~2 GB RAM for 7B model
- **Quantized 8-bit**: ~4 GB RAM for 7B model
- **No quantization**: ~15 GB RAM (not recommended for CPU)

**Configuration for 4-bit:**
```python
config = LLMConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype="float16",
    bnb_4bit_use_double_quant=True,  # Nested quantization
    bnb_4bit_quant_type="nf4",       # NormalForm4 quantization
)
```

#### Making Calls

```python
# Single call
result = wrapper.call(
    prompt="Your prompt here",
    output_format="json"  # or "text" or "skill_decision"
)

# Result structure
{
    "success": True,                    # Parse succeeded
    "output": {"skill": "explore", ...}, # Parsed output
    "error": None,                      # Error message or None
    "metadata": {
        "call_id": 1,
        "response_time_sec": 0.234,
        "input_tokens": 512,
        "output_tokens": 64,
        "total_calls": 1,
    }
}
```

#### Deterministic Generation

```python
# Temperature = 0.1 ensures:
# - Greedy-like decoding (low randomness)
# - Same prompt always produces same output
# - Low variation across runs

result1 = wrapper.call(prompt)  # "skill": "explore"
wrapper.reset_episode_stats()
result2 = wrapper.call(prompt)  # "skill": "explore" (same)
```

**How it works:**
1. `temperature=0.1` → Output logits are scaled down
2. High-probability tokens dominate
3. Sampling is effectively greedy
4. Very low variance in outputs

---

### 3. LLMCallLog

Immutable log of each LLM call:

```python
@dataclass
class LLMCallLog:
    call_id: int                    # Sequential call ID
    timestamp: str                  # ISO format
    prompt_length: int              # Characters
    response_length: int            # Characters
    response_time_sec: float        # Execution time
    tokens_input: int               # Input tokens
    tokens_output: int              # Output tokens
    parsed_output: Dict             # Parsed JSON
    raw_output: str                 # Raw text (truncated)
    valid_json: bool                # JSON valid?
    error_message: Optional[str]    # Parse error
```

All calls logged automatically, saved to JSON:

```json
{
  "config": {
    "model": "mistralai/Mistral-7B-Instruct-v0.1",
    "quantization_bits": 4,
    "temperature": 0.1
  },
  "calls": [
    {
      "call_id": 1,
      "timestamp": "2024-01-21T16:10:00",
      "prompt_length": 1770,
      "response_length": 128,
      "response_time_sec": 0.234,
      "tokens_input": 512,
      "tokens_output": 64,
      "valid_json": true,
      "error_message": null
    },
    ...
  ],
  "summary": {
    "call_count": 42,
    "total_input_tokens": 21504,
    "total_output_tokens": 2688,
    "avg_tokens_per_call": 576
  }
}
```

---

### 4. OutputValidator

Validates and extracts information from LLM output:

```python
# Validate skill decision
output = {
    "skill": "explore",
    "reasoning": "...",
    "confidence": 0.8,
}

is_valid = OutputValidator.validate_skill_decision(output)  # True
skill = OutputValidator.extract_skill(output)  # "explore"

# Validate action sequence
actions_output = {
    "actions": ["move_north", "move_east"],
    "reasoning": "..."
}

is_valid = OutputValidator.validate_action_sequence(actions_output)  # True
actions = OutputValidator.extract_actions(actions_output)  # [...]
```

---

## Workflow: Prompt → LLM → Decision

### Step 1: Prepare Prompt

```python
from prompts import create_base_prompt

base_prompt = create_base_prompt()
prompt_string = base_prompt.to_prompt_string()

# Append current state
game_state = """
CURRENT STATE
- Health: 80/100
- Hunger: 40/100
- Visible: goblin (weak)

Decide next skill (JSON format).
"""

full_prompt = prompt_string + game_state
```

### Step 2: Call LLM

```python
wrapper = LLMWrapper(config=config, log_dir="/tmp/logs")

result = wrapper.call(
    prompt=full_prompt,
    output_format="skill_decision"
)

if result['success']:
    skill = result['output']['skill']
    confidence = result['output'].get('confidence', 0.5)
else:
    print(f"LLM error: {result['error']}")
    skill = "explore"  # Fallback
```

### Step 3: Extract Decision

```python
from agent import OutputValidator

if result['success']:
    skill = OutputValidator.extract_skill(result['output'])
    
    if skill:
        # Execute skill
        execute_skill(skill)
    else:
        # Invalid skill
        logger.error(f"Invalid skill: {result['output']['skill']}")
        execute_skill("explore")  # Fallback
```

### Step 4: Log and Continue

```python
# Logs are automatically saved
stats = wrapper.get_usage_stats()
print(f"Calls remaining: {stats['calls_remaining']}")
print(f"Tokens remaining: {stats['token_budget_remaining']}")

# At episode end
wrapper.save_logs(filepath=f"episode_{episode_id}.json")
wrapper.reset_episode_stats()  # For next episode
```

---

## Token Budgeting

### Per-Episode Budget

```
Budget = max_tokens_per_call × max_calls_per_episode
       = 256 × 100
       = 25,600 tokens per episode
```

### Typical Usage

| Component | Tokens | Count | Total |
|-----------|--------|-------|-------|
| Base prompt | 150 | 100 | 15,000 |
| Game state | 50 | 100 | 5,000 |
| LLM output | 50 | 100 | 5,000 |
| **Total** | | | **25,000** |

Room for 600 extra tokens (retries, long descriptions).

### Monitoring

```python
stats = wrapper.get_usage_stats()
print(f"Calls: {stats['call_count']}/{config.max_calls_per_episode}")
print(f"Tokens: {stats['total_input_tokens'] + stats['total_output_tokens']} / 25,600")

if stats['calls_remaining'] < 10:
    logger.warning("Near call limit")
```

---

## JSON Output Validation

### Skill Decision Format

Expected:
```json
{
    "skill": "explore|fight|flee|eat|use_item",
    "reasoning": "explanation",
    "confidence": 0.0-1.0,
    "parameters": {}
}
```

Validation:
```python
# Must have "skill" and "reasoning"
is_valid = OutputValidator.validate_skill_decision(output)

# Extract and validate skill name
skill = OutputValidator.extract_skill(output)  # None if invalid
```

### Error Handling

```python
# JSON parse error
result = wrapper.call(prompt)
if not result['success']:
    print(f"Parse error: {result['error']}")
    # Retry enabled (max_retries=2)
    # If still fails, return error

# Invalid skill name
if skill not in ["explore", "fight", "flee", "eat", "use_item"]:
    logger.error(f"Invalid skill: {skill}")
    skill = "explore"  # Fallback
```

---

## Determinism and Reproducibility

### Why Deterministic Output Matters

Without determinism:
```
Run 1: "skill": "explore"
Run 2: "skill": "fight"  # Same prompt, different output!
```

This breaks reproducibility and makes debugging impossible.

### How We Ensure Determinism

```python
config = LLMConfig(
    temperature=0.1,        # Critical: low temperature
    top_p=0.95,            # Ignored when temperature ≈ 0
)

# In generation:
# - No sampling (do_sample=False)
# - Greedy decoding from logits
# - Same random seed (set via PyTorch)
```

### Testing Determinism

```python
# Test: Same prompt, multiple runs
prompt = "What is 2+2?"
results = [wrapper.call(prompt)['output'] for _ in range(5)]

# All should be identical
assert all(r == results[0] for r in results)  # ✓
```

---

## Fallback to Mock LLM

If transformers is not installed, wrapper falls back to mock:

```python
# Automatic fallback
wrapper = LLMWrapper()
# → "Failed to load model: No module named 'transformers'"
# → "Falling back to mock LLM for testing"

# Returns plausible mock responses
result = wrapper.call("decide skill")
# {"skill": "explore", "reasoning": "Mock response", ...}
```

**Use cases:**
- Testing pipeline without model
- CI/CD without GPU
- Development and prototyping

---

## Integration with Agent

The agent uses LLMWrapper to decide skills:

```python
from agent import NetHackAgent
from agent.llm_wrapper import LLMWrapper, LLMConfig

config = LLMConfig()
llm = LLMWrapper(config=config, log_dir="/tmp/logs")

agent = NetHackAgent(llm=llm, skill_library=skills, prompt_manager=pm)

# Agent internally calls:
skill = agent.decide_skill(observation)
# → llm.call(prompt, output_format="skill_decision")
# → Validate and extract skill
# → Execute skill
```

---

## Performance Characteristics

### Model: Mistral-7B-Instruct (4-bit quantized)

| Metric | Value | Notes |
|--------|-------|-------|
| Model size (4-bit) | ~2 GB | Fits in CPU RAM |
| Latency | 0.5-2.0 s | Per call (CPU) |
| Tokens/sec | 50-100 | Throughput |
| Memory (peak) | 3-4 GB | With overhead |

### Optimization

```python
# For faster generation:
config = LLMConfig(
    max_tokens_per_call=100,  # Reduce max tokens
    top_p=0.9,                # Smaller search space
)

# For better quality:
config = LLMConfig(
    temperature=0.05,         # More deterministic
    model_name="mistralai/Mistral-7B-Instruct-v0.1",  # Larger models available
)
```

---

## Testing the LLM Wrapper

Run comprehensive tests:

```bash
python agent/test_llm_wrapper.py
```

Tests verify:
- ✓ Configuration defaults
- ✓ Model initialization (with fallback)
- ✓ Skill decision calls
- ✓ Batch sequential calls
- ✓ Output validation
- ✓ Episode call limits
- ✓ Log saving and JSON structure
- ✓ Deterministic generation setup

---

## Troubleshooting

### Issue: "No module named 'transformers'"

**Solution:** Install dependencies
```bash
pip install -r requirements.txt
# Or: pip install transformers torch bitsandbytes
```

### Issue: Slow inference on CPU

**Solution:**
- Use smaller model: `TinyLlama` instead of Mistral
- Reduce `max_tokens_per_call`
- Run multiple agents in parallel (if system allows)

### Issue: Out of memory on CPU

**Solution:**
- Increase quantization: `load_in_4bit=True` (default)
- Reduce model size: `TinyLlama-1.1B` or `Phi-2`
- Use `device_map="cpu"` (already default)

### Issue: Non-deterministic output

**Solution:**
- Check `temperature` is 0.1 or lower
- Verify `do_sample=False` in generation
- Check random seed is set properly

---

## Next Steps

Phase 4 will implement the **Agent + Skills** layer, which:
- Uses LLMWrapper to decide skills
- Executes fixed skill library
- Manages state and failures
- Integrates with NetHack environment

The full loop: Prompt → LLM → Skill → Action → Observation → Repeat
