# STEP 1: Collect Hidden States for Confidence Intervention

This document explains how to collect hidden states and original answers from the model using the neutral template.

## Overview

**Goal**: For each MMLUPro sample, extract:
1. **Hidden states** of the last input token (residual stream, all layers: 0-31)
2. **Original answer** (greedy decode: A/B/C/D) under neutral template
3. **Logits & softmax** for analysis

**Template**:
```
Would you answer the following question with A, B, C or D?
Question: {context}
Your answer among "A, B, C, D" is:
```
(No character/role steering - purely neutral)

---

## Data Structure

Each sample will have these new fields:

```json
{
  "task": "abstract_algebra",
  "text": "Question: ... A) ... B) ...",
  "label": 0,
  "num_options": 4,
  "hidden_states": [
    [0.123, -0.456, ...],  // layer 0, last token, 4096 dims
    [0.234, -0.567, ...],  // layer 1
    ...                    // layers 2-30
    [0.234, -0.567, ...]   // layer 31
  ],
  "original_answer": "A",
  "original_logits": [2.34, 1.23, 0.45, -0.12],
  "original_softmax": [0.85, 0.10, 0.04, 0.01]
}
```

---

## How to Run

### Option 1: Local Run (Recommended for Initial Testing)

Use `collect_hidden_states_mmlupro_local.py` if you have the MMLU-Pro JSON file locally.

```bash
python collect_hidden_states_mmlupro_local.py \
  --model_dir "Qwen/Qwen2.5-7B" \
  --input_file "/path/to/mmlupro_test.json" \
  --output_dir "./output" \
  --output_file "mmlupro_with_hidden_states.json"
```

**Arguments**:
- `--model_dir`: Model path (local or HF model ID), default: `Qwen/Qwen2.5-7B`
- `--input_file`: Path to input MMLU-Pro JSON (**required**)
- `--output_dir`: Output directory, default: `./output`
- `--output_file`: Output filename, default: `mmlupro_with_hidden_states.json`
- `--suite`: Prompt suite (`default` or `vanilla`), default: `default`
- `--use_E`: Include refusal label (flag)

**Example**:
```bash
# For Llama3-8B-IT
python collect_hidden_states_mmlupro_local.py \
  --model_dir "meta-llama/Llama-2-7b-chat-hf" \
  --input_file "mmlupro_test.json" \
  --output_dir "./hidden_states_output" \
  --suite "default"
```

### Option 2: Server Run with Base Directory

Use `collect_hidden_states_mmlupro.py` for runs on the server with centralized data paths.

```bash
python collect_hidden_states_mmlupro.py \
  --model "qwen2.5_base" \
  --model_dir "Qwen/Qwen2.5-7B" \
  --hs "qwen2.5" \
  --size "7B" \
  --test_file "mmlupro/mmlupro_test.json" \
  --base_dir "/data2/paveen/RolePlaying/components" \
  --suite "default"
```

**Arguments**:
- `--model`: Model name identifier
- `--model_dir`: Model path
- `--hs`: Hidden state size identifier
- `--size`: Model size (7B, 8B, etc.)
- `--test_file`: Path relative to base_dir (e.g., `mmlupro/mmlupro_test.json`)
- `--base_dir`: Base directory (falls back to `/{data}/paveen/RolePlaying/components`)
- `--use_E`: Include refusal label (flag)
- `--suite`: Prompt suite (`default` or `vanilla`)

**Output location**: `{base_dir}/{model}/hidden_states/mmlupro_test_with_hidden_states.json`

---

## Expected Output

The script will:
1. Load the model once
2. For each task:
   - Extract hidden states for each sample using neutral template
   - Get logits and select original answer
   - Display progress: `Task Name: 100%|█████| 42/42`
3. Save updated JSON with all samples
4. Display summary:
   ```
   ✅ Saved all samples with hidden states to:
      ./output/mmlupro_with_hidden_states.json
      Total samples processed: 12,000
   ```

---

## Key Implementation Details

### Hidden States Extraction

```python
hidden_states_list = vc.get_hidden_states(neutral_prompt)
# Returns: [[(L0_H), (L1_H), ..., (L31_H)]]
# Each (Li_H) is np.ndarray of shape (4096,)
```

The method:
- Tokenizes the prompt
- Runs forward pass with `output_hidden_states=True`
- Extracts hidden states for the last input token position
- Returns all 32 layers (0-31)

### Original Answer Selection

```python
opt_logits = [logits[opt_id] for opt_id in opt_ids]  # Get logits for A, B, C, D
pred_idx = argmax(opt_logits)  # Greedy selection
original_answer = LABELS[pred_idx]  # A, B, C, or D
```

---

## Memory Management

- Uses `torch.no_grad()` to disable gradient computation
- Calls `gc.collect()` and `torch.cuda.empty_cache()` between samples
- Should work on single GPU with ~10-15GB VRAM

---

## Next Steps (For STEP 2)

Once you have hidden states collected, you'll:
1. Build training data structure with:
   - `hidden_states` (from STEP 1)
   - `original_answer` (from STEP 1)
   - `steered_answer_pos` (α=+4, already have)
   - `steered_answer_neg` (α=-4, already have)
   - `ground_truth` (from data)
   - `label_pos`: +1 if Wrong→Correct, -1 if Correct→Wrong, else 0
   - `label_neg`: +1 if Correct→Wrong, -1 if Wrong→Correct, else 0

2. Train confidence-direction probe on these hidden states

---

## Troubleshooting

### Error: "No decoder layers found"
- Check model architecture matches (Llama3, Qwen3, etc.)
- Verify model loads correctly with `transformers`

### Memory Issues
- Reduce batch size (currently 1 sample per iteration)
- Use smaller model or GPU with more VRAM
- Process smaller subsets of data

### Missing Fields
- Ensure input JSON has: `task`, `text`, `label`, `num_options`
- Check templates are loaded correctly (print first prompt)

---

## Files Created

1. **`collect_hidden_states_mmlupro_local.py`** - Local version (recommended)
2. **`collect_hidden_states_mmlupro.py`** - Server version with base_dir support
3. **This README** - Documentation

---

## References

- `VicundaModel.get_hidden_states()` - Core extraction method in `llms.py:921`
- `select_templates_pro()` - Template selection in `template.py`
- Neutral template definition in `template.py:92-96`
