# Run an audit on your own data

This guide starts after the [README installation](../README.md). Use **Linux Bash**, with the environment activated and the current directory set to the repository root. On Windows, use WSL and create a Linux environment inside WSL; do not reuse a Windows virtual environment.

Run the blocks in order in the same shell. Stop if a command fails. The main audit needs one locally held claimed model, a legitimate calibration pool, and held-out responses. A substitute model is needed only for the optional power experiment.

## 1. Choose a run directory and reference model

Choose an unused directory. To start another run, change only `RUN_DIR` and repeat the setup. Do not rerun completed output-producing blocks in the same directory.

```bash
export RUN_DIR=work/audit1
python - <<'PY'
import os
from pathlib import Path

run = Path(os.environ["RUN_DIR"])
run.mkdir(parents=True, exist_ok=False)
print("Run directory:", run)
PY
```

Install the optional scoring dependencies if you have not already done so. Use a PyTorch build suitable for your GPU. Set `CLAIMED_MODEL` to a checkpoint you have obtained under its model terms, with its tokenizer and a chat template supported by Transformers.

```bash
python -m pip install -e ".[score]"
export CLAIMED_MODEL=/path/to/your/claimed-checkpoint
export DEVICE=cuda:0
export DTYPE=bfloat16
export CHAT_KWARGS='{"enable_thinking": false}'
export MAX_LENGTH=4096

MODEL_ARGS=(--model "$CLAIMED_MODEL" --device "$DEVICE" --dtype "$DTYPE"
            --chat-kwargs "$CHAT_KWARGS" --local-files-only)
apitail doctor --gpu
```

Replace the checkpoint path before continuing. These commands load local files only. No model weights are bundled or downloaded by the guide. Obtain a compatible checkpoint from its distributor; [ASSETS.md](ASSETS.md) describes the separate assets.

The tested 27B model used about 50 GiB of GPU memory on short inputs; longer inputs can require more. For a small model on CPU, set `DEVICE=cpu` and `DTYPE=float32` **before defining `MODEL_ARGS`**. Choose chat kwargs appropriate for your tokenizer, such as `'{}'` when no extra options are needed. Keep the claimed model and scoring settings fixed throughout the run.

## 2. Supply independent calibration and test prompts

Prepare two UTF-8 JSONL files with **at least 400 distinct prompts each**. Each line is an object with a unique `pid` and nonempty `text`, for example:

```json
{"pid": "cal-001", "text": "Explain why a rainbow forms."}
```

Use different prompt content in calibration and testing, not the same prompts with new IDs. The three bundled example prompts are too small for this workflow.

Copy your files into the run directory, replacing the two input paths below:

```bash
cp /path/to/your/cal.jsonl "$RUN_DIR/cal-prompts.jsonl"
cp /path/to/your/test.jsonl "$RUN_DIR/test-prompts.jsonl"
```

Check format, minimum size, ID separation and repeated text before doing model work:

```bash
python - <<'PY'
import os
from pathlib import Path
from apitail.io import prompt_map

run = Path(os.environ["RUN_DIR"])
cal = prompt_map(run / "cal-prompts.jsonl")
test = prompt_map(run / "test-prompts.jsonl")
if min(len(cal), len(test)) < 400:
    raise ValueError("Provide at least 400 distinct prompts in each file")
if set(cal) & set(test):
    raise ValueError("Calibration and test prompt IDs overlap")
normalize = lambda text: " ".join(text.split()).casefold()
cal_text = {normalize(row["text"]) for row in cal.values()}
test_text = {normalize(row["text"]) for row in test.values()}
if len(cal_text) != len(cal) or len(test_text) != len(test):
    raise ValueError("Duplicate prompt content within a file")
if cal_text & test_text:
    raise ValueError("Calibration and test prompt content overlaps")
print("Prompts checked:", len(cal), "calibration;", len(test), "test")
PY
```

This catches exact text duplicates after whitespace/case normalization. Check for paraphrased duplicates when constructing the datasets.

## 3. Generate and calibrate a legitimate pool

The following recipe uses two local generation temperatures, 0.7 and 1.0, with two responses per calibration prompt. It is an executable starting pool, not the full paper pool. Add legitimate configurations relevant to your deployment before interpreting the resulting audit.

Create the generation recipes and a 200-request calibration:

```bash
python - <<'PY'
import copy
import json
import os
from pathlib import Path

run = Path(os.environ["RUN_DIR"])
base = json.loads(Path("configs/generation.example.json").read_text(encoding="utf-8"))
for name, temperature, responses in [
    ("t07", 0.7, 2), ("t10", 1.0, 2), ("test", 0.7, 1)
]:
    cfg = copy.deepcopy(base)
    cfg["responses_per_prompt"] = responses
    cfg["max_length"] = int(os.environ["MAX_LENGTH"])
    cfg["sampling"]["temperature"] = temperature
    (run / ("gen-" + name + ".json")).write_text(
        json.dumps(cfg, indent=2), encoding="utf-8")
pool = {
    "budget": 200, "trials": 2000, "alpha": 0.05, "seed": 20261025,
    "method": "align", "z_cutoff": 3.0,
    "members": {"temperature_0.7": "cal-t07.npz",
                "temperature_1.0": "cal-t10.npz"}
}
(run / "pool.json").write_text(json.dumps(pool, indent=2), encoding="utf-8")
PY
```

Generate replies, score them under the claimed reference model, then calibrate:

```bash
for tag in t07 t10; do
  apitail generate "${MODEL_ARGS[@]}" --prompts "$RUN_DIR/cal-prompts.jsonl" \
    --config "$RUN_DIR/gen-$tag.json" --output "$RUN_DIR/cal-$tag.jsonl" || break
  apitail score "${MODEL_ARGS[@]}" --prompts "$RUN_DIR/cal-prompts.jsonl" \
    --responses "$RUN_DIR/cal-$tag.jsonl" --output "$RUN_DIR/cal-$tag.npz" \
    --max-length "$MAX_LENGTH" || break
done
```

Continue only when both score files were produced successfully:

```bash
apitail calibrate --config "$RUN_DIR/pool.json" \
  --output "$RUN_DIR/calibration.json"
```

Pool-member paths resolve relative to `pool.json`. Each calibration trial selects 200 distinct prompts and one response per prompt. If you change the budget, keep it no larger than the distinct-prompt count in every source, and use that same budget for the final audit batch.

## 4. Obtain the batch to audit: choose A or B

Both routes produce one response for each of the first 200 held-out prompts. Choose one route, then continue to step 5.

### A. Collect from an API

Create an endpoint configuration:

```bash
cp configs/endpoint.example.json "$RUN_DIR/endpoint.json"
python - <<'PY'
import json
import os
from pathlib import Path

path = Path(os.environ["RUN_DIR"]) / "endpoint.json"
cfg = json.loads(path.read_text(encoding="utf-8"))
cfg["api_key_env"] = "APITAIL_API_KEY"
cfg["responses_per_prompt"] = 1
path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
PY
```

Open `$RUN_DIR/endpoint.json` in your text editor:

- Set `url` to your provider's OpenAI-compatible chat-completions URL. The template uses OpenRouter.
- Replace `model` with the provider's actual model ID.
- Keep `api_key_env` as `APITAIL_API_KEY`; it names an environment variable, not the secret itself.
- Check the generation parameters and leave `responses_per_prompt` at one.

Inspect the request plan first. This command sends no requests and needs no key:

```bash
apitail collect --config "$RUN_DIR/endpoint.json" \
  --prompts "$RUN_DIR/test-prompts.jsonl" --limit 200 \
  --output "$RUN_DIR/api.jsonl"
```

After reviewing the plan, enter your key without echoing it or putting its value in shell history, then explicitly send the requests:

```bash
read -rs -p "API key: " APITAIL_API_KEY
printf '\n'
export APITAIL_API_KEY
apitail collect --config "$RUN_DIR/endpoint.json" \
  --prompts "$RUN_DIR/test-prompts.jsonl" --limit 200 \
  --output "$RUN_DIR/api.jsonl" --execute
unset APITAIL_API_KEY
```

This sends 200 prompts to the configured provider and may incur charges. On failure, stop and retain any `.partial` file; the collector does not resume it or retry ambiguous requests automatically. Choose a new output/run directory before another collection attempt.

After successful collection:

```bash
export RESPONSES="$RUN_DIR/api.jsonl"
```

### B. Generate a known local source instead

This route makes no API calls. It generates the audit batch with the claimed model:

```bash
apitail generate "${MODEL_ARGS[@]}" --prompts "$RUN_DIR/test-prompts.jsonl" \
  --config "$RUN_DIR/gen-test.json" --limit 200 \
  --output "$RUN_DIR/audit-local.jsonl"
```

After successful generation:

```bash
export RESPONSES="$RUN_DIR/audit-local.jsonl"
```

## 5. Score and audit the selected batch

```bash
apitail score "${MODEL_ARGS[@]}" --prompts "$RUN_DIR/test-prompts.jsonl" \
  --responses "$RESPONSES" --output "$RUN_DIR/audit-scores.npz" \
  --max-length "$MAX_LENGTH"
apitail audit --scores "$RUN_DIR/audit-scores.npz" \
  --calibration "$RUN_DIR/calibration.json" --output "$RUN_DIR/audit.json"
```

Read the main output fields:

```bash
python - <<'PY'
import json
import os
from pathlib import Path

result = json.loads(
    (Path(os.environ["RUN_DIR"]) / "audit.json").read_text(encoding="utf-8"))
print({key: result[key] for key in ["reject", "count", "threshold", "temperature", "n"]})
PY
```

`reject` is the alarm: it is true exactly when `count > threshold`. `count` is the number of responses above the standardized-score cutoff; `temperature` is the fitted reference temperature; `n` is the batch size. The decision uses the supplied legitimate pool, reference model, scoring settings and budget. An alarm identifies a batch inconsistent with that calibration; it does not by itself establish that the provider substituted a model. No alarm means this batch did not trigger the rule.

The audit requires exactly 200 responses from distinct test prompts in this recipe. It rejects calibration-ID overlap and incompatible scoring settings. Do not label the result from an unknown API as an honest false-alarm rate or substitution power.

## Optional: measure false alarms and simulated power on known sources

This experiment requires a second, known substitute checkpoint. It uses all held-out prompts, generates both sources locally, and scores both with the **claimed** reference model. It is separate from the API audit above.

```bash
export SUBSTITUTE_MODEL=/path/to/your/substitute-checkpoint
export SUB_CHAT_KWARGS='{}'
apitail generate "${MODEL_ARGS[@]}" --prompts "$RUN_DIR/test-prompts.jsonl" \
  --config "$RUN_DIR/gen-test.json" --output "$RUN_DIR/power-honest.jsonl"
apitail generate --model "$SUBSTITUTE_MODEL" --device "$DEVICE" --dtype "$DTYPE" \
  --chat-kwargs "$SUB_CHAT_KWARGS" --local-files-only \
  --prompts "$RUN_DIR/test-prompts.jsonl" --config "$RUN_DIR/gen-test.json" \
  --output "$RUN_DIR/power-substitute.jsonl"
for source in honest substitute; do
  apitail score "${MODEL_ARGS[@]}" --prompts "$RUN_DIR/test-prompts.jsonl" \
    --responses "$RUN_DIR/power-$source.jsonl" --output "$RUN_DIR/power-$source.npz" \
    --max-length "$MAX_LENGTH" || break
done
```

Replace the substitute path and choose its chat kwargs before running. Continue only after both score banks are complete and cover the same test prompt IDs:

```bash
apitail evaluate --scores "$RUN_DIR/power-honest.npz" \
  --calibration "$RUN_DIR/calibration.json" --trials 1000 --seed 1234 \
  --output "$RUN_DIR/known-false-alarm.json"
apitail evaluate --scores "$RUN_DIR/power-honest.npz" \
  --substitute "$RUN_DIR/power-substitute.npz" --rho 0.1 \
  --calibration "$RUN_DIR/calibration.json" --trials 1000 --seed 1234 \
  --output "$RUN_DIR/simulated-power.json"
```

In `known-false-alarm.json`, `kind` is `honest_endpoint_false_alarm` and `rate` is the rejection fraction for the known local source. In `simulated-power.json`, `kind` is `simulated_partial_substitution_power`; `rate` is the rejection fraction when each request is independently assigned to the substitute with probability `rho=0.1`. These rates describe the chosen local sources and simulated batches.

## Scoring settings and other methods

The HF scorer processes one response at a time, with float32 exponentials/products and float64 sums. It rejects overlength inputs rather than truncating them. Increase `MAX_LENGTH` only within the model's supported context and use the same value for every score bank.

To try `mle`, `fixed`, `huber` or `nucleus`, start a new run and change `method` in the pool recipe. For nucleus, add `--nucleus` to every scoring command and inspect `nucleus_overflow`: the top-4,096 candidate set may fail to cover 95% mass. Full details and baseline interfaces are in [SCHEMAS.md](SCHEMAS.md) and [DESIGN_MAP.md](DESIGN_MAP.md).
