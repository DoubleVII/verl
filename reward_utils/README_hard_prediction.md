# Hard prediction RL reward

The reward aligns source span locations and reference ordinal difficulty levels;
it does not estimate the current translation model's failure probability.

Position weights and Hungarian assignment reproduce GRRM `eval/span_proxy.py`:
character units, asymmetric overlap, `gamma=0.5`, `under_gamma=0.25`.
Difficulty never influences assignment, including tie breaking. For matched edges:

```
q = exp(-abs(predicted_score - reference_score) / difficulty_tau)
joint_reward = 2 * sum(w * (1 - difficulty_weight + difficulty_weight * q)) / (n + m)
s(d) = d / 10
miss_penalty = miss_penalty_weight * mean(s(reference_score_j) * (1 - t_j))
score = (joint_reward - miss_penalty) * score_scale_factor
```

Here `t_j` is the position weight of the matched reference edge, or zero when
unmatched. The miss penalty uses reference difficulty only and applies to valid
outputs with nonempty references. With no references the penalty is zero.
Matching remains position-only, including when multiple assignments tie.

Defaults: `difficulty_weight=0.3`, `difficulty_tau=2.0`, `miss_penalty_weight=0.2`.
The latter must be finite and nonnegative; setting it to zero restores the previous
reward. These are initial training
choices, not empirically established optimum values. No separate text-score reward
or confidence discount is used. High confidence (`w >= 0.8`) is diagnostic only.
The linear difficulty weight expresses a task preference, not a measured loss ratio.

`score_scale_factor=1.0` multiplies the final reward, including the invalid-output
penalty and both-empty reward. It must be finite and nonnegative; zero disables
reward contributions. Group reward sums use the same scale; localization metrics,
difficulty errors and counts remain unscaled.

Before scaling, invalid outputs score -1, without additional miss penalties.
Valid outputs score 1 when both span lists are empty and 0 when only the reference
is empty. For an empty prediction and nonempty reference, reward is
`-miss_penalty_weight * mean(reference_score / 10)`: a single difficulty 1, 9 or 10
reference scores -0.02, -0.18 or -0.20 with the default weight.
Valid scores lie in `[-miss_penalty_weight, 1]` before scaling.
Exact repeated quotes invalidate the output;
overlapping quotes with different boundaries consume separate prediction slots.
Quotes must be exact source substrings, at most 48 characters / 8 whitespace words
by default (sample `extra_info` may override these limits). The overall score must
equal the maximum span score, or zero for no spans. `analysis` and each span's
`explanation` must be nonempty. JSON scores must be integers, excluding booleans.

The default JSON parser accepts bare JSON, one complete final JSON fence, or native thinking
followed by either format. If `</think>` is present only the suffix is parsed.
It never extracts JSON from arbitrary prose or repairs output. Corrupt reference
labels and configuration raise errors rather than penalizing the policy.

## Prepare one input file

Run from the GRRM checkout, using its Python environment:

```bash
.venv/bin/python -m scripts.prepare_RL_hard_prediction_training_data \
  --data_path=/home/nfs06/yangs/data/parquet_data/oss_hard_annotations/tower_all.oss_notes.difficulty5.subset2000.oss.difficulty_annotations.json \
  --output_path=/path/to/hard.train.parquet
```

The script accepts one concrete JSON path, with no directory scanning, merging,
or split creation. Only valid `status=ok` references are retained, including empty
span lists. `max_samples` caps retained samples before seeded shuffling; the default
seed is 114514. `include_calibration=False` and `examples_per_score=2` control the
shared SFT prompt. The script adds the output schema and artifact-specific limits.
No annotation, reference translation, comment or provenance enters the input prompt.
If calibration examples are enabled, those fixed examples are included explicitly.

Output columns are `data_source`, `prompt`, `ability`, `reward_model`, `extra_info`.
`ground_truth` is the strict reference result serialized as JSON; `extra_info`
contains the source, language direction, provenance, prompt version and span limits.
The conversion prints filtering counts and saved empty/nonempty reference counts.

## Configure verl

From the verl checkout, add these overrides to the usual training command:

```bash
reward_model.reward_manager=batch \
custom_reward_function.path=reward_utils/hard_prediction_reward.py \
custom_reward_function.name=compute_score_batch \
+custom_reward_function.reward_kwargs.difficulty_weight=0.3 \
+custom_reward_function.reward_kwargs.difficulty_tau=2.0 \
+custom_reward_function.reward_kwargs.miss_penalty_weight=0.2 \
+custom_reward_function.reward_kwargs.print_batch_summary=True \
+custom_reward_function.reward_kwargs.score_scale_factor=1.0
```

The single-sample `compute_score` entry point also supports the usual verl scalar
reward managers. Each entry point returns a dictionary with numeric diagnostics.
Batch input lengths must agree. No external judge, model, or GRRM import is needed.

### Instruct-model prompts

The data script also accepts `--prompt_type=inst`, reusing exactly the direct-mode
prompt builder used by GRRM `inference/run_inst_hard_prediction.py`. Select
`--prompt_variant=full`, `full_schema`, `compact6`, `compact3`, or
`analysis_codeblock`. The instruct default is `full`; existing calls keep using
`prompt_type=sft`. `include_calibration` only controls SFT prompts: instruct prompts
always include their builder's calibration examples. Compact variants use one
example per anchor score; the others respect `examples_per_score`.

For analysis followed by final JSON, add these options to data preparation:

```bash
--prompt_type=inst --prompt_variant=analysis_codeblock
```

The data's `extra_info` records `prompt_type`, `prompt_variant` and `response_format`.
The reward reads that format automatically (older records default to `json`).
To explicitly choose a parser via verl `reward_kwargs`, use:

```bash
+custom_reward_function.reward_kwargs.response_format=analysis_codeblock
```

An explicit `response_format` overrides sample metadata. Supported values are
`json` and `analysis_codeblock`. The latter requires nonempty analysis under
`# Step-by-step Analysis`, then `# Final JSON` and a final complete `json` fence,
matching GRRM's response envelope. Additional JSON fences in the analysis and
trailing text are invalid. Native thinking is removed before either parser runs.
Both parsers apply the same strict annotation checks and reward formula.

## Interpret diagnostics

- `parser_pass` means the selected parser decoded a JSON object; `valid_output`
  additionally requires all strict annotation checks. A decoded list is a parser
  failure; an object with missing or invalid fields is a validation failure.
- `joint_reward`, `difficulty_penalty`, `miss_penalty`, `invalid_output_penalty`
  are unscaled components. Invalid outputs have zero joint reward, localization
  metrics and difficulty/miss penalties, and an invalid-output penalty of 1.
  Every result satisfies:

  ```
  joint_reward = span_f1 - difficulty_penalty
  score = score_scale_factor * (joint_reward - miss_penalty - invalid_output_penalty)
  ```

- `span_precision`, `span_recall`, `span_f1` use soft position counts.
- `prediction_count`, `reference_count`, `match_count` track output volume.
- `difficulty_mae`, `difficulty_bias` report unweighted matched-edge errors;
  bias is predicted minus reference. Their `high_confidence_` variants restrict
  diagnostics to position weights at least 0.8. Error means are zero for no matches:
  always interpret them alongside the corresponding match count.
- For pooled difficulty metrics, divide summed `difficulty_abs_error_sum` or
  `difficulty_error_sum` by summed `match_count`. For high-confidence errors use
  `high_confidence_abs_error_sum` / `high_confidence_error_sum` and
  `high_confidence_match_count`.
- Divide summed `empty_reference_reward_sum` by `empty_reference_count`, and
  likewise for `nonempty_reference_*`, to obtain group rewards including invalid
  outputs. These group counts also expose the empty-reference proportion.
- Divide `false_positive_nonempty_count` by `valid_empty_reference_count` and
  `false_negative_empty_count` by `valid_nonempty_reference_count` for sentence
  false-positive / false-negative rates among valid outputs. Report `valid_output`
  coverage too. Invalid outputs have zero localization/count diagnostics and are
  excluded from these two valid-output rates. Zero denominators mean unavailable.

## Batch summary logging

`compute_score_batch` prints one JSON line prefixed `[hard_prediction_reward]`
per successful invocation, with `flush=True`. Set
`+custom_reward_function.reward_kwargs.print_batch_summary=False` to disable it.
The single-sample function does not print. Logs summarize only the current local
batch, without cross-worker aggregation, source text or model responses.

- `batch_size`, `parser_pass_count`, `valid_output_count`, `parser_pass_pct`,
  `valid_output_pct`, `parser_failure_count`, `validation_failure_count` distinguish
  extraction failures from strict-validation failures. Percentages range from 0 to 100.
- `batch_mean` averages the final scaled `score` and unscaled reward components
  over all samples, including invalid outputs; the reward decomposition above also
  holds for these batch means. `valid_mean` reports localization, reward components
  and prediction/reference counts averaged over strictly valid samples only.
- `matched` and `high_confidence` report counts, pooled difficulty MAE and signed
  bias. They divide summed errors by the total matching count, rather than taking
  the average of per-sample error means.
- `empty_reference` and `nonempty_reference` contain sample count, mean scaled
  reward (including invalid outputs), valid-output count, sentence error count and
  sentence error rate. The error is a nonempty prediction for an empty reference,
  or an empty prediction for a nonempty reference. Rates use the corresponding
  valid-output count and range from 0 to 1.
- `parameters` records β (`miss_penalty_weight`), λ (`difficulty_weight`), τ
  (`difficulty_tau`) and `score_scale_factor`.

Unavailable means/rates serialize as JSON `null`, including on an empty batch,
which returns `[]` and still prints its zero-count summary. Per-sample numeric
diagnostics retain zero values for unavailable errors for verl compatibility.
Class balance and exploration can still favor always-empty or always-nonempty
policies; inspect group rewards and sentence error rates alongside mean reward.
