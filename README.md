# IMDb LoRA fine-tuning and promotion check

This submission fine-tunes a causal language model to respond to an IMDb review
with its sentiment, then evaluates test-set accuracy before reporting whether
the adapter meets a simple promotion gate.

## Design decisions

- **Dataset and provenance:** The pipeline loads the `stanfordnlp/imdb` training
  split from Hugging Face Datasets. It records the loaded dataset fingerprint
  with every run in `run_metrics.json`, alongside the dataset identifier and
  seed. The fingerprint provides a compact identifier for the dataset content
  used by a run; a separate dataset artifact is not vendored.
- **Data budget and split:** To limit training cost, it samples a deterministic
  20% subset, then makes deterministic 80/10/10 train/validation/test splits.
  IMDb's integer labels are converted to `negative` and `positive` assistant
  targets; review text remains the user message.
- **Model and training:** Qwen/Qwen3.5-2B is fine-tuned with 4-bit quantization
  and LoRA. This keeps the trained parameter count and GPU memory lower than
  full fine-tuning. If `local-qwen3.5-2B-imdb-adapter` already contains a
  complete adapter, the pipeline loads it and skips training; otherwise it
  fine-tunes and saves the adapter there. TensorBoard logs are written locally.
- **Promotion:** A run is marked ready only when generated sentiment labels
  match at least 80% of 50 deterministically selected test examples (or the
  entire test split if it has fewer than 50 rows). Generation is greedy
  (sampling is disabled); unrecognized outputs count as incorrect. The small
  sample keeps evaluation quick, but is only a demonstration gate, not a
  production-quality estimate. Accuracy, evaluated example count, and decision
  are recorded in `qwen3.5-2B-imdb-adapter/run_metrics.json`.
- **Testability:** Preprocessing and the promotion decision live in
  `pipeline_utils.py`. The model evaluation logic is covered by a unit test
  using a lightweight fake model and tokenizer, so tests do not require a GPU
  or external dataset.

## Reproduce

Use Python 3.10+ in an environment with a CUDA-capable GPU and compatible
PyTorch, Transformers, Datasets, PEFT, TRL, Accelerate, and bitsandbytes
installations. Install a PyTorch build matching the local CUDA environment,
then install the remaining libraries:

```powershell
python -m pip install datasets transformers peft trl accelerate bitsandbytes tensorboard
```

The first pipeline run needs internet access (or a populated
Hugging Face cache) to download the dataset and model. For repeatable results,
use the same package versions, model/dataset snapshots, hardware, and seed;
the random seed and all dataset split seeds are fixed to `42` (with derived
seeds for subsequent splits).

Run the unit tests using only the Python standard library:

```powershell
python -m unittest -v
```

Run training and evaluation:

```powershell
python submission_zx.py
```

With the saved adapter present, this loads it in evaluation mode and runs the
test-set promotion check without training. To start a fresh training run,
move the adapter directory out of the way before running the command; the
pipeline then trains and saves it. Training logs/checkpoints are written to
`qwen3.5-2B-imdb-adapter`, and the run summary is saved to
`qwen3.5-2B-imdb-adapter/run_metrics.json`. TensorBoard logs can be viewed with:

```powershell
tensorboard --logdir qwen3.5-2B-imdb-adapter
```

GPU kernels and library versions can introduce small nondeterministic
differences, so the fixed seeds make data selection and random initialization
repeatable under a fixed environment but do not promise bit-for-bit identical
training across different hardware or software builds.

## Assumptions

- IMDb's labels are interpreted as `0 = negative` and `1 = positive`.
- The compute budget permits a 2B parameter checkpoint with 4-bit loading;
  the model and dataset can be downloaded before an offline/on-premises run.
- Test-set accuracy of at least 80% is an acceptable demonstration gate. A
  real release decision would require a baseline and a representative test set.
- Reproduction means rerunning with the same seed and compatible dependencies,
  not guaranteeing identical GPU outputs across environments.

## With more time

I would add a baseline comparison and calibrate the accuracy threshold against
an agreed quality target. I would also pin immutable model and dataset revisions
and capture dependency versions in a lockfile so the provenance record supports
stronger reproduction.