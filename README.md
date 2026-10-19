# Q Team LLM Deployment Assignment: Ng Zheng Xun
## Option 2: IMDb LoRA fine-tuning and promotion check

This submission fine-tunes a causal language model with limited compute to respond to an IMDb review
with its sentiment, then evaluates test-set accuracy before reporting whether
the adapter meets a simple promotion gate. Several techniques (Quantisation, LoRA) are utilised to minimise compute requirements.

## Design decisions

- **Dataset and Logging:** The pipeline loads the `stanfordnlp/imdb` training
  split from Hugging Face Datasets. It records the loaded dataset fingerprint
  with every run in `run_metrics.json`, alongside the dataset identifier and
  seed. The fingerprint provides a compact identifier for the dataset content
  used by a run; a separate dataset artifact is not vendored.
  
- **Data Budgeting and Preprocessing:** To limit training cost, it samples a deterministic
  20% subset, then makes deterministic 80/10/10 train/validation/test splits.
  IMDb's integer labels are converted to `negative` and `positive` assistant natural language
  targets.
  
- **Model and Finetuning:** Qwen/Qwen3.5-2B is fine-tuned with 4-bit quantization
  and LoRA. This keeps the trained parameter count and GPU memory lower than
  full fine-tuning. If `local-qwen3.5-2B-imdb-adapter` already contains a
  complete adapter (ie finetuning has already been done previously), the pipeline loads it and skips training. Otherwise, it
  fine-tunes and saves the adapter there. 
  
- **Model Promotion:** A run is marked ready only when generated sentiment labels
  match at least 80% of 50 deterministically selected test example, with unrecognized outputs counting as incorrect. The small
  sample keeps evaluation quick, but is only a demonstration gate, not a
  production-quality estimate. Accuracy, evaluated example count, and decision
  are recorded in `qwen3.5-2B-imdb-adapter/run_metrics.json`.
  
- **Testability:** Preprocessing and the promotion decision live in
  `pipeline_utils.py`. The model evaluation logic is covered by a unit test
  using a lightweight fake model and tokenizer, so tests do not require a GPU
  or external dataset.

## Reproducing this Experiment/Project:

Use Python 3.10+ in an environment with a CUDA-capable GPU and compatible
PyTorch, Transformers, Datasets, PEFT, TRL, Accelerate, and bitsandbytes
installations. First install a PyTorch build matching the local CUDA environment,
then install the remaining libraries:

```powershell
python -m pip install datasets transformers peft trl accelerate bitsandbytes tensorboard
```

The first pipeline run needs internet access (or a populated
Hugging Face cache) to download the dataset and model. For repeatable results, use the same random seed (set to `42` in current project).

Run the unit tests using only the Python standard library:

```powershell
python -m unittest -v
```

Run training and evaluation:

```powershell
python submission_zx.py
```

To start a fresh training run, move the adapter directory out of the way before running the command; the
pipeline then trains and saves it. Training logs/checkpoints are written to
`qwen3.5-2B-imdb-adapter`, and the run summary is saved to
`qwen3.5-2B-imdb-adapter/run_metrics.json`. 


## Assumptions

- IMDb's labels are interpreted as `0 = negative` and `1 = positive`.
- Test-set accuracy of at least 80% is an acceptable threshold for deployment.

## With more time

I would add a baseline comparison and calibrate the accuracy threshold against
an agreed quality target. 

I would also do some additional hyperparameter tuning for the best results (LoRA Config)

I would also investigate better usages of data to improve the results of the model (Data Augmentation, Synthetic Data Generation etc)
