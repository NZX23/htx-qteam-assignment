# TASK 2: build a minimal model fine-tuning and promotion pipeline.
import json
from pathlib import Path

import torch
from datasets import load_dataset
from peft import (
    get_peft_model,
    LoraConfig,
    PeftModel,
    prepare_model_for_kbit_training,
)
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer
from transformers import set_seed

from pipeline_utils import (
    evaluate_sentiment_model,
    format_dataset,
    format_sentiment_labels,
    is_promotion_ready,
)


SEED = 42
MIN_TEST_ACCURACY_FOR_PROMOTION = 0.8
TEST_EVAL_EXAMPLES = 10
DATASET_ID = "stanfordnlp/imdb"
ADAPTER_DIR = Path("local-qwen3.5-2B-imdb-adapter")
OUTPUT_DIR = Path("qwen3.5-2B-imdb-adapter")

# Set seed for reproducibility across random, numpy, and pytorch
set_seed(SEED)


# 1. LOAD DATASET
# loading IMDB dataset from HuggingFace
dataset = load_dataset(DATASET_ID, split="train")
dataset_fingerprint = dataset._fingerprint

# observe the first few rows of the dataset
# print(dataset[:2]) 

# 2. PREPROCESSING
# preprocessing the dataset for sentiment analysis task (reviews, sentiment) --> clean any HTML tags, special characters, and extra whitespace, or missing values

# convert csv dataset into a conversational format: 
    # {"messages":[
    #   {"role": "system", "content": "<general directives>"},
    #   {"role": "user", "content": "<prompt text>"},
    #   {"role": "assistant", "content": "<ideal generated text>"}
    # ]}

dataset = dataset.map(format_dataset)
dataset = dataset.remove_columns(['text', 'label'])
print(len(dataset)) # 25k


# 3. SPLIT DATASET + DATA CLEANING
# split into train, val, test in 80:10:10 ratio 
# due to compute constraints, use a subset (20%) of IMDB for training
dataset = dataset.train_test_split(test_size=0.80, seed=SEED)
dataset = dataset["train"]

dataset = dataset.train_test_split(test_size=0.2, seed=SEED + 1) # 80-20
test_val = dataset['test'].train_test_split(test_size=0.5, seed=SEED + 2) # 20--> 10:10
dataset['test'] = test_val['test']
dataset['validation'] = test_val['train']

# Apply to your dataset splits
dataset["train"] = dataset["train"].map(format_sentiment_labels)
dataset["validation"] = dataset["validation"].map(format_sentiment_labels)
dataset["test"] = dataset["test"].map(format_sentiment_labels)

# filter out empty rows
dataset["train"] = dataset["train"].filter(lambda x: isinstance(x.get("messages"), list) and len(x["messages"]) > 0)
dataset["validation"] = dataset["validation"].filter(lambda x: isinstance(x.get("messages"), list) and len(x["messages"]) > 0)
dataset["test"] = dataset["test"].filter(lambda x: isinstance(x.get("messages"), list) and len(x["messages"]) > 0)


# 4. LOAD MODEL
# loaded model from Huggingface: Qwen3.5-2B
# load quantised version
bnb_config = BitsAndBytesConfig(
   load_in_4bit=True,
   bnb_4bit_quant_type="nf4",
   bnb_4bit_use_double_quant=True,
   bnb_4bit_compute_dtype=torch.float32
)

repo_id = "Qwen/Qwen3.5-2B" 
model = AutoModelForCausalLM.from_pretrained(repo_id, device_map="cuda:0",
quantization_config=bnb_config
)

tokenizer = AutoTokenizer.from_pretrained(repo_id)
tokenizer.pad_token = tokenizer.unk_token
tokenizer.pad_token_id = tokenizer.unk_token_id

# print(tokenizer.chat_template) # prints a complicated series of regex and if else processing for handling special tokens


# 5. FINE-TUNING OR LOADING A TRAINED ADAPTER
adapter_weights = (
    ADAPTER_DIR / "adapter_model.safetensors",
    ADAPTER_DIR / "adapter_model.bin",
)
has_trained_adapter = (
    (ADAPTER_DIR / "adapter_config.json").is_file()
    and any(weight_file.is_file() for weight_file in adapter_weights)
)
if ADAPTER_DIR.exists() and not has_trained_adapter:
    raise RuntimeError(
        f"Found an incomplete adapter at {ADAPTER_DIR}; expected adapter_config.json "
        "and adapter_model.safetensors or adapter_model.bin."
    )

training_log_history = []
if has_trained_adapter:
    print(f"Found trained adapter at {ADAPTER_DIR}; skipping training.")
    model = PeftModel.from_pretrained(model, ADAPTER_DIR)
    trainer_state_path = OUTPUT_DIR / "trainer_state.json"
    if trainer_state_path.is_file():
        with trainer_state_path.open(encoding="utf-8") as state_file:
            training_log_history = json.load(state_file).get("log_history", [])
else:
    print("No trained adapter found; starting fine-tuning.")
    # default parameters from HuggingFace Blog. Given extra time, some hyperparam tuning should be done
    config = LoraConfig(
        r=8, # rank, lower ranks train fewer parameters
        lora_alpha=16,
        bias="none",
        lora_dropout=0.05,
        task_type="CAUSAL_LM",
        target_modules=['o_proj', 'qkv_proj', 'gate_up_proj', 'down_proj'],
    )

    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, config)

    trainable_parms, tot_parms = model.get_nb_trainable_parameters()
    print(f'Trainable parameters:             {trainable_parms/1e6:.2f}M')
    print(f'Total parameters:                 {tot_parms/1e6:.2f}M')
    print(f'Fraction of trainable parameters: {100*trainable_parms/tot_parms:.2f}%')

    # for the current LoRA, we have
    # Trainable parameters:             1.77M
    # Total parameters:                 1883.59M
    # Fraction of trainable parameters: 0.09%

    sft_config = SFTConfig(
        # GROUP 1: Memory usage
        gradient_checkpointing=True, # saves memory
        gradient_checkpointing_kwargs={'use_reentrant': False},
        gradient_accumulation_steps=1,
        per_device_train_batch_size=16,
        auto_find_batch_size=True,

        # GROUP 2: Dataset-related
        max_length=64, 
        packing=True, # no padding
        packing_strategy='wrapped',

        # GROUP 3: These are typical training parameters
        num_train_epochs=2,
        learning_rate=3e-4,
        optim='paged_adamw_8bit',

        # GROUP 4: Logging parameters
        logging_steps=10,
        output_dir=str(OUTPUT_DIR),
        report_to='tensorboard',
    )


    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        args=sft_config,
        train_dataset=dataset["train"],  
        eval_dataset=dataset["validation"],
    )

    trainer.train()
    trainer.save_model(str(ADAPTER_DIR))
    trainer.save_state()
    training_log_history = trainer.state.log_history

# 6. MODEL PROMOTION CHECK
# Evaluate in inference mode, then apply the accuracy-based promotion gate.
with torch.inference_mode():
    test_subset = dataset["test"].select(
        range(min(TEST_EVAL_EXAMPLES, len(dataset["test"])))
    )
    test_examples = [
        (
            next(message["content"] for message in sample["messages"] if message["role"] == "user"),
            next(message["content"] for message in sample["messages"] if message["role"] == "assistant"),
        )
        for sample in test_subset
    ]
    test_results = evaluate_sentiment_model(model, tokenizer, test_examples)
print(f'Test results: {test_results}')
promotion_ready = is_promotion_ready(
    test_results, min_accuracy=MIN_TEST_ACCURACY_FOR_PROMOTION
)
if promotion_ready:
    print("Model is ready for promotion.")
else:
    print("Model did not meet the promotion criteria.")

metrics_path = OUTPUT_DIR / "run_metrics.json"
metrics_path.parent.mkdir(parents=True, exist_ok=True)
numeric_test_results = {
    key: float(value)
    for key, value in test_results.items()
    if isinstance(value, (int, float))
}
with metrics_path.open("w", encoding="utf-8") as metrics_file:
    json.dump(
        {
            "dataset": DATASET_ID,
            "dataset_fingerprint": dataset_fingerprint,
            "seed": SEED,
            "training_log_history": training_log_history,
            "test_metrics": numeric_test_results,
            "promotion": {
                "ready": promotion_ready,
                "minimum_test_accuracy": MIN_TEST_ACCURACY_FOR_PROMOTION,
            },
        },
        metrics_file,
        indent=2,
    )
print(f"Run metrics saved to {metrics_path}")

# Unit Test