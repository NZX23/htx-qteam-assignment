import math
import re
from numbers import Real

# Helper functions for submission_zx.py

def format_dataset(examples):
    if isinstance(examples["text"], list):
        output_texts = []
        for i in range(len(examples["text"])):
            converted_sample = [
                {"role": "user", "content": examples["text"][i]},
                {"role": "assistant", "content": examples["label"][i]},
            ]
            output_texts.append(converted_sample)
        return {'messages': output_texts}
    else:
        converted_sample = [
            {"role": "user", "content": examples["text"]},
            {"role": "assistant", "content": examples["label"]},
        ]
        return {'messages': converted_sample}


def format_sentiment_labels(example):
    cleaned_messages = []

    for message in example["messages"]:
        role = message.get("role")
        content = message.get("content")

        if role == "assistant":
            if content in (-1, 0):
                content = "negative"
            elif content == 1:
                content = "positive"
            else:
                content = str(content)
        else:
            content = str(content)

        cleaned_messages.append({"role": role, "content": content})

    return {"messages": cleaned_messages}


def extract_sentiment(text):
    matches = re.findall(r"\b(positive|negative)\b", text.lower())
    return matches[0] if matches else None


def predict_sentiment(model, tokenizer, review, max_new_tokens=4):
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": review}],
        tokenize=False,
        add_generation_prompt=True,
    )
    tokenized_prompt = tokenizer(
        prompt,
        return_tensors="pt",
        add_special_tokens=False,
    )
    input_ids = tokenized_prompt["input_ids"].to(model.device)
    generated_ids = model.generate(
        input_ids=input_ids,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
    )
    generated_text = tokenizer.decode(
        generated_ids[0, input_ids.shape[-1]:],
        skip_special_tokens=True,
    )
    return extract_sentiment(generated_text)


def calculate_accuracy(predictions, labels):
    if len(predictions) != len(labels):
        raise ValueError("Predictions and labels must have the same length.")
    if not labels:
        raise ValueError("Accuracy is undefined for an empty test set.")

    return sum(prediction == label for prediction, label in zip(predictions, labels)) / len(labels)


def evaluate_sentiment_model(model, tokenizer, examples):
    model.eval()
    predictions = [
        predict_sentiment(model, tokenizer, review)
        for review, _ in examples
    ]
    labels = [label for _, label in examples]
    return {
        "accuracy": calculate_accuracy(predictions, labels),
        "num_examples": len(labels),
        "predictions": predictions,
    }


def is_promotion_ready(metrics, min_accuracy=0.8):
    accuracy = metrics.get("accuracy")
    return (
        isinstance(accuracy, Real)
        and math.isfinite(accuracy)
        and 0 <= accuracy <= 1
        and isinstance(min_accuracy, Real)
        and math.isfinite(min_accuracy)
        and 0 <= min_accuracy <= 1
        and accuracy >= min_accuracy
    )
