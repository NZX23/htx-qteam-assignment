import unittest

from pipeline_utils import (
    calculate_accuracy,
    evaluate_sentiment_model,
    format_sentiment_labels,
    is_promotion_ready,
)


class FakeTokenBatch:
    def __init__(self, tokens):
        self.tokens = [tokens]
        self.shape = (1, len(tokens))

    def to(self, _device):
        return self

    def __getitem__(self, index):
        row, token_slice = index
        return self.tokens[row][token_slice]


class FakeModel:
    device = "test-device"

    def __init__(self, generated_tokens):
        self.generated_tokens = generated_tokens
        self.generation_kwargs = None
        self.is_eval_mode = False

    def eval(self):
        self.is_eval_mode = True

    def generate(self, **kwargs):
        self.generation_kwargs = kwargs
        return FakeTokenBatch([10, 11, *self.generated_tokens])


class FakeTokenizer:
    pad_token_id = 0

    def __init__(self):
        self.messages_seen = []
        self.decoded_texts = iter(
            ["The review sentiment is positive.", "This review is unclear."]
        )

    def apply_chat_template(self, messages, **kwargs):
        self.messages_seen.append(messages)
        self.template_kwargs = kwargs
        return "formatted test prompt"

    def __call__(self, prompt, **kwargs):
        self.prompt = prompt
        self.tokenize_kwargs = kwargs
        return {"input_ids": FakeTokenBatch([10, 11])}

    def decode(self, token_ids, **kwargs):
        self.decoded_tokens = token_ids
        self.decode_kwargs = kwargs
        return next(self.decoded_texts)


class FormatSentimentLabelsTests(unittest.TestCase):
    def test_formats_imdb_labels_and_preserves_review_text(self):
        example = {
            "messages": [
                {"role": "user", "content": "A useful review"},
                {"role": "assistant", "content": 0},
            ]
        }

        self.assertEqual(
            format_sentiment_labels(example),
            {
                "messages": [
                    {"role": "user", "content": "A useful review"},
                    {"role": "assistant", "content": "negative"},
                ]
            },
        )

    def test_converts_positive_label(self):
        self.assertEqual(
            format_sentiment_labels(
                {"messages": [{"role": "assistant", "content": 1}]}
            )["messages"][0]["content"],
            "positive",
        )


class PromotionGateTests(unittest.TestCase):
    def test_evaluates_generated_model_outputs_against_labels(self):
        model = FakeModel(generated_tokens=[12, 13])
        tokenizer = FakeTokenizer()
        metrics = evaluate_sentiment_model(
            model,
            tokenizer,
            [
                ("A good movie.", "positive"),
                ("A bad movie.", "negative"),
            ],
        )

        self.assertEqual(
            metrics,
            {
                "accuracy": 0.5,
                "num_examples": 2,
                "predictions": ["positive", None],
            },
        )
        self.assertEqual(model.generation_kwargs["max_new_tokens"], 4)
        self.assertFalse(model.generation_kwargs["do_sample"])
        self.assertEqual(tokenizer.decoded_tokens, [12, 13])
        self.assertEqual(
            tokenizer.messages_seen,
            [
                [{"role": "user", "content": "A good movie."}],
                [{"role": "user", "content": "A bad movie."}],
            ],
        )
        self.assertEqual(tokenizer.prompt, "formatted test prompt")
        self.assertFalse(tokenizer.template_kwargs["tokenize"])
        self.assertFalse(tokenizer.tokenize_kwargs["add_special_tokens"])
        self.assertTrue(model.is_eval_mode)

    def test_promotes_at_or_above_minimum_test_accuracy(self):
        self.assertTrue(is_promotion_ready({"accuracy": 0.8}))
        self.assertFalse(is_promotion_ready({"accuracy": 0.799}))

    def test_rejects_missing_or_non_finite_accuracy(self):
        self.assertFalse(is_promotion_ready({}))
        self.assertFalse(is_promotion_ready({"accuracy": float("nan")}))
        self.assertFalse(is_promotion_ready({"accuracy": float("inf")}))
        self.assertFalse(is_promotion_ready({"accuracy": 1.01}))

    def test_accuracy_requires_matching_non_empty_inputs(self):
        with self.assertRaises(ValueError):
            calculate_accuracy(["positive"], [])
        with self.assertRaises(ValueError):
            calculate_accuracy(["positive"], ["positive", "negative"])


if __name__ == "__main__":
    unittest.main()
