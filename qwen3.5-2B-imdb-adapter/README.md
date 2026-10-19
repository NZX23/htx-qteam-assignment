---
base_model: Qwen/Qwen3.5-2B
library_name: transformers
model_name: qwen3.5-2B-imdb-adapter
tags:
- generated_from_trainer
- sft
- trl
licence: license
---

# Model Card for qwen3.5-2B-imdb-adapter

This model is a fine-tuned version of [Qwen/Qwen3.5-2B](https://huggingface.co/Qwen/Qwen3.5-2B).
It has been trained using [TRL](https://github.com/huggingface/trl).

## Quick start

```python
from transformers import pipeline

question = "If you had a time machine, but could only go to the past or the future once and never return, which would you choose and why?"
generator = pipeline("text-generation", model="None", device_map="auto")
output = generator([{"role": "user", "content": question}], max_new_tokens=128, return_full_text=False)[0]
print(output["generated_text"])
```

## Training procedure

 



This model was trained with SFT.

### Framework versions

- TRL: 1.16.0
- Transformers: 5.20.1
- Pytorch: 2.8.0+cu129
- Datasets: 5.1.0
- Tokenizers: 0.23.3

## Citations



Cite TRL as:
    
```bibtex
@software{vonwerra2020trl,
  title   = {{TRL: Transformers Reinforcement Learning}},
  author  = {von Werra, Leandro and Belkada, Younes and Tunstall, Lewis and Beeching, Edward and Thrush, Tristan and Lambert, Nathan and Huang, Shengyi and Rasul, Kashif and GallouÃ©dec, Quentin},
  license = {Apache-2.0},
  url     = {https://github.com/huggingface/trl},
  year    = {2020}
}
```