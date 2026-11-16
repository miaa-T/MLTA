"""Create a tiny random Qwen-style model + tokenizer in tests/tiny.
Used by tests/smoke_test.sh to check the code without any download or GPU."""
from pathlib import Path

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
from transformers import AutoModelForCausalLM, PreTrainedTokenizerFast, Qwen2Config

import common as C

out = Path(__file__).parent / "tiny"
corpus = [m["content"] for ex in C.medical_examples(200, synthetic=True) +
          C.general_examples(200, synthetic=True) for m in ex]
corpus += ["system user assistant You are a helpful assistant. Passages from medical textbooks that may help: [1] [2] [3] passage about organs A B C D"]
special = ["<|endoftext|>", "<|im_start|>", "<|im_end|>"]
# byte-level BPE, the same kind of tokenizer as the real Qwen models
t = Tokenizer(models.BPE())
t.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
t.decoder = decoders.ByteLevel()
t.train_from_iterator(corpus, trainers.BpeTrainer(
    vocab_size=600, special_tokens=special, initial_alphabet=pre_tokenizers.ByteLevel.alphabet()))
tok = PreTrainedTokenizerFast(tokenizer_object=t, eos_token="<|im_end|>", pad_token="<|endoftext|>")
tok.chat_template = (
    "{% for m in messages %}<|im_start|>{{ m['role'] }}\n{{ m['content'] }}<|im_end|>\n{% endfor %}"
    "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}")
tok.save_pretrained(out)

cfg = Qwen2Config(vocab_size=len(tok), hidden_size=32, intermediate_size=64, num_hidden_layers=4,
                  num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=2048,
                  eos_token_id=tok.eos_token_id, pad_token_id=tok.pad_token_id)
AutoModelForCausalLM.from_config(cfg).save_pretrained(out)

from transformers import AutoTokenizer
check = AutoTokenizer.from_pretrained(out)
text = "Question: Which organ ?\nA. lung\nAnswer:"
assert check.decode(check(text, add_special_tokens=False)["input_ids"]) == text, "tokenizer round trip failed"
assert len(check("A", add_special_tokens=False)["input_ids"]) == 1
print("tiny model saved to", out, "| vocab", len(tok))
