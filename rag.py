"""Retrieval-augmented generation (RAG) on MedQA.

  python rag.py build                              (once: index the textbooks)
  python rag.py eval --name base                   (MedQA with and without passages)
  python rag.py eval --name lora --run runs/lora

The answer is the option letter (A-D) the model finds most likely as its
first word. Both scores, without and with passages, use this same method,
so their difference is the effect of retrieval alone.
Writes results/rag/<name>.json."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

import common as C

EMBEDDER = "BAAI/bge-small-en-v1.5"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
INDEX_DIR = Path("rag_index")


class Retriever:
    def __init__(self, fake_passages=None):
        import faiss
        if fake_passages:                       # test mode: random vectors
            self.texts, self.model = fake_passages, None
            self.index = faiss.IndexFlatIP(16)
            self.index.add(np.random.default_rng(0).random((len(self.texts), 16), dtype=np.float32))
        else:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(EMBEDDER)
            self.index = faiss.read_index(str(INDEX_DIR / "index.faiss"))
            self.texts = json.loads((INDEX_DIR / "texts.json").read_text())

    def search(self, queries, k=3):
        if self.model is None:
            q = np.random.default_rng(1).random((len(queries), 16), dtype=np.float32)
        else:
            q = self.model.encode([QUERY_PREFIX + x for x in queries], batch_size=64,
                                  normalize_embeddings=True, convert_to_numpy=True)
        _, idx = self.index.search(q.astype(np.float32), k)
        return [[self.texts[i] for i in row] for row in idx]


def build(max_chunks=None):
    import faiss
    from datasets import load_dataset
    from sentence_transformers import SentenceTransformer
    ds = load_dataset(C.TEXTBOOKS, split="train")
    texts = list(ds["content"])[:max_chunks]
    emb = SentenceTransformer(EMBEDDER).encode(texts, batch_size=256, normalize_embeddings=True,
                                               convert_to_numpy=True, show_progress_bar=True)
    index = faiss.IndexFlatIP(emb.shape[1])
    index.add(emb.astype(np.float32))
    INDEX_DIR.mkdir(exist_ok=True)
    faiss.write_index(index, str(INDEX_DIR / "index.faiss"))
    (INDEX_DIR / "texts.json").write_text(json.dumps(texts))
    print(f"indexed {len(texts)} passages in {INDEX_DIR}/")


def rag_prompt(passages, question_prompt):
    ctx = "\n\n".join(f"[{i + 1}] {p}" for i, p in enumerate(passages))
    return f"Passages from medical textbooks that may help:\n\n{ctx}\n\n{question_prompt}"


@torch.no_grad()
def predict_letters(model, tok, user_prompts, batch_size=8, max_len=1536):
    """Index (0-3) of the most likely option letter for each prompt."""
    letter_ids = [tok.encode(l, add_special_tokens=False)[0] for l in C.LETTERS]
    tok.padding_side = "left"                   # so the last position is the real last token
    preds = []
    for i in range(0, len(user_prompts), batch_size):
        texts = [tok.apply_chat_template(C.chat(u), tokenize=False, add_generation_prompt=True)
                 for u in user_prompts[i:i + batch_size]]
        enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False,
                  truncation=True, max_length=max_len).to(model.device)
        logits = model(**enc).logits[:, -1, letter_ids]
        preds += logits.argmax(dim=-1).tolist()
    return preds


def load_questions(limit, synthetic):
    if synthetic:
        ex = C.medical_examples(limit or 16, seed=5, synthetic=True)
        return [m[0]["content"] for m in ex], [C.LETTERS.index(m[1]["content"][0]) for m in ex]
    from datasets import load_dataset
    ds = load_dataset(C.MEDQA, split="test")
    if limit:
        ds = ds.select(range(min(limit, len(ds))))
    prompts = [C.mcq_prompt(d["sent1"], [d[f"ending{j}"] for j in range(4)]) for d in ds]
    return prompts, [int(d["label"]) for d in ds]


def evaluate(args):
    prompts, labels = load_questions(args.limit, args.synthetic)
    fake = [f"passage {i} about organs" for i in range(50)] if args.synthetic else None
    passages = Retriever(fake).search(prompts, args.k)
    model, tok = C.load_model(args.model, args.run)
    plain = predict_letters(model, tok, prompts, args.batch_size)
    with_rag = predict_letters(model, tok, [rag_prompt(p, q) for p, q in zip(passages, prompts)],
                               args.batch_size)
    acc = lambda pred: float(np.mean([p == y for p, y in zip(pred, labels)]))
    res = {"name": args.name, "run": args.run, "n": len(labels), "k": args.k,
           "medqa_no_rag": acc(plain), "medqa_rag": acc(with_rag)}
    res["rag_gain"] = res["medqa_rag"] - res["medqa_no_rag"]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{args.name}.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--max_chunks", type=int, default=None)
    e = sub.add_parser("eval")
    e.add_argument("--name", required=True)
    e.add_argument("--run", default=None)
    e.add_argument("--model", default=C.BASE_MODEL)
    e.add_argument("--k", type=int, default=3, help="passages per question")
    e.add_argument("--limit", type=int, default=None)
    e.add_argument("--batch_size", type=int, default=8)
    e.add_argument("--synthetic", action="store_true")
    e.add_argument("--out", default="results/rag")
    args = p.parse_args()
    build(args.max_chunks) if args.cmd == "build" else evaluate(args)


if __name__ == "__main__":
    main()
