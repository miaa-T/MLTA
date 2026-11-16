"""Small chat demo: the same question answered by several models side by side.

  python demo.py lora=runs/lora replay20=runs/replay20

The base model is always shown. Runs must be LoRA adapters.
In Colab, add --share to get a public link."""
import argparse

import torch

import common as C


def main():
    p = argparse.ArgumentParser()
    p.add_argument("runs", nargs="*", help="name=folder pairs")
    p.add_argument("--model", default=C.BASE_MODEL)
    p.add_argument("--share", action="store_true")
    args = p.parse_args()
    runs = dict(r.split("=", 1) for r in args.runs)

    import gradio as gr
    from peft import PeftModel

    model, tok = C.load_model(args.model)
    for i, (name, path) in enumerate(runs.items()):
        if i == 0:
            model = PeftModel.from_pretrained(model, path, adapter_name=name)
        else:
            model.load_adapter(path, adapter_name=name)
    model.eval()

    @torch.no_grad()
    def generate(question):
        text = tok.apply_chat_template(C.chat(question), tokenize=False, add_generation_prompt=True)
        enc = tok(text, return_tensors="pt", add_special_tokens=False).to(model.device)

        def run():
            out = model.generate(**enc, max_new_tokens=300, do_sample=False)
            return tok.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True)

        if not runs:
            return [run()]
        with model.disable_adapter():
            answers = [run()]
        for name in runs:
            model.set_adapter(name)
            answers.append(run())
        return answers

    with gr.Blocks(title="Forgetting demo") as ui:
        gr.Markdown("## Base model vs. fine-tuned models\n"
                    "Try a medical question, then a general one "
                    "(for example: *Write a haiku about autumn, in exactly three lines.*)")
        box = gr.Textbox(label="Question", lines=3)
        btn = gr.Button("Ask")
        with gr.Row():
            outs = [gr.Textbox(label=n, lines=12) for n in ["base", *runs]]
        btn.click(generate, box, outs)
    ui.launch(share=args.share)


if __name__ == "__main__":
    main()
