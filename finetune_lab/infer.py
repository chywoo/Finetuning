"""Generate one answer using the lesson's training prompt format."""
import argparse

from finetune_lab.evaluate import load_text_model, resolve_device
from finetune_lab.text_encoding import format_prompt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", default="main")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="cuda")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    args = parser.parse_args()
    if not args.prompt.strip() or args.max_new_tokens < 1:
        parser.error("prompt must be non-empty and max-new-tokens positive")
    import torch
    device = resolve_device(args.device)
    model, tokenizer, _ = load_text_model(args.model, args.revision, device)
    inputs = tokenizer(format_prompt(args.prompt), add_special_tokens=False, return_tensors="pt").to(device)
    limit = getattr(model.config, "max_position_embeddings", None) or getattr(model.config, "n_positions", None)
    if limit and inputs["input_ids"].shape[1] + args.max_new_tokens > limit:
        parser.error("prompt + new tokens exceeds model context; shorten prompt or generation")
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False,
                                 pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
    print(tokenizer.decode(outputs[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True))


if __name__ == "__main__":
    main()
