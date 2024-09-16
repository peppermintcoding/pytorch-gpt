import argparse
import torch

import tiktoken
from model import GPT, GPTConfig


def setup(model_path: str):
    tokenizer = tiktoken.get_encoding("gpt2")
    checkpoint = torch.load(model_path, weights_only=True)
    model = GPT(GPTConfig(**checkpoint["model_args"]))

    # rename keys because of torch >=2.1
    state_dict = {}
    for key, val in checkpoint["model"].items():
        if key.startswith("_orig_mod"):
            state_dict[key[10:]] = val
        else:
            state_dict[key] = val
    model.load_state_dict(state_dict)
    model.to("cuda")
    model.eval()
    return model, tokenizer

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
line_divider = (
    ".~^~._.~^~._.~^~._.~^~._.~^~._.~^~._.~^~._.~^~._.~^~._.~^~._.~^~._.~^~._.~^~._-~"
)
thin_line_divider = (
    "- -- - -- - -- - -- - -- - -- - -- - -- - -- - -- - -- - -- - -- - --- -- - -- -"
)
header = rf"""
{line_divider}
 .~----~.                                                              .~----~.
(  ~^^~  )             ______       __ __ _                           (  ~^^~  ) 
 )      (             / ____/____  / // /(_)___  ____  ___             )      (
(  (())  )           / /    / __ \/ // // / __ \/ __ \/ _ \           (  (())  ) 
 |  ||  |            | |___/ /_/ / // // / /_/ / /_/ / ___/            |  ||  |
 |  ||  |            \____/\__._/_//_//_/\____/ .___/\___/             |  ||  |
 |  ||  |                                    /_/                       |  ||  |
 |  ||  |                   -- your personal muse --                   |  ||  |
(________)                                                            (________)
{line_divider}
"""


def get_multiline_input(start: str):
    lines = []
    while True:
        line = input(start)
        if line == "":
            break
        lines.append(line)
    return "\n".join(lines)


def generate(model, tokenizer, prompt: str, temp: float, maxlen: int) -> str:
    idx = model.generate(
        torch.tensor(
            [tokenizer.encode(prompt)], device=DEVICE
        ),
        max_new_tokens=maxlen,
        temperature=temp,
    )
    return tokenizer.decode(idx[0].cpu().numpy())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate with Calliope",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model",
        type=str,
        default="out/ckpt.pt",
        help="Model Checkpoint to load",
    )
    parser.add_argument(
        "--maxlen", type=int, default=128, help="Max number of tokens to generate"
    )
    parser.add_argument(
        "--temp", type=float, default=0.9, help="Temperature in the softmax"
    )
    args = parser.parse_args()

    print(header)
    print(f"loading {args.model} on {DEVICE}..")
    model, tokenizer = setup(model_path=args.model)
    print(f"loaded with temp: {args.temp}, maxlen: {args.maxlen}")
    print(thin_line_divider)

    print("prompt calliope, start a poem's song")
    print("an empty line tells calliope to follow along")
    print("write 'exit' to leave, if you won't stay long")

    prompt = get_multiline_input(start=">> ")
    while prompt != "exit":
        if len(prompt) != 0:
            print(line_divider)
            print(generate(model, tokenizer, prompt, args.temp, args.maxlen))
            print(line_divider)
        prompt = get_multiline_input(start=">> ")

    print("bye wanderer, may your path be ever light")
