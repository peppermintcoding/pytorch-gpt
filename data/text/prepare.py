import argparse
import os
import time

import numpy as np
import tiktoken
from tqdm import tqdm


def tokenize_txt(filedir: str, out_filename: str):
    files = os.listdir(filedir)
    tokenizer = tiktoken.get_encoding("gpt2")
    input_ids = []
    for file in tqdm(files):
        with open(f"{filedir}/{file}", "r") as f:
            data = f.read()
        input_ids.extend(tokenizer.encode(data, allowed_special={"<|endoftext|>"}))
    input_ids = np.array(input_ids).astype(np.uint16)
    np.save(out_filename, input_ids)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Tokenize txt file with tiktoken",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--txt",
        type=str,
        default="data/text/files",
        help="Folder with txt files",
    )
    parser.add_argument(
        "--out", type=str, default="train", help="File to write tokens to"
    )
    args = parser.parse_args()

    t = time.time()
    print(f"Starting to tokenize {args.txt} using gpt2 tokenizer..")
    tokenize_txt(args.txt, args.out)
    train_data = np.load(f"{args.out}.npy")
    print(f"Number of trainings token: {len(train_data):,}")
    print(f"took {time.time()-t:.2f}s")
