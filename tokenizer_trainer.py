from pathlib import Path
from tokenizers import ByteLevelBPETokenizer

import argparse


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('-i', '--datafolder', type=str, default="data", help='input data path')
    parser.add_argument('-o', '--outpath', type=str, default="deeppotter", help='model-tokenizer output path')
    parser.add_argument('-f', '--min_frequency', type=int, default=2, help='minimum frequency of tokens')
    parser.add_argument('-v', '--vocab_size', type=int, default=52000, help='vocabulary size')

    args = vars(parser.parse_args())

    paths = [str(x) for x in Path(f"./{args['datafolder']}").glob("**/*.txt")]

    # Initialize a tokenizer
    tokenizer = ByteLevelBPETokenizer()

    # Customize training
    tokenizer.train(files=paths, vocab_size=args['vocab_size'], min_frequency=args['min_frequency'], special_tokens=[
        "<s>",
        "<pad>",
        "</s>",
        "<unk>",
        "<mask>",
    ])

    tokenizer.save_model(f"./{args['outpath']}")

    print(f"Tokenizer saved with # {tokenizer.get_vocab_size()} tokens and saved to {args['datafolder']}")
