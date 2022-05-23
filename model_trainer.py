from tokenizers.implementations import ByteLevelBPETokenizer
from tokenizers.processors import BertProcessing

from transformers import RobertaConfig
from transformers import RobertaTokenizerFast

from transformers import LineByLineTextDataset
from transformers import DataCollatorForLanguageModeling

from transformers import Trainer, TrainingArguments
from transformers import RobertaForMaskedLM
from transformers import pipeline

import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('-i', '--datafolder', type=str, default="data", help='input data path')
    parser.add_argument('-o', '--outpath', type=str, default="deeppotter", help='model-tokenizer output path')
    parser.add_argument('-p', '--pretrained_path', type=str, default="deeppotter", help='pretrained path')
    parser.add_argument('-e', '--epoch', type=int, default=1, help='epoch')
    parser.add_argument('-b', '--batch_size', type=int, default=64, help='batch size')

    args = vars(parser.parse_args())

    tokenizer = ByteLevelBPETokenizer(
        f"./{args['pretrained_path']}/vocab.json",
        f"./{args['pretrained_path']}/merges.txt",
    )

    tokenizer._tokenizer.post_processor = BertProcessing(
        ("</s>", tokenizer.token_to_id("</s>")),
        ("<s>", tokenizer.token_to_id("<s>")),
    )

    tokenizer.enable_truncation(max_length=512)

    config = RobertaConfig(
        vocab_size=52_000,
        max_position_embeddings=514,
        num_attention_heads=12,
        num_hidden_layers=6,
        type_vocab_size=1,
    )

    tokenizer = RobertaTokenizerFast.from_pretrained("./deeppotter", max_len=512)



    model = RobertaForMaskedLM(config=config)
    print(f"Model created with : # {model.num_parameters()} parameters")

    # load dataset
    dataset = LineByLineTextDataset(
        tokenizer=tokenizer,
        file_path=f"./{args['datafolder']}/book.txt",
        block_size=128,
    )

    # data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=0.15
    )

    # training config
    training_args = TrainingArguments(
        output_dir=f"./{args['outpath']}",
        overwrite_output_dir=True,
        num_train_epochs={args['epoch']},
        per_gpu_train_batch_size={args['batch_size']},
        save_steps=10_000,
        save_total_limit=2,
        prediction_loss_only=True,
    )

    # trainer instance
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=dataset,
    )

    # start training
    trainer.train()

    # save model
    trainer.save_model(f"./{args['outpath']}")

    # load model and inference
    fill_mask = pipeline(
        "fill-mask",
        model=f"./{args['pretrained_path']}",
        tokenizer=f"./{args['pretrained_path']}"
    )

    result = fill_mask("wingardium <mask>.")
    print("Answer is : ", result)
