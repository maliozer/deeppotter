# load model
from transformers import pipeline

fill_mask = pipeline(
    "fill-mask",
    model="./deeppotter",
    tokenizer="./deeppotter"
)

fill_mask("wingardium <mask>.")
