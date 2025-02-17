from transformers import MarianMTModel, MarianTokenizer

src_text = [
    ">>vie<< this is a sentence in English that we want to translate to French",
    ">>vie<< Sick",
    ">>vie<< long",
]

# Specify the multilingual model
model_name = "../models"
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)

# Tokenize the source text
inputs = tokenizer(src_text, return_tensors="pt", padding=True)

# Generate the translations
translated = model.generate(**inputs)

# Decode the translated text
tgt_text = [tokenizer.decode(t, skip_special_tokens=True) for t in translated]
print(tgt_text)