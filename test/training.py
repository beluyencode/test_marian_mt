from transformers import MarianMTModel, MarianTokenizer, Seq2SeqTrainer, Seq2SeqTrainingArguments, DataCollatorForSeq2Seq
from datasets import Dataset
import logging
import os

# Thiết lập logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Tạo tập dữ liệu từ các tệp văn bản
def load_data(source_file, target_file):
    logger.info(f"Loading data from {source_file} and {target_file}")
    with open(source_file, 'r', encoding='utf-8') as src_f, open(target_file, 'r', encoding='utf-8') as tgt_f:
        src_lines = src_f.readlines()
        tgt_lines = tgt_f.readlines()
    logger.info(f"Loaded {len(src_lines)} source lines and {len(tgt_lines)} target lines")
    return Dataset.from_dict({"translation": [{"en": src.strip(), "vi": tgt.strip()} for src, tgt in zip(src_lines, tgt_lines)]})

source_file = 'train.source'
target_file = 'train.target'
eval_source_file = 'dev.source'
eval_target_file = 'dev.target'

if not os.path.exists(source_file) or not os.path.exists(target_file):
    raise FileNotFoundError(f"One or both of the files {source_file} and {target_file} do not exist.")

train_dataset = load_data(source_file, target_file)
eval_dataset = load_data(eval_source_file, eval_target_file)

# Tạo tokenizer và mô hình
model_name = "Helsinki-NLP/opus-mt-en-vi"
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)

# Tokenize dữ liệu
def preprocess_function(examples):
    inputs = [ex["en"] for ex in examples["translation"]]
    targets = [ex["vi"] for ex in examples["translation"]]
    model_inputs = tokenizer(inputs, max_length=128, truncation=True, padding="max_length")
    labels = tokenizer(text_target=targets, max_length=128, truncation=True, padding="max_length")
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_datasets = train_dataset.map(preprocess_function, batched=True)
tokenized_eval_dataset = eval_dataset.map(preprocess_function, batched=True)

# Thiết lập các tham số huấn luyện
training_args = Seq2SeqTrainingArguments(
    output_dir="./results",   # Thư mục lưu checkpoint & log
    evaluation_strategy="epoch",  # Đánh giá mô hình sau mỗi epoch
    save_strategy="epoch",  # Lưu checkpoint sau mỗi epoch
    per_device_train_batch_size=8,  # Batch size khi huấn luyện
    per_device_eval_batch_size=8,  # Batch size khi đánh giá
    learning_rate=1e-5,  # Tốc độ học
    num_train_epochs=10,  # Số epochs
    weight_decay=0.01,  # Giảm trọng số (regularization)
    predict_with_generate=True,  # Dùng `generate()` để đánh giá
    logging_dir="./logs",  # Lưu log TensorBoard
    logging_steps=100,  # Ghi log sau mỗi 100 bước
)

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

# Tạo Trainer
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets,
    eval_dataset=tokenized_eval_dataset,
    data_collator=data_collator,  # Sử dụng data_collator thay vì tokenizer
)

# Huấn luyện mô hình
logger.info("Starting training...")
train_result = trainer.train()
logger.info("Training completed.")

# Lưu mô hình và tokenizer
model.save_pretrained("./results")
tokenizer.save_pretrained("./results")
logger.info("Model and tokenizer saved.")

# In kết quả huấn luyện
print(train_result)