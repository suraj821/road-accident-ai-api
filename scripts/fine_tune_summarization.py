from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer
import os

DATA_PATH = 'data/sample_reports.csv'
MODEL_NAME = 'sshleifer/distilbart-cnn-12-6'
OUTPUT_DIR = 'models/summarizer-finetuned'

os.makedirs(OUTPUT_DIR, exist_ok=True)

raw = load_dataset('csv', data_files=DATA_PATH)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

max_input_length = 512
max_target_length = 128

def preprocess(examples):
    inputs = examples['report_text']
    targets = examples['summary']
    model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True)
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, max_length=max_target_length, truncation=True)
    model_inputs['labels'] = labels['input_ids']
    return model_inputs

train_dataset = raw['train'].map(preprocess, batched=True)

args = Seq2SeqTrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=2,
    num_train_epochs=3,
    predict_with_generate=True,
    logging_steps=10,
    save_total_limit=2,
)

trainer = Seq2SeqTrainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    tokenizer=tokenizer,
)

trainer.train()
trainer.save_model(OUTPUT_DIR)
print('Fine-tune complete. Model saved to', OUTPUT_DIR)
