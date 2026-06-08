"""Fine-tune BERTimbau for Brazilian Portuguese tasks."""
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset
import torch

def fine_tune_sentiment(base_model: str = "neuralmind/bert-base-portuguese-cased",
                        output_dir: str = "./bertimbau-sentiment-br",
                        epochs: int = 5):
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForSequenceClassification.from_pretrained(base_model, num_labels=3)
    dataset = load_dataset("json", data_files={"train": "data/train.jsonl", "test": "data/test.jsonl"})
    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, max_length=512, padding="max_length")
    tokenized = dataset.map(tokenize, batched=True)
    args = TrainingArguments(output_dir=output_dir, num_train_epochs=epochs,
        per_device_train_batch_size=32, evaluation_strategy="epoch",
        learning_rate=2e-5, weight_decay=0.01, load_best_model_at_end=True)
    trainer = Trainer(model=model, args=args,
        train_dataset=tokenized["train"], eval_dataset=tokenized["test"])
    trainer.train()
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Model saved to {output_dir}")
