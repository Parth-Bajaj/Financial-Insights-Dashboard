# finbert_finetune.py
import pandas as pd
import re
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset

# --- Optional: Install required packages if needed ---
# !pip install transformers datasets

# Define a basic cleaning function (if needed)
def clean_text(text):
    text = re.sub(r'\n+', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Load pre-trained FinBERT model and tokenizer
model_name = "yiyanghkust/finbert-tone"  # Pre-trained FinBERT for financial sentiment
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name)

# Prepare a sample training dataset (dummy examples)
data = [
    {"input": "What is the revenue for Tata Steel?", "output": "Revenue: 24335.30 Billion ₹"},
    {"input": "How is Infosys performing in terms of net profit?", "output": "Net Profit: 24108.00 Billion ₹"},
    {"input": "Tell me Reliance's EBITDA.", "output": "EBITDA: 176988.00 Billion ₹"}
]
df_train = pd.DataFrame(data)

# For simplicity, assign a dummy label (e.g., 0) to every example.
df_train["label"] = 0

# Convert DataFrame to Hugging Face Dataset
dataset = Dataset.from_pandas(df_train)

# Tokenization function: tokenize the 'input' text.
def tokenize_function(example):
    return tokenizer(example["input"], truncation=True, padding="max_length", max_length=128)

tokenized_dataset = dataset.map(tokenize_function, batched=True)
# Remove extra columns to avoid errors in the Trainer

tokenized_dataset = tokenized_dataset.remove_columns(["input", "output"])


# Set up training arguments (evaluation disabled to avoid eval dataset error)
training_args = TrainingArguments(
    output_dir="finbert_finetuned",
    num_train_epochs=1,  # Use 1 epoch for demo purposes; increase for real training
    per_device_train_batch_size=2,
    evaluation_strategy="no",
    logging_steps=1,
    logging_dir='logs',
    report_to="none"  # Disable wandb logging
)

# Define the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset
)

# Fine-tune the model
trainer.train()

# Save the fine-tuned model and tokenizer locally
model.save_pretrained("finbert_finetuned_model")
tokenizer.save_pretrained("finbert_finetuned_model")

print("Finetuning complete. Model saved in folder 'finbert_finetuned_model'.")
