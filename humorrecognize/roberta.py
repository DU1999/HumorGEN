import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, RobertaForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
import numpy as np
# from datasets import load_metric
import evaluate


pretrained_model_name = "hfl_chinese-roberta-wwm-ext-large"

# 1. load data
df = pd.read_csv("chinese_humor.csv")  
df = df[['joke', 'label']]

# 2. splid dataset
train_texts, val_texts, train_labels, val_labels = train_test_split(
    df['joke'].tolist(), df['label'].tolist(), test_size=0.2, random_state=42
)

# 3. Tokenizer
tokenizer = BertTokenizer.from_pretrained(pretrained_model_name)

train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=128)
val_encodings = tokenizer(val_texts, truncation=True, padding=True, max_length=128)

# 4. define Dataset
class HumorDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(int(self.labels[idx]))
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = HumorDataset(train_encodings, train_labels)
val_dataset = HumorDataset(val_encodings, val_labels)

# 5. load model
model = RobertaForSequenceClassification.from_pretrained(pretrained_model_name, num_labels=2)

# 6. evaluate
# accuracy = load_metric("accuracy")
accuracy = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return accuracy.compute(predictions=predictions, references=labels)

# 7. args
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=4,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True
)

# 8. training
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics
)

trainer.train()

# 9. save model
trainer.save_model("./humor-roberta")
tokenizer.save_pretrained("./humor-roberta")
