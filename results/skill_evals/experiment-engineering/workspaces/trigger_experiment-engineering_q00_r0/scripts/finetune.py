from transformers import Trainer, TrainingArguments
from model import load_base, load_dataset

model = load_base("gpt2-medium")
ds = load_dataset("data/items.csv")
args = TrainingArguments(output_dir="ckpts", num_train_epochs=3,
                         per_device_train_batch_size=8)
trainer = Trainer(model=model, args=args, train_dataset=ds)
trainer.train()
