import torch
from transformers import Trainer, TrainingArguments
from models.ie_higraph_led import IEHiGraphLED
from utils.data_loader import load_dataset

def main():
    model = IEHiGraphLED()
    dataset = load_dataset()

    training_args = TrainingArguments(
        output_dir="./outputs",
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        evaluation_strategy="steps",
        save_steps=1000,
        logging_steps=100,
        num_train_epochs=3,
        learning_rate=3e-5,
        weight_decay=0.01,
        fp16=True
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
    )

    trainer.train()

if __name__ == "__main__":
    main()
