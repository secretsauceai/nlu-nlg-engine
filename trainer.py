import toml
import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from transformers import (AutoModelForSeq2SeqLM, AutoTokenizer, Trainer,
                          TrainingArguments, EarlyStoppingCallback, DataCollatorWithPadding)

# Load the toml file
config = toml.load("config/training_config.toml")
bf16 = config.get("bf16", True)
shuffle_dataset = config.get("shuffle_dataset", True)
lora_enabled = config.get("lora_enabled", False)

# Load the model and tokenizer
model_id = config["model_id"]
tokenizer = AutoTokenizer.from_pretrained(model_id)

if lora_enabled:
    print("LoRA is enabled")
    from peft import LoraConfig, get_peft_model, TaskType  # Import PEFT only if LoRA is enabled
    # Define LoRA Config
    lora_config = LoraConfig(
        r=config.get("lora_r", 16),
        lora_alpha=config.get("lora_alpha", 32),
        target_modules=config.get("target_modules", ["q", "v"]),
        lora_dropout=config.get("lora_dropout", 0.05),
        bias=config.get("lora_bias", "none"),
        task_type=TaskType.SEQ_2_SEQ_LM
    )
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id, device_map='auto', torch_dtype=torch.bfloat16 if bf16 else torch.float32)
    model = get_peft_model(model, lora_config)
else:
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id, device_map='auto', torch_dtype=torch.bfloat16 if bf16 else torch.float32)

# Ensure that the rest of your script, including dataset preparation, tokenization, training setup, etc.,
# follows here, adapting as necessary to fit your overall training procedure.



# Load the processed dataset
df = pd.read_csv(config["data_file"], sep=',')
dataset = Dataset.from_pandas(df)

if shuffle_dataset:
    dataset = dataset.shuffle(seed=42)

def tokenize_batch(batch):
    # Tokenize the prompts and answers
    # Padding to the max length might be necessary depending on your model's requirements.
    # Adjust max_length as per your model's specification or dataset's length distribution.
    tokenized_inputs = tokenizer(batch["prompt"], padding='max_length', truncation=True,  max_length=512)
    tokenized_labels = tokenizer(batch["answer"], padding='max_length', truncation=True, max_length=64)

    # Flan-T5 and other seq2seq models use decoder_input_ids for labels during training
    batch["input_ids"] = tokenized_inputs.input_ids
    batch["attention_mask"] = tokenized_inputs.attention_mask
    batch["labels"] = tokenized_labels.input_ids

    return batch

tokenized_dataset = dataset.map(tokenize_batch, batched=True)
essential_columns = ["input_ids", "attention_mask", "labels", "task_type"]
tokenized_dataset = tokenized_dataset.remove_columns(
    [col for col in tokenized_dataset.column_names if col not in essential_columns])

# Split your tokenized dataset into training and validation sets
split_dataset = tokenized_dataset.train_test_split(test_size=config["test_split"])
train_dataset = split_dataset["train"]
val_dataset = split_dataset["test"]

args = TrainingArguments(
    output_dir=config["model_output_directory"],
    evaluation_strategy="epoch",
    learning_rate=config["learning_rate"],
    per_device_train_batch_size=config["per_device_train_batch_size"],
    per_device_eval_batch_size=config["per_device_eval_batch_size"],
    num_train_epochs=config["num_train_epochs"],
    weight_decay=0.01,
    logging_strategy="steps",
    logging_steps=10,
    load_best_model_at_end=True,  # Load best model at end of training
    metric_for_best_model="eval_loss",  # Change this based on what metric you're optimizing for
    greater_is_better=False,  # Change based on metric (False for loss, True for accuracy)
    save_strategy="epoch",
    gradient_accumulation_steps=config["gradient_accumulation_steps"],
    bf16=bf16
)

def compute_metrics(eval_pred):
    logits, labels = eval_pred.predictions, eval_pred.label_ids

    # If logits is a tuple, we're assuming the actual logits are the first element
    if isinstance(logits, tuple):
        logits = logits[0]

    logits_np = logits.detach().cpu().numpy() if hasattr(logits, 'detach') else logits
    labels_np = labels.detach().cpu().numpy() if hasattr(labels, 'detach') else labels
    pred_ids = np.argmax(logits_np, axis=-1)

    task_types = trainer.eval_dataset['task_type']

    scores = {'intent': {'exact_match': 0}, 'entity': {'exact_match': 0}}
    count = {'intent': 0, 'entity': 0}

    for pred, label, task_type in zip(pred_ids, labels_np, task_types):
        # Skip comparison for padding tokens (-100 is often used for padding in Hugging Face)
        valid_indices = label != -100
        pred = pred[valid_indices]
        label = label[valid_indices]

        # Calculate exact match for this example
        exact_match = np.all(pred == label)

        # Increment counts and add to scores based on task type
        scores[task_type]['exact_match'] += exact_match
        count[task_type] += 1

    avg_scores = {}
    for task, task_count in count.items():
        avg_score_key = f"{task}_exact_match"
        if task_count > 0:
            avg_scores[avg_score_key] = scores[task]['exact_match'] / task_count
        else:
            avg_scores[avg_score_key] = 0

    return avg_scores


trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
    tokenizer=tokenizer
    )
# data_collator=data_collator

# Include the EarlyStoppingCallback with the desired patience
trainer.add_callback(
    EarlyStoppingCallback(early_stopping_patience=config["early_stopping_patience"]))

trainer.train()

model.save_pretrained(config["model_output_directory"])
tokenizer.save_pretrained(config["model_output_directory"])