import toml
import numpy as np
import pandas as pd
from datasets import Dataset
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, Trainer, TrainingArguments

# Load the toml file
config = toml.load("nlu-engine/config/config.toml")

# Load the model and tokenizer
model_id = config["model_id"]
model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Load the processed dataset
df = pd.read_csv(config["data_file"], sep=',')
dataset = Dataset.from_pandas(df)

def tokenize_batch(batch):
    # Tokenize the prompts and answers
    # Padding to the max length might be necessary depending on your model's requirements.
    # Adjust max_length as per your model's specification or dataset's length distribution.
    tokenized_inputs = tokenizer(batch["prompt"], padding="max_length", truncation=True, max_length=512)
    tokenized_labels = tokenizer(batch["answer"], padding="max_length", truncation=True, max_length=64)

    # Flan-T5 and other seq2seq models use decoder_input_ids for labels during training
    batch["input_ids"] = tokenized_inputs.input_ids
    batch["attention_mask"] = tokenized_inputs.attention_mask
    batch["labels"] = tokenized_labels.input_ids

    return batch

tokenized_dataset = dataset.map(tokenize_batch, batched=True)
essential_columns = ["input_ids", "attention_mask", "labels", "task_type"]
tokenized_dataset = tokenized_dataset.remove_columns([col for col in tokenized_dataset.column_names if col not in essential_columns])

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
    logging_steps=10  # Log every 10 steps. Adjust this based on your preference.
)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    
    # Access the evaluation dataset from the global scope or context
    # This requires that the eval_dataset have a 'task_type' column
    eval_dataset = trainer.eval_dataset  # Assuming trainer object is accessible
    task_types = eval_dataset['task_type']

    # Flatten predictions and labels for computing metrics
    pred_flat = predictions.flatten()
    labels_flat = labels.flatten()
    task_types_flat = task_types.flatten()  # Ensure task_types are flattened and aligned with predictions/labels
    
    # Initialize score dictionaries
    scores = {'intent': {'exact_match': 0}, 'entity': {'exact_match': 0}}
    count = {'intent': 0, 'entity': 0}
    
    # Iterate over each prediction and corresponding label
    for idx, (pred, label) in enumerate(zip(pred_flat, labels_flat)):
        if label == -100:  # Ignoring the padding tokens
            continue
        
        task_type = task_types_flat[idx]  # Determine task type for this prediction
        
        # Calculate exact match for the relevant task
        exact_match = int(pred == label)
        scores[task_type]['exact_match'] += exact_match
        count[task_type] += 1
    
    # Calculate average scores
    for task, score in scores.items():
        if count[task] > 0:
            scores[task]['exact_match'] /= count[task]


    return scores


trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
    tokenizer=tokenizer,
)

trainer.train()

model.save_pretrained(config["model_output_directory"])
tokenizer.save_pretrained(config["model_output_directory"])