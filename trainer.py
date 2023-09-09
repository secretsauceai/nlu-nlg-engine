import toml
import re
import numpy as np
import pandas as pd

from datasets import Dataset
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, Trainer, TrainingArguments

# Load the toml file
config = toml.load("config/config.toml")

# Load the model and tokenizer
model_id = config["model_id"]
model = AutoModelForSeq2SeqLM.from_pretrained(model_id, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_id)


# Load the processed dataset
df = pd.read_csv(config["data_file"], sep=',')
dataset = Dataset.from_pandas(df)


def get_model_output(prompt):
    inputs = tokenizer.encode(prompt, return_tensors='pt')
    #inputs.to('cuda')
    print(f"Number of tokens in prompt: {len(inputs[0])}")
    outputs = model.generate(inputs, max_length=768)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def tokenize_batch(batch):
    encoded_inputs = tokenizer(batch['entity_type_prompt'], padding='max_length', truncation=False, max_length=512, return_attention_mask=True)
    encoded_labels = tokenizer(batch['annotated_utterance'], padding='max_length', truncation=True, max_length=32)

    batch['input_ids'] = encoded_inputs['input_ids']
    batch['attention_mask'] = encoded_inputs['attention_mask']
    batch['labels'] = encoded_labels['input_ids']

    return batch

tokenized_dataset = dataset.map(tokenize_batch, batched=True)
essential_columns = ['input_ids', 'attention_mask', 'labels']
tokenized_dataset = tokenized_dataset.remove_columns([col for col in tokenized_dataset.column_names if col not in essential_columns])

# Split your tokenized dataset into training and validation sets
train_dataset = tokenized_dataset.train_test_split(test_size=config['test_split'])['train']
val_dataset = tokenized_dataset.train_test_split(test_size=config['test_split'])['test']

args = TrainingArguments(
    output_dir=config["output_directory"],
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
    logits, labels = eval_pred.predictions, eval_pred.label_ids

    # Debugging
    #TODO: remove the print statements, do I need to keep logits = logits[0]?
    print("Logits type:", type(logits))
    if isinstance(logits, tuple):
        print("Logits tuple length:", len(logits))
        for idx, item in enumerate(logits):
            print(f"Item {idx} type: {type(item)}, shape: {item.shape}")
        logits = logits[0]  # Assuming the first item in the tuple is the actual logits

    print("Raw logits shape:", logits.shape)
    print("Raw labels shape:", labels.shape)

    logits_np = logits.cpu().numpy() if hasattr(logits, 'is_cuda') and logits.is_cuda else logits
    labels_np = labels.cpu().numpy() if hasattr(labels, 'is_cuda') and labels.is_cuda else labels

    print("Logits shape after conversion:", logits_np.shape)
    print("Labels shape after conversion:", labels_np.shape)

    pred_ids = np.argmax(logits_np, axis=-1)
    match = (pred_ids == labels_np).all(axis=1).mean()

    return {"exact_match": match}

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
    tokenizer=tokenizer,
)

trainer.train()

model.save_pretrained(config["output_directory"])
tokenizer.save_pretrained(config["output_directory"])