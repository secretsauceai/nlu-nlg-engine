import os
import toml
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import pandas as pd
from tqdm import tqdm

class ModelInference:
    def __init__(self, config, device: str = 'cuda'):
        """
        Initialize the model inference with model and tokenizer.

        Parameters:
        config (dict): Configuration loaded from TOML.
        device (str): Device to run the model on, 'cuda' or 'cpu'.
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        model_dir = config["model_output_directory"]
        tokenizer_dir = config["model_output_directory"]
        lora_enabled = config.get("lora_enabled", False)
        use_bf16 = config.get("use_bf16", False)

        if lora_enabled:
            from peft import PeftModel, PeftConfig
            # Load LoRA model
            peft_model_id = model_dir  # Assuming the directory contains the necessary PeftModel configuration
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_dir, torch_dtype=torch.bfloat16 if use_bf16 else torch.float32, device_map='auto')
            peft_config = PeftConfig.from_pretrained(peft_model_id)
            self.model = PeftModel.from_pretrained(self.model, peft_model_id, device_map='auto')
        else:
            # Load regular model
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_dir, torch_dtype=torch.bfloat16 if use_bf16 else torch.float32, device_map='auto')
        
        self.model.to(self.device).eval()  # Ensure model is in evaluation mode
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)

    def predict(self, text: str) -> str:
        """
        Run model inference on the provided text.

        Parameters:
        text (str): Text to run inference on.

        Returns:
        str: The model's prediction.
        """
        inputs = self.tokenizer.encode(text, return_tensors='pt').to(self.device)
        outputs = self.model.generate(inputs, max_length=512)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

def load_and_prepare_data(file_path: str) -> pd.DataFrame:
    """
    Load and prepare data for inference.

    Parameters:
    file_path (str): File path to the CSV containing the data.

    Returns:
    pd.DataFrame: Prepared DataFrame.
    """
    df = pd.read_csv(file_path)
    # Additional preparation steps can go here if necessary
    return df

def run_inference(data: pd.DataFrame, inferencer: ModelInference):
    """
    Run inference on the provided data using the given model.

    Parameters:
    data (pd.DataFrame): DataFrame containing the data.
    inferencer (ModelInference): The inference object.

    Returns:
    pd.DataFrame: The DataFrame with an added column for predictions.
    """
    tqdm.pandas(desc="Running Inference")
    data['predictions'] = data['prompt'].progress_apply(inferencer.predict)
    return data