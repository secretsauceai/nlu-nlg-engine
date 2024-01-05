import os
import toml
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import pandas as pd

class ModelInference:
    def __init__(self, model_dir: str, tokenizer_dir: str, device: str = 'cuda'):
        """
        Initialize the model inference with model and tokenizer.

        Parameters:
        model_dir (str): Directory where the trained model is saved.
        tokenizer_dir (str): Directory where the tokenizer is saved.
        device (str): Device to run the model on, 'cuda' or 'cpu'.
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_dir).to(self.device)
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
    data['predictions'] = [inferencer.predict(row['prompt']) for _, row in data.iterrows()]
    return data

if __name__ == "__main__":
    # Load configuration
    config = toml.load("config/config.toml")

    # Initialize the model inference
    inferencer = ModelInference(model_dir=config["model_output_directory"],
                                tokenizer_dir=config["model_output_directory"])

    # Load and prepare data for inference
    prepared_data = load_and_prepare_data(config["processed_data_file"])

    # Run inference
    prepared_data = run_inference(prepared_data, inferencer)

    # Define the output path for the predictions CSV from the config file
    output_csv_path = config["predictions_output_file"]

    # Ensure the directory exists
    output_dir = os.path.dirname(output_csv_path)
    os.makedirs(output_dir, exist_ok=True)

    # Export the updated DataFrame with predictions to a CSV file
    prepared_data.to_csv(output_csv_path, index=False)
    print(f"Predictions exported successfully to {output_csv_path}")