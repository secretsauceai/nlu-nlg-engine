import os
import toml
from modules.inference import ModelInference

if __name__ == "__main__":
    # Load configuration
    config = toml.load("config/test_inference_config.toml")

    # Initialize the model inference
    inferencer = ModelInference(config=config)

    # Load and prepare data for inference
    prepared_data = inferencer.load_and_prepare_data(config["processed_data_file"])

    # Run inference
    prepared_data = inferencer.run_inference(prepared_data, inferencer)

    # Define the output path for the predictions CSV from the config file
    output_csv_path = config["predictions_output_file"]

    # Ensure the directory exists
    output_dir = os.path.dirname(output_csv_path)
    os.makedirs(output_dir, exist_ok=True)

    # Export the updated DataFrame with predictions to a CSV file
    prepared_data.to_csv(output_csv_path, index=False)
    print(f"Predictions exported successfully to {output_csv_path}")