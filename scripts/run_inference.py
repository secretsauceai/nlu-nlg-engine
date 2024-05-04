import os
import pandas as pd
import toml
from modules.data_preprocessing import DataPreprocessor
from modules.inference import ModelInference, run_inference

SAVE_INTERVAL = 1000  # Adjust the interval to your needs

def save_checkpoint(df, output_path, stage):
    stage_output_path = f"{os.path.splitext(output_path)[0]}_{stage}.csv"
    df.to_csv(stage_output_path, index=False)
    print(f"Checkpoint saved to {stage_output_path}")

def load_checkpoint(output_path):
    if os.path.exists(output_path):
        print(f"Loading checkpoint from {output_path}")
        return pd.read_csv(output_path)
    else:
        return None

def main(inference_config_path):
    # Load the inference configuration
    inference_config = toml.load(inference_config_path)
    output_path = inference_config["predictions_output_file"]
    print(f"Loaded inference configuration from {inference_config_path}")

    # Load or initialize the DataFrame
    df_cleaned = load_checkpoint(output_path)
    if df_cleaned is None:
        print(f"Loading data from {inference_config['data_file']}")
        preprocessor = DataPreprocessor(inference_config_path)
        df_cleaned = preprocessor.clean_and_structure_data()
        print("Data loaded successfully")
        
    inferencer = ModelInference(config=inference_config)
    
    # Step 1: Generate intent prompts if not done yet
    if 'intent_prompt' not in df_cleaned.columns:
        print("Generating intent prompts")
        df_cleaned['intent_prompt'] = df_cleaned.apply(preprocessor.get_domain_intent_prompt, axis=1)
        save_checkpoint(df_cleaned, output_path, 'intent_prompts')
        print("Intent prompts generated and saved successfully")

    # Step 2: Run intent inference if not done yet
    if 'predicted_intent' not in df_cleaned.columns:
        print("Running intent inference")
        intent_predictions = run_inference(df_cleaned[['intent_prompt']].rename(columns={'intent_prompt': 'prompt'}), inferencer)
        print("Intent inference completed successfully")
        df_cleaned['predicted_intent'] = intent_predictions['predictions']
        save_checkpoint(df_cleaned, output_path, 'predicted_intents')
    
    # Step 3: Generate entity prompts using predicted intents if not done yet
    def prepare_entity_prompt(row):
        # Update intent with predicted intent
        row['intent'] = row['predicted_intent']

        # Find the predicted domain based on the predicted intent
        matching_rows = df_cleaned[df_cleaned['intent'] == row['predicted_intent']]
        predicted_domain = matching_rows['domain'].iloc[0] if not matching_rows.empty else row['domain']
        row['domain'] = predicted_domain

        return preprocessor.get_entity_type_prompt(row, preprocessor.get_intents_and_entities())

    if 'entity_prompt' not in df_cleaned.columns:
        # Select the appropriate entity template
        if inference_config.get('use_entity_bracket', True):
            preprocessor.entity_template = inference_config["prompt_templates"]["entity_bracket_template"]
        else:
            preprocessor.entity_template = inference_config["prompt_templates"]["entity_slot_template"]

        print("Generating entity prompts")
        df_cleaned['entity_prompt'] = df_cleaned.apply(prepare_entity_prompt, axis=1)
        save_checkpoint(df_cleaned, output_path, 'entity_prompts')

    # Step 4: Run entity inference if not done yet
    if 'prediction_annotation' not in df_cleaned.columns:
        print("Running entity inference")
        entity_predictions = run_inference(df_cleaned[['entity_prompt']].rename(columns={'entity_prompt': 'prompt'}), inferencer)
        print("Entity inference completed successfully")
        df_cleaned['prediction_annotation'] = entity_predictions['predictions']
        save_checkpoint(df_cleaned, output_path, 'predicted_annotations')

    # Export the final result to a CSV file
    df_cleaned.to_csv(output_path, index=False)
    print(f"Final data exported successfully to {output_path}")

if __name__ == "__main__":
    main("config/inference_config.toml")