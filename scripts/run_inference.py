import pandas as pd
import toml
from modules.data_preprocessing import DataPreprocessor
from modules.inference import ModelInference, run_inference

def main(inference_config_path):
    # Load the inference configuration
    inference_config = toml.load(inference_config_path)
    
    # Initialize the Data Preprocessor with the same config file
    preprocessor = DataPreprocessor(inference_config_path)
    df_cleaned = preprocessor.clean_and_structure_data()
    
    # Step 1: Generate intent prompts
    df_cleaned['intent_prompt'] = df_cleaned.apply(preprocessor.get_domain_intent_prompt, axis=1)
    
    # Initialize the model inference
    inferencer = ModelInference(config=inference_config)

    # Step 2: Run intent inference
    intent_predictions = run_inference(df_cleaned[['intent_prompt']].rename(columns={'intent_prompt': 'prompt'}), inferencer)
    
    # Add intent predictions to the dataframe
    df_cleaned['predicted_intent'] = intent_predictions['predictions']
    
    # Step 3: Generate entity prompts using predicted intents
    def prepare_entity_prompt(row):
        # Update intent with predicted intent
        row['intent'] = row['predicted_intent']
        return preprocessor.get_entity_type_prompt(row, preprocessor.get_intents_and_entities())
    
    # Select the appropriate entity template
    if inference_config.get('use_entity_bracket', True):
        preprocessor.entity_template = inference_config["prompt_templates"]["entity_bracket_template"]
    else:
        preprocessor.entity_template = inference_config["prompt_templates"]["entity_slot_template"]
    
    df_cleaned['entity_prompt'] = df_cleaned.apply(prepare_entity_prompt, axis=1)

    # Step 4: Run entity inference
    entity_predictions = run_inference(df_cleaned[['entity_prompt']].rename(columns={'entity_prompt': 'prompt'}), inferencer)

    # Add entity predictions to the dataframe
    df_cleaned['prediction_annotation'] = entity_predictions['predictions']

    # Export the result to a CSV file
    output_path = inference_config["predictions_output_file"]
    df_cleaned.to_csv(output_path, index=False)
    print(f"Data exported successfully to {output_path}")

if __name__ == "__main__":
    main("config/inference_config.toml")
