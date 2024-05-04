from modules.data_preprocessing import DataPreprocessor


if __name__ == "__main__":
    # Define the path to your configuration file
    CONFIG_PATH = "config/data_processing_config.toml"

    # Initialize the data preprocessor
    preprocessor = DataPreprocessor(CONFIG_PATH)

    # Clean and structure the data
    cleaned_data = preprocessor.clean_and_structure_data()

    # Prepare the data for tasks based on configuration
    task_data = preprocessor.prepare_task_data()

    # Get the full path for the processed data file from the config
    processed_data_file_path = preprocessor.config["processed_data_file"]

    # Export the task data to a CSV file
    preprocessor.export_to_csv(task_data, processed_data_file_path)
