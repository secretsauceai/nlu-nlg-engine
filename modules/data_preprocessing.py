from typing import Dict
import re
import random
import os
import pandas as pd
import toml

class DataPreprocessor:
    """
    DataPreprocessor handles the preparation of data for training and inference tasks. It loads
    data from a specified CSV file and provides functionalities to clean, structure, and generate
    prompts based on domain intent and entity recognition tasks. The class supports exporting
    processed data to a specified format for use in model training or evaluation.

    Attributes:
        config (dict): Configuration settings loaded from a TOML file, including paths and
                       options for data processing.
        df (pd.DataFrame): The loaded and potentially preprocessed dataframe from the specified
                           data source.
        domain_intent_task (bool): Flag indicating if domain-intent task data should be
                                       generated.
        entity_bracket_task (bool): Flag indicating if entity recognition data should be
                                generated for bracket format.
        entity_slot_task (bool): Flag indicating if entity recognition data should be
                                generated for slot format.

    Methods:
        clean_and_structure_data: Cleans and structures the dataframe by selecting necessary
                                  columns and handling renames.
        get_domain_intent_prompt: Generates a domain intent prompt for a given row of data.
        get_entity_types_from_annotation: Extracts entity types from the annotated utterance.
        get_intents_and_entities: Creates a dictionary of intents and their associated entities.
        get_entity_type_prompt: Generates an entity type prompt for a given row of data.
        prepare_task_data: Prepares the data by generating prompts for the selected tasks.
        export_to_csv: A static method that exports the given dataframe to a CSV file in the
                       specified directory.
    """
    def __init__(self, config_path: str):
        """
        Initialize the DataPreprocessor with configuration settings.

        Parameters:
        config_path (str): Path to the configuration file.
        """
        self.config = toml.load(config_path)
        self.df = pd.read_csv(self.config["data_file"], sep=',')
        self.domain_intent_task = self.config.get("domain_intent_task", False)
        self.entity_bracket_task = self.config.get("entity_bracket_task", False)
        self.entity_slot_task = self.config.get("entity_slot_task", False)
        self.domain_intent_template = self.config["prompt_templates"]["domain_intent_template"]
        self.entity_template = None


    def clean_and_structure_data(self) -> pd.DataFrame:
        """
        Clean and structure the dataframe by selecting necessary columns and handling renames.

        Returns:
        pd.DataFrame: The cleaned and structured dataframe.
        """
        if 'annotated_utterance_cleaned_and_removed_incorrect_tags' in self.df.columns:
            self.df = self.df.drop(columns=['annotated_utterance'])
            self.df = self.df.rename(
                columns={'annotated_utterance_cleaned_and_removed_incorrect_tags': 'annotated_utterance'})

        self.df = self.df[['utterance', 'domain', 'intent', 'annotated_utterance']]
        return self.df

    def get_domain_intent_prompt(self, selected_row: pd.Series) -> str:
        """
        Generate a domain intent prompt for a given row of data.

        Parameters:
        selected_row (pd.Series): A row from the dataframe.

        Returns:
        str: A generated domain intent prompt.
        """
        selected_utterance = selected_row['utterance']
        selected_domain = selected_row['domain']

         # count number of unique domains
        number_of_domains = self.df['domain'].nunique()

        # create a random number between 3 and number_of_domains
        # NOTE: This should be changed as needed to get a good range.
        if number_of_domains < 10:
            random_number = number_of_domains
            selected_domains = self.df['domain'].unique().tolist()
        else:
            random_number = random.randint(10, number_of_domains)
            # select a random number of unique domains
            selected_domains = self.df['domain'].sample(n=random_number).unique().tolist()
            # check if selected domain is not in selected domains, if not add it in a random position in the list
            if selected_domain not in selected_domains:
                selected_domains.insert(random.randint(0, len(selected_domains)), selected_domain)

        domain_intents = {}
        for domain in selected_domains:
            domain_intents[domain] = self.df[
                self.df['domain'] == domain]['intent'].unique().tolist()

        domain_intent_prompt = self.domain_intent_template.format(
            selected_utterance=selected_utterance, selected_domain=selected_domain, domain_intents=domain_intents)
        
        return domain_intent_prompt

    def get_entity_types_from_annotation(self, annotated_utterance: str) -> str:
        """
        Extracts entity types from the annotated utterance.

        Parameters:
        annotated_utterance (str): The annotated utterance text.

        Returns:
        str: A string of unique entity types joined by a comma.
        """
        entity_types = [
            match.split(' : ')[0].replace('[', '') for match in re.findall(
                r'\[.*?\]', annotated_utterance)]
        return ', '.join(set(entity_types))

    def get_intents_and_entities(self) -> Dict[str, str]:
        """
        Create a dictionary of intents and their entities from the dataframe.

        Returns:
        Dict[str, str]: A dictionary mapping intents to their entities.
        """
        intents_entities = {}
        for index, row in self.df.iterrows():
            entity_types = self.get_entity_types_from_annotation(row['annotated_utterance'])
            if entity_types != '':
                try:
                    intents_entities[row['intent']] = intents_entities[row['intent']] + ', ' + entity_types
                except KeyError:
                    intents_entities[row['intent']] = entity_types

        # for the above dictionary, remove the duplicates from the values
        for key, value in intents_entities.items():
            intents_entities[key] = ', '.join(set(value.split(', ')))
        return intents_entities
    
    def convert_annotated_utterance_to_slots(self, annotated_utterance):
        """
        This function takes in an annotated utterance and returns a string of the entity type for each word in the utterance. 
        """
        # Regular expression to find bracketed expressions with entity types
        pattern = re.compile(r'\[(.*?) : (.*?)\]')
        
        # Placeholder for the transformed words
        transformed_words = []
        
        # Tracks the last index processed to handle text outside brackets
        last_idx_processed = 0
        
        # Iterate over all matches of bracketed expressions
        for match in pattern.finditer(annotated_utterance):
            # Extract the entity type and entity value(s) from the match
            entity_type, entities = match.groups()
            
            # Split entities on spaces assuming multiple entities can be separated by spaces
            entity_values = entities.split()
            
            # Handle the text before the current bracketed expression (if any)
            pre_text = annotated_utterance[last_idx_processed:match.start()].strip()
            if pre_text:
                transformed_words.extend(['0'] * len(pre_text.split()))
            
            # Replace each entity value with the entity type
            transformed_words.extend([entity_type] * len(entity_values))
            
            # Update the last index processed
            last_idx_processed = match.end()
        
        # Handle any remaining text after the last bracketed expression
        post_text = annotated_utterance[last_idx_processed:].strip()
        if post_text:
            transformed_words.extend(['0'] * len(post_text.split()))
        
        return ' '.join(transformed_words)


    def get_entity_type_prompt(
            self, selected_row: pd.Series, intents_entities: Dict[str, str]) -> str:
        """
        Generate an entity type prompt for a given row of data.

        Parameters:
        selected_row (pd.Series): A row from the dataframe.
        intents_entities (Dict[str, str]): A dictionary of intents and their entities.

        Returns:
        str: A generated entity type prompt.
        """
        selected_utterance = selected_row['utterance']
        selected_domain = selected_row['domain']
        selected_intent = selected_row['intent']

        entity_types_in_intent = intents_entities.get(selected_intent, '')



        # TODO: Create several prompts and assign them at random for intent and entity tasks

        entity_type_prompt = self.entity_template.format(
            selected_utterance=selected_utterance, selected_domain=selected_domain,
            selected_intent=selected_intent, entity_types_in_intent=entity_types_in_intent)

        return entity_type_prompt

    def prepare_task_data(self) -> pd.DataFrame:
        """
        Prepare the data by generating prompts for the selected tasks.

        Returns:
        pd.DataFrame: A dataframe with the selected tasks.
        """
        task_dfs = []  # List to hold task dataframes

        if self.domain_intent_task:
            # Apply logic to generate domain intent training data
            self.df['intent_prompt'] = self.df.apply(self.get_domain_intent_prompt, axis=1)
            intent_task_df = self.df[['intent', 'intent_prompt']].rename(
                columns={'intent': 'answer', 'intent_prompt': 'prompt'})
            intent_task_df['task_type'] = 'intent'
            task_dfs.append(intent_task_df)

        if self.entity_bracket_task:
            # Apply logic to generate entity training data
            self.entity_template = self.config["prompt_templates"]["entity_bracket_template"]
            intents_entities = self.get_intents_and_entities()
            self.df['entity_type_prompt'] = self.df.apply(
                lambda row: self.get_entity_type_prompt(row, intents_entities), axis=1)
            entity_task_df = self.df[['annotated_utterance', 'entity_type_prompt']].rename(
                columns={'annotated_utterance': 'answer', 'entity_type_prompt': 'prompt'})
            entity_task_df['task_type'] = 'entity'
            task_dfs.append(entity_task_df)

        if self.entity_slot_task:
            # Apply logic to generate entity slot training data
            self.entity_template = self.config["prompt_templates"]["entity_slot_template"]
            intents_entities = self.get_intents_and_entities()
            self.df['annotated_utterance'] = self.df['annotated_utterance'].apply(
                self.convert_annotated_utterance_to_slots)
            self.df['entity_type_prompt'] = self.df.apply(
                lambda row: self.get_entity_type_prompt(row, intents_entities), axis=1)
            entity_task_df = self.df[['annotated_utterance', 'entity_type_prompt']].rename(
                columns={'annotated_utterance': 'answer', 'entity_type_prompt': 'prompt'})
            entity_task_df['task_type'] = 'entity'
            task_dfs.append(entity_task_df)


        # Combine all task dataframes
        task_df = pd.concat(task_dfs, ignore_index=True)
        return task_df

def export_to_csv(df: pd.DataFrame, file_path: str):
    """
    Export the given dataframe to a CSV file at the specified file path.

    Parameters:
    df (pd.DataFrame): Dataframe to export.
    file_path (str): Full file path to export the CSV file.
    """
    # Split the file path to get the directory
    directory = os.path.dirname(file_path)

    # Check if the directory exists, if not, create it
    os.makedirs(directory, exist_ok=True)

    # Export the dataframe
    df.to_csv(file_path, index=False)
    print(f"Data exported successfully to {file_path}")