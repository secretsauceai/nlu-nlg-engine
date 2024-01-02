import toml
import pandas as pd
import re
import random
import os
from typing import Dict

class DataPreprocessor:
    def __init__(self, config_path: str):
        """
        Initialize the DataPreprocessor with configuration settings.

        Parameters:
        config_path (str): Path to the configuration file.
        """
        self.config = toml.load(config_path)
        self.df = pd.read_csv(self.config["data_file"], sep=',')
        self.domain_intent_training = self.config.get("domain_intent_training", False)
        self.entity_training = self.config.get("entity_training", False)

    def clean_and_structure_data(self) -> pd.DataFrame:
        """
        Clean and structure the dataframe by selecting necessary columns and handling renames.

        Returns:
        pd.DataFrame: The cleaned and structured dataframe.
        """
        if 'annotated_utterance_cleaned_and_removed_incorrect_tags' in self.df.columns:
            self.df = self.df.drop(columns=['annotated_utterance'])
            self.df = self.df.rename(columns={'annotated_utterance_cleaned_and_removed_incorrect_tags': 'annotated_utterance'})

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
        random_number = random.randint(3, number_of_domains)

        # select a random number of unique domains
        selected_domains = self.df['domain'].sample(n=random_number).unique().tolist()

        # check if selected domain is not in selected domains, if not add it in a random position in the list
        if selected_domain not in selected_domains:
            selected_domains.insert(random.randint(0, len(selected_domains)), selected_domain)

        domain_intents = {}
        for domain in selected_domains:
            domain_intents[domain] = self.df[self.df['domain'] == domain]['intent'].unique().tolist()

        domain_intent_prompt = f"""
        Given this utterance: '{selected_utterance}' pick the unique intent from these domain categories:
        {domain_intents}.
        If none of the intents match, return fallback.
        Intent:"""
        return domain_intent_prompt

    def get_entity_types_from_annotation(self, annotated_utterance: str) -> str:
        """
        Extracts entity types from the annotated utterance.

        Parameters:
        annotated_utterance (str): The annotated utterance text.

        Returns:
        str: A string of unique entity types joined by a comma.
        """
        entity_types = [match.split(' : ')[0].replace('[', '') for match in re.findall(r'\[.*?\]', annotated_utterance)]
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

    def get_entity_type_prompt(self, selected_row: pd.Series, intents_entities: Dict[str, str]) -> str:
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

        try:
            entity_types_in_intent = intents_entities[selected_intent]
        except:
            entity_types_in_intent = ''


        # TODO: Create several prompts and assign them at random for intent and entity tasks
        # TODO: Watch out for the current entity task, it gives the entity types in the prompt!

        entity_type_prompt = f"""
        ### Instructions
        Given an utterance, tag the entity or entities only when they match from the list of entity types.
        There can be more entities in an utterance and more than one entity type in an utterance.
        There can also be no entities in an utterance. In that case, just return the utterance.
        If there aren't any entity types, just return the utterance.
        To help, the domain and intent are provided for the utterance.

        ### Example
        Utterance: Set an alarm for 6am
        Domain: alarm
        intent: set_alarm
        Entity types: date, time
        Annotated utterance: Set an alarm for [time : 6am]

        ### Task
        Utterance: {selected_utterance}
        Domain: {selected_domain}
        Intent: {selected_intent}
        Entity types: {entity_types_in_intent}
        Annotated utterance:"""

        return entity_type_prompt

    def prepare_task_data(self) -> pd.DataFrame:
        """
        Prepare the data by generating prompts for the selected tasks.

        Returns:
        pd.DataFrame: A dataframe with the selected tasks.
        """
        task_dfs = []  # List to hold task dataframes

        if self.domain_intent_training:
            # Apply logic to generate domain intent training data
            self.df['intent_prompt'] = self.df.apply(lambda row: self.get_domain_intent_prompt(row), axis=1)
            intent_task_df = self.df[['intent', 'intent_prompt']].rename(columns={'intent': 'answer', 'intent_prompt': 'prompt'})
            intent_task_df['task_type'] = 'intent'
            task_dfs.append(intent_task_df)

        if self.entity_training:
            # Apply logic to generate entity training data
            intents_entities = self.get_intents_and_entities()
            self.df['entity_type_prompt'] = self.df.apply(lambda row: self.get_entity_type_prompt(row, intents_entities), axis=1)
            entity_task_df = self.df[['annotated_utterance', 'entity_type_prompt']].rename(columns={'annotated_utterance': 'answer', 'entity_type_prompt': 'prompt'})
            entity_task_df['task_type'] = 'entity'
            task_dfs.append(entity_task_df)

        # Combine all task dataframes
        task_df = pd.concat(task_dfs, ignore_index=True)
        return task_df

def export_to_csv(df: pd.DataFrame, file_name: str, directory: str = "data/processed"):
    """
    Export the given dataframe to a CSV file in the specified directory.

    Parameters:
    df (pd.DataFrame): Dataframe to export.
    file_name (str): Name of the CSV file.
    directory (str): Directory path to export the CSV file.
    """
    # Check if the directory exists, if not, create it
    os.makedirs(directory, exist_ok=True)
    
    # Define the full path
    full_path = os.path.join(directory, file_name)
    
    # Export the dataframe
    df.to_csv(full_path, index=False)
    print(f"Data exported successfully to {full_path}")

if __name__ == "__main__":
    # Define the path to your configuration file
    config_path = "config/config.toml"

    # Initialize the data preprocessor
    preprocessor = DataPreprocessor(config_path)

    # Clean and structure the data
    cleaned_data = preprocessor.clean_and_structure_data()

    # Prepare the data for tasks based on configuration
    task_data = preprocessor.prepare_task_data()

    # Export the task data to a CSV file
    export_to_csv(task_data, "task_data.csv")
