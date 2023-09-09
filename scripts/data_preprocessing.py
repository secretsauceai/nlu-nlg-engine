import toml
import pandas as pd
import re
import os



# Load the toml file
config = toml.load("nlu-engine/config/config.toml")

# Load the dataset
df = pd.read_csv(config["data_file"], sep=',')

def preprocess_function(df):
    '''
    This is because I am lazy and I will simply save this version of the file later. LOL
    '''
    # only use utterance, domain, intent, and annotated_utterance_cleaned_and_removed_incorrect_tags columns
    df = df[['utterance', 'domain', 'intent', 'annotated_utterance_cleaned_and_removed_incorrect_tags']]

    # rename annotated_utterance_cleaned_and_removed_incorrect_tags to annotated_utterance
    df = df.rename(columns={'annotated_utterance_cleaned_and_removed_incorrect_tags': 'annotated_utterance'})
    return df


def get_entity_types_from_annotation(annotated_utterance):
  """
  Extracts the entity types from the annotated utterance
  """
  entity_types = [match.split(' : ')[0].replace('[', '') for match in re.findall(r'\[.*?\]', annotated_utterance)]
  # turn entity_types list into a set and turn that into a joined string with a comma
  return ', '.join(set(entity_types))


def get_intents_and_entities(df):
    """
    Returns a dictionary of the intents and their entities
    """
    intents_entities = {}
    for index, row in df.iterrows():
        entity_types = get_entity_types_from_annotation(row['annotated_utterance'])
        if entity_types != '':
         try:
                intents_entities[row['intent']] = intents_entities[row['intent']] + ', ' + entity_types
         except KeyError:
             intents_entities[row['intent']] = entity_types

    # for the above dictionary, remove the duplicates from the values
    for key, value in intents_entities.items():
        intents_entities[key] = ', '.join(set(value.split(', ')))

    return intents_entities

#TODO: refactor this for the entity prompt

def get_entity_type_prompt(selected_row):
    selected_utterance = selected_row['utterance']
    selected_domain = selected_row['domain']
    selected_intent = selected_row['intent']

    try:
      entity_types_in_intent = intents_entities[selected_intent]
    except:
      entity_types_in_intent = ''



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
    Annotated utterance:
    """

    return entity_type_prompt

df = preprocess_function(df)
intents_entities = get_intents_and_entities(df)

data_set_df = pd.DataFrame()
data_set_df['entity_type_prompt'] = df.apply(lambda row: get_entity_type_prompt(row), axis=1)
data_set_df['annotated_utterance'] = df.annotated_utterance

# save the data set in data/processed folder, if the folder doesn't exist, create it
if not os.path.exists('data/processed'):
    os.makedirs('data/processed')
data_set_df.to_csv('data/processed/entity_data_set.csv', index=False)
