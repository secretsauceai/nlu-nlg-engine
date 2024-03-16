# nlu-engine
Encoder-Decoder zero shot domain (skill), intent, and entity tagging NLU model. This is a work in progress. Once it works really well, this repo will be moved to a Secret Sauce AI repo.

Once this is done, we will work on adding NLG capabilities to the model.

## Model
The model used is an encoder-decoder model. We use t5-based models. The model I personally work on is `flan-t5`. Why are we using this model? It is good to use a model that can do both intent and entity tagging. We want to have one model to do both, and we want it to be zero shot. The use case is for people who have limited computational resources, such as a raspi4. Having multiple models wouldn't be ideal. We want it to be zero shot, so that it works with any intents and entities for voice assistants. Flan is pretty good for this since it was pre-trained using prompting on a variety of NLP tasks. Generally, encoder-decoder models perform better on the kind of tasks we are interested in than decoder only models when there is a constraint on computational resources.

## 1. Creating a data set

### Base data set
I used the [NLU Evaluation Dataset](https://github.com/xliuhw/NLU-Evaluation-Data) as a basis for the data set. This data set isn't the highest quality. Therefore, I worked on it and created a smaller data set of higher quality data. See [NLU engine prototype benchmarks](https://github.com/secretsauceai/NLU-engine-prototype-benchmarks) for details.

For testing purposes, the [Snips data set](https://github.com/snipsco/snips-nlu-metrics/tree/master/samples) was also used. 

The data was formatted into the following columns:
- `utterance`: The user's input such as 'wake me up at 7am'
- `domain`: The domain of the user's input such as 'alarm'. This can be thought of as the skill the user is trying to use.
- `intent`: The user's intent such as 'set_alarm'. This can be thought of as the action the user is trying to perform.
- `annotated_utterance`: The user's input with the entity tags such as 'wake me up at [time : 7am]'. For the Snips data set, the entity tags are in the form of '0 0 0 0 time'.

### Creating our processed data set
We need to take our data and process it into a format we can use with the model. We want to turn our data into prompted tasks for our model to learn. 

In the `config/data_processing_config.toml` file, we specify:
- `data_file`: The path to the data set
- `processed_data_file`: The path to the processed data set
- `domain_intent_training`: Do we want to create prompts for intent training? (true or false)
- `entity_bracket_training`: Do we want to create entity prompts with square brackets? (true or false), e.g. 'wake me up at [time : 7am]'
- `entity_slot_training`: Do we want to create entity prompts with slots? (true or false), e.g. '0 0 0 0 time'
- `domain_intent_template`: The template for the intent prompts
- `entity_bracket_template`: The template for the entity prompts with square brackets
- `entity_slot_template`: The template for the entity prompts with slots

Then we can run the following command to process the data:
```bash
python scripts/process_data.py
```

## 2. Training the model
The training can be configured in the `config/training_config.toml` file.

We can then run the following command to train the model:
```bash
python trainer.py
```

### 3. Inference
We can then use the trained model to make predictions. You can configure the inference in the `config/inference_config.toml` file.

We can then run the following command to make predictions:
```bash
python scripts/inference.py
```