# nlu-nlg-engine
**NLU**: Encoder-Decoder zero shot domain (skill), intent, and entity tagging NLU model.
**NLG**: Encoder-Decoder zero shot response generation.

This is a work in progress. Once it works really well, this repo will be moved to a Secret Sauce AI repo.

Once this is done, we will work on adding NLG capabilities to the model.

## Model
The model used is an encoder-decoder model. We use t5-based models. The model I personally work on is `flan-t5`. Why are we using this model? It is good to use a model that can do both intent and entity tagging. We want to have one model to do both, and we want it to be zero shot. The use case is for people who have limited computational resources, such as a raspi4. Having multiple models wouldn't be ideal. We want it to be zero shot, so that it works with any intents and entities for voice assistants. Flan is pretty good for this since it was pre-trained using prompting on a variety of NLP tasks. Generally, encoder-decoder models perform better on the kind of tasks we are interested in than decoder only models when there is a constraint on computational resources.

## 1. Creating a training data set

### NLU base data set
I used the [NLU Evaluation Dataset](https://github.com/xliuhw/NLU-Evaluation-Data) as a basis for the data set. This data set isn't the highest quality. Therefore, I worked on it and created a smaller data set of higher quality data. See [NLU engine prototype benchmarks](https://github.com/secretsauceai/NLU-engine-prototype-benchmarks) for details.

For testing purposes, the [Snips data set](https://github.com/snipsco/snips-nlu-metrics/tree/master/samples) was also used. 

The data was formatted into the following columns:
- `utterance`: The user's input such as 'wake me up at 7am'
- `domain`: The domain of the user's input such as 'alarm'. This can be thought of as the skill the user is trying to use.
- `intent`: The user's intent such as 'set_alarm'. This can be thought of as the action the user is trying to perform.
- `annotated_utterance`: The user's input with the entity tags such as 'wake me up at [time : 7am]'. For the Snips data set, the entity tags are in the form of '0 0 0 0 time'.

### NLG base data set
I created a small NLG data set consisting of the following columns:
- `domain`: The domain of the user's input such as 'alarm'. This can be thought of as the skill the user is trying to use.
- `intent`: The user's intent such as 'set_alarm'. This can be thought of as the action the user is trying to perform.
- `annotated_utterance`: The user's input with the entity tags such as 'wake me up at [time : 7am]'.
- `api_call`: The API call that would be made to fulfill the user's request such as a GET or POST request.
- `api_response`: The response from the API call which is a dictionary of the response from the API.
- `nlg_response`: The response that the voice assistant would give to the user such as 'I have set an alarm for 7am'.

### Creating our processed data set

#### Processing a data set
You can use the module `modules/data_preprocessing.py` to generally preprocess a data set.

#### Processing training data
We need to take our data and process it into a format we can use with the model. We want to turn our data into prompted tasks for our model to learn. 

In the `config/training_data_processing_config.toml` file, we specify:
- `data_file`: The path to the data set
- `processed_data_file`: The path to the processed data set
- `domain_intent_task`: Do we want to create prompts for intent training? (true or false)
- `entity_bracket_task`: Do we want to create entity prompts with square brackets? (true or false), e.g. 'wake me up at [time : 7am]'
- `entity_slot_task`: Do we want to create entity prompts with slots? (true or false), e.g. '0 0 0 0 time'
- `nlg_task`: Do we want to create prompts for NLG training? (true or false)
- `domain_intent_template`: The template for the intent prompts
- `entity_bracket_template`: The template for the entity prompts with square brackets
- `entity_slot_template`: The template for the entity prompts with slots
- `nlg_template`: The template for the NLG prompts

Then we can run the following command to process the data:
```bash
python -m scripts.training_data_preprocessing
```

## 2. Training the model
The training can be configured in the `config/training_config.toml` file.

We can then run the following command to train the model:
```bash
python trainer.py
```

### 3. Inference

#### Run test inference
You can test the model on a test set where the intents and/or entities are already labeled. This requires pre-processing the data beforehand so you have your intent/entity prompts ready. 

You can configure the test inference in the `config/test_inference_config.toml` file.

We can run the following command to make predictions on the test data:
```bash
python -m scripts.run_test_inference
```

#### Run inference
We can then use the trained model to make predictions on a data set where we might not have the intents/entities annotated. This does not require pre-processing the data beforehand. It will take the columns:
 - `utterance`
 - `domain`
 - `intent`
 - `annotated_utterance`

 and make predictions on the intent and/or entities. If you are doing entity prediction, you will still need to have the `intent` and domain for the entity prompt. So if you really want to see how it does, predict both intents and entities.

You can configure the inference in the `config/inference_config.toml` file.


We can then run the following command to make predictions:
```bash
python -m scripts.run_inference
```
