# Inflight, Intrinsic Self-Correction Evaluation (v5)

## Installation

Make sure you have [Pyenv](https://github.com/pyenv/pyenv?tab=readme-ov-file#installation) installed with the correct version (3.12.6) of Python available.
```bash
pyenv install 3.12.6
```

Now, with this repository as your current working directory,create your virtual environment and install dependencies:
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pre-commit install
```

To use the Cohere API, you'll need an API key. Create an `.env` file with the following:
```bash
COHERE_API_KEY=your_actual_api_key_here
```
This environment variable file is git-ignored, so your precious credentials won't be checked into the git repository.

## Processing data
#### Scripts:
- `downloaders`: Contains scripts to download and adapt datasets from HuggingFace (e.g. NuminaMath CoT, Zebralogic, MATH, etc.) - Generates CSVs with {row_id, problem, solution} columns.
    - `download_numina.py`: Downloads the NuminaMath CoT dataset (cn_k12 subset) from HuggingFace, adapts it to our format, and saves it to a CSV file.
    - `download_zebralogic.py`: Downloads the Zebralogic dataset from HuggingFace, adapts it to our format, and saves it to a CSV file.
    - `download_math.py`: Downloads the MATH dataset from HuggingFace, adapts it to our format, and saves it to a CSV file.
- `generate_solvable_problem_solutions.py`: For a model under evaluation, generate a number of solutions for each of a collection of problems, with the goal of identifying problems which are "solvable" by the model, but not trivially so (determined by the success % rate for an invidiaul problem). 
- `generate_incorrect_solutions_off_policy.py`: For the row_ids identified as "solvable," generate a number of verified-as-incorrect solutions for each problem, using a model that is NOT the model under evaluation (Off-Policy, e.g. Command-R).
- `generate_incorrect_solutions_on_policy.py`: For the row_ids identified as "solvable," generate number of verified-as-incorrect solutions for each problem, using a model that is IS the model under evaluation (On-Policy, e.g. Command-R+ or LLaMA 3 405B).
- `generate_completions.py`: Given a collection of incorrect solutions (either on-policy of off-policy), use the model under evaluation to generate completions for a truncated prefix of each incorrect solution. Verify the correctness of the generated completion, and save the results to a CSV file.
#### Experiment Configuration:
- `model_providers.py`: Contains the mappings between models and their appropriate OpenRouter providers.
- `experiment_helpers.py`: Contains the implementations of the `Helper` class, like `CohereExperimentHelper`, which encapsulate logic around generating a (straight-shot) solution, verification, and completion generation. To run an experiment on a new model, you'll likely need to create a new `Helper`-implementing class in this file for your model and inference provider.
- `prompts.py`: Contains the prompts used for the experiment.
- There are additionally some experiment-level hyperparmeters (like the number of solutions per problem, when evaluating problem difficulty level) that can be configured for each of the files in the "Scripts" subsection above.

## Viewing results
- You can use the `view_completions.py` script to manually review the result of the `generate_completions.py` script. Change the `EXPERIMENT_NAME` variable to the path to the name of your most recent experiment. This is a Flask app, so you can run the script and view the results in your browser.
- This script is useful to have a qualitative understanding of the apparent recoveries. It's the case that there is "noise" in the recovery rates, because our naive truncation/prefixing strategy for incorrect solutions doesn't always capture the flaw in reasoning that resulted in an incorrect solution.


## Tips
- To maintain throughput, you might be setting Timeouts on your API calls to (eg) OpenRouter -- but consider the tokens per second (tps) that your provider is producing -- OpenRouter providdes insight into this under "https://openrouter.ai/activity", where you can see the tps of recent requests. Consider how many tokens your completer models will be generating for your dataset that you're evaluating, and set timeouts appropriately so that you're not wasting requests.
- If you want to test some changes to the pipeline of scripts, I've created a DummyExperimentHelper class in `experiment_helpers.py` that will allow you to run the scripts without making any API calls. This is useful for testing certain changes to the pipeline without having to spend money on API calls.
- In `generate_completions.py`, we're only generating a single coroutine for each incorrect solution, and running them concurrently. As a result, you might see "laggards" as your completions near completion, if those requests had to be retried.