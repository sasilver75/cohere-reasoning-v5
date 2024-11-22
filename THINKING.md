

Things I want:

1. Use of token bucket instead of batching
    - Might need multiple buckets

2. Fixing the prefix to .7 (configurable)

3. Should be able to serve any model
    -  Requires different token buckets for each service.
    - Requires that calls to get solutoins and completions are are handled identically.
    - Some Class with Base Class?


--------------------------------

220 Homework
5 rules of tokenizer
Maybe tokenize by space, exlucde this symbol, etc.
The main point is that you can't use any existing tokenization libraries


Sam Note Nov 21 3:13
Was leavin the "test if the token bucket is working" thing runing... seems like it might be, though this was a bad script to test it on (since I'm serially checking problems on at a time.. but 10 attempst is enough to exhaust the one I used (5 capacity))

Anywas, once that's done we can regenerate on/offpolicy if we like.


------------------------------------

╰─❯ /Users/sam/code/cohere/cohere-reasoning-v5/venv/bin/python /Users/sam/code/cohere/cohere-reasoning-v5/generate_solvable_problem_solutions.py
Starting...
Loading problems from datasets/original/cn_k12_math_problems.csv...
Loaded 276589 problems.
Appraising problems until we've found 10 solvable problems...
Finding solvable problems:   0%|                                                                                                                                                                     | 0/10 [00:00<?, ?it/s]Success rate for row 159073: 0.0
Success rate for row 217998: 0.0
Success rate for row 234914: 1.0
Success rate for row 194964: 0.0
Success rate for row 239483: 0.33333333333333337
Finding solvable problems:  10%|███████████████▋                                                                                                                                             | 1/10 [01:00<09:01, 60.14s/it]Success rate for row 162713: 0.0
Success rate for row 106016: 0.0
Success rate for row 115435: 1.0
TokenBucket request count: 50
Success rate for row 91401: 0.0
Success rate for row 40669: 0.33333333333333337
Finding solvable problems:  20%|███████████████████████████████▍                                                                                                                             | 2/10 [02:08<08:39, 64.94s/it]Success rate for row 44424: 0.0
Success rate for row 217463: 0.0
Success rate for row 114941: 0.0
Success rate for row 202013: 0.33333333333333337
Finding solvable problems:  30%|███████████████████████████████████████████████                                                                                                              | 3/10 [02:58<06:47, 58.15s/it]Success rate for row 199486: 1.0
Success rate for row 245856: 0.0
TokenBucket request count: 100
Success rate for row 148210: 1.0
Success rate for row 131350: 0.0
Success rate for row 174230: 1.0
Success rate for row 54179: 0.0
Success rate for row 137187: 0.33333333333333337
Finding solvable problems:  40%|██████████████████████████████████████████████████████████████▊                                                                                              | 4/10 [04:32<07:14, 72.37s/it]Success rate for row 3870: 0.0
Success rate for row 230770: 1.0
Success rate for row 173567: 1.0
TokenBucket request count: 150
Success rate for row 250323: 0.0
Retrying models.CohereExperimentHelper.get_solution in 4.0 seconds as it raised TimeoutError: .
Success rate for row 189582: 0.0
Success rate for row 52144: 0.0
Success rate for row 104429: 0.0
Success rate for row 181636: 0.0
Success rate for row 76450: 0.0
Success rate for row 59503: 0.33333333333333337
Finding solvable problems:  50%|██████████████████████████████████████████████████████████████████████████████                                                                              | 5/10 [09:34<12:55, 155.11s/it]Success rate for row 188384: 0.6666666666666667
Finding solvable problems:  60%|█████████████████████████████████████████████████████████████████████████████████████████████▌                                                              | 6/10 [09:42<07:00, 105.12s/it]Success rate for row 80463: 0.0
TokenBucket request count: 200
Success rate for row 88421: 1.0
Success rate for row 77315: 0.0
Success rate for row 76156: 0.6666666666666667
Finding solvable problems:  70%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████▉                                           Retrying models.CohereExperimentHelper.get_solution in 4.0 seconds as it raised TimeoutError: .
Success rate for row 220754: 0.0
Success rate for row 94634: 0.33333333333333337
Finding solvable problems:  80%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▌                               | 8/10 [12:15<03:04, 92.11s/it]Success rate for row 11176: 1.0
Success rate for row 140455: 0.0
Success rate for row 8924: 0.6666666666666667
Finding solvable problems:  90%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▎               | 9/10 [13:01<01:17, 77.45s/it]TokenBucket request count: 250
Success rate for row 187773: 0.33333333333333337
Finding solvable problems: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 10/10 [13:15<00:00, 57.85s/it]Found 10 solvable problems.
Saving results to datasets/derived/interesting_problems_test.csv and datasets/derived/interesting_problems_test.txt...
Saved results to datasets/derived/interesting_problems_test.csv and datasets/derived/interesting_problems_test.txt.
Finding solvable problems: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 10/10 [13:15<00:00, 79.53s/it]
Done! Elapsed: 802.99s


This took like 13 minutes to generate 10 solvable problems (with ). That's not tenable!
# TUNABLE PARAMETERS
HELPER = CohereExperimentHelper()  # Encapsulates logic about the specific models we're using
SOURCE_PATH = Path("datasets/original/cn_k12_math_problems.csv")
SINK_PATH = Path("datasets/derived/interesting_problems_test.csv")
TARGET_N_SOLVABLE_PROBLEMS = 10  # The number of solvable problems we want to identify. Note that the stronger the model and the lower the success rate bounds, the more problems we'll have to evaluate (and the more requests we'll make)
N_SOLUTION_ATTEMPTS_PER_PROBLEM = 3  # For each problem, the number of solution attempts over which we'll evaluate problem difficulty. Note that without retries we'll have 2*{N_SOLUTION_ATTEMPTS_PER_PROBLEM} API calls per problem.
LOWER_SUCCESS_RATE_BOUND = .2  # The lower bound on the success rate of the solutions we'll accept as solvable/interesting; Number if [0, 1). Note that the lower the succcess rate bound, the more problems we'll have to evaluate here, but also less incorrect solution looping we'll have to do in in later scripts.
UPPER_SUCCESS_RATE_BOUND = .7  # The upper bound on the success rate of the solutions we'll accept as solvable/interesting; Number in [0, 1). Note that the lower the succcess rate bound, the more problems we'll have to evaluate here, but also less incorrect solution looping we'll have to do in in later scripts.
EPSILON = 1e-5  # To help with floating point division giving .199999 when it really should be .2. I don't think theres' really a reason to tune this.
SEED = 42  # Random seed for dataset shuffling; We'll iterate through rows of this shuffled dataset until we identify the target number of solvable problems.
# END OF TUNABLE PARAMETERS



Now let's try to parallelize it with semaphore and run the same experiment.
(We could later further parallelize it by paralellizing the serial generate/verify loops)