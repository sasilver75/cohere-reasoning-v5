

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