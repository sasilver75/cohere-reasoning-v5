

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