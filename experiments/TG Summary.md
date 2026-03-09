
This RAG Evaluation Application application was built to make a Chat bot trained strictly on the closed captions of the Office tv show. It is a training excercise with the idea of taking an extermely sparse corpus, and building intelligence around it.


The experimentation phases wer as follows: 
generally moving from cheap and simple to more complex and slightly more expensive:

The importance of this application is to express how important it is to make changes measurable. AI makes it so easy to implement changes, but it also creates massive amounts of context. It takes human judgement to intervine, direct, and orchestrate changes. 

Phases:

1. naive chunking, GPT 4.1 nano, cheap semantic. Expiermented with chunk sizes, k values, similarity search.

- Broad result: good at finding lines or episodes when somethign occurs. Poor at reasoning, some hallucination.
2. MMR (Maximum Marginal Relevance - reduce diversity of retrived results) - tuning knobs: fetch-k (initial results retrieved), lambda mult (relevance vs diversity 0 to 1) result: timeline questions slightly improved, no reasoning still, more hallucination at the LLM question step, as the a cheap llm infers across contexts (4.1 nano)

3. Scene leve chunking strategy: Often preferred on reddit, but when it's not an actual script, the data becomes more segmented, and too large to pass to cheap models. The overlap windows must be restrained,  creating issues. 
Result: incresed precision, decreated narrative cohesion. 72% more vector increases chance for semantic adjacency but narrative irrelevance. overly fragmented.

4. Query Expansion: (generate multiple semantically-aligned search queris to cover phrasinc / facets / increase vector accuracy) 
- Poor performance without Fusion

5. QE + RRF (reciprocal rank fusion): deduplicate the generated queries, retrieve on top 3, dedupe and rank the results. Important step for real world models, but did not improve actually thinking or results.  Poor results without MMR. 
With MMR is best results yet.


Transition Phase -> Index-Time Enrichment
A vector db on its own with a approximate nearest neihbor algorithm (HNSW, here), finds high quality approximation of nearest vectors, but doesn't naturally contain interperative glue (arcs, motifs, importance). So I shifted to enriching the data.

1. add metadata: Important step I should have done first. Time cost to re-evaluate chunking strategy, rewrite parsing scripts, produce good json off the raw data. Lower hanging fruit items that should have been incorporated first time.

- Increases accuracy and integrity, empowers reasoning additions, but does not add reasoning itself.

2. Build derived Corpus: 
Use map-reduce with GPT 4.1-nano (pyramid strateigy) to chunk episodes -> summaries -> seasons -> series. 
Sounds daunting, but quite easy and not very expensive. But the summaries and reductions stage through LLM evals was weak. Had to redo this with GPT 4.1-mini was hugely improved summaries. 
Difficult to retain provenance - chaining citations difficult. 

3. Routing: Once you have derived data, you have to direct how to use it, there are several strategies. I tested over a couple of simple patterns, but should have spent more time making a stronger algorithm. 
a. filter strategy by question type (regex): very poor 
b. tier questions: run through cards then scripts - mediocre, but could be explored

many other strategies to explore tehre.

4. Derived data Cards
- Increased derived data with chracter bios, relationship timelines, character arcs, etc. 
- Powerful for higher level reasonings, not helpful without strong choice of card details, provenance, and indexing.

# Phase 3:
Dashboard:
1. matlab plotlib insufficient -> plotly better
2. AI as judge - Had this in mind from startt but shoudl have enriched more data earlier to nto have to re-score runs.
A. first Gave the judge both gold_answer and retrieved_context. This can bias correctness upward even when retrieval is weak; the rubric tries to counterbalance via groundedness, but it’s still a common failure mode. 
B. Must use better LLM for answers for judgement, carefully craft gold-standards, and Prompt Engineer carefully. 
C. Even with (B) a much more robust pattern is 2-pass judging: (1) context-only groundedness, (2) gold-only correctness, then combine. 
D. Juding is extremely important and hugely increases the decision making capabilities. I got tired by the time I got here, and shoudl have considered judging from the start, and made more modularity in running sweeps, parameterizing more so I could re-test with new questions

## Scoring insights & mistakes:
- Start with 2+ phase and prompt engineer individually instead of cramming into one judge
- Cleaner data first: 
-   while having tons of metrics is nice, I should have also chosen specific data points earlier on to highlight (first and second classes), 
- Should have bookmarked best run results and important failures instead of asking AI to summarize later. Hand written notes much stronger and faster for retrospectives and forensics.
- Modularity: as the model grew, genreations gave better insights over individual tuning. Having th ability to pass different routing stratgies, judges, and questions asgainst older models would have helped better express performance gains.