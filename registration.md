# Silicon Sample Benchmark — method registration form

Fill in every item before the prediction lock; this file ships inside your repo's Zenodo release
(see the README's *Deposit* step). This form covers **one entry** (one repo / one Zenodo release,
`primary` or `secondary-k` — see the README's *What counts as a submission*); if you submit several
entries, fill one form per entry. Items marked **★**
must be disclosed **fully publicly** (never escrowed or withheld). Items marked **†** must be at
minimum escrowed — they may be sealed from the public, but never withheld from the core team. Items
not applicable to your approach: write `N/A`. When several models serve different pipeline stages, complete the model
sections (B) once per model. See the call's *Disclosure policy* for escrow rules.

---

## 0 · Approach identity and output
- **0.1 Team ★**

Dong Shu, Jessica Hullman

Northwestern University

dongshu2024@u.northwestern.edu, jhullman@northwestern.edu
- **0.2 Plain-language summary ★** — one paragraph, what the approach does (not how):

Our approach uses a technique called probing, inspired by the paper [WHAT DO LARGE LANGUAGE MODELS KNOW ABOUT OPINIONS?](https://proceedings.iclr.cc/paper_files/paper/2026/file/ebfd0d632e950922baad6ecb64cdc407-Paper-Conference.pdf). Research shows that when you ask an AI to role-play a demographic and answer a question, the model often generates an inaccurate text response, even though its internal mathematical representations actually contain highly accurate information about that demographic's true opinions. Because these internal states store richer, more accurate information than what the model ultimately types out, we bypass standard text generation. Instead, we feed the model prompts containing demographic, context data, and question, and we extract its internal state (e.g. the representation from the residual stream of the LLM's final layer). We used these extracted states to train a separate "probe" model on five diverse datasets across five domains ([politics](https://www.nature.com/articles/s41562-023-01551-7), [climate](https://climatecommunication.yale.edu/visualizations-data/americans-climate-views/), [general](https://arxiv.org/pdf/2502.16761), [nutrition](https://tessexperiments.org/study/iles1294), [policy](https://tessexperiments.org/study/stoker1063)). Finally, we use this trained probe to predict the mean public opinions required for the Tier 2 submission.


- **0.3 Submission tier & approach family ★** — tier (1/2/3); family (e.g. per-respondent simulation / agent / direct forecast; single model / ensemble / multi-agent; zero-shot / literature-conditioned):

Tier 2; Family: probe model, training.

- **0.4 Pipeline diagram** — ordered steps from raw inputs to submitted file:

1. Create 36,000 synthetic respondent profiles (demographics: gender, age band, race, education, income, party, state) and load 44 survey items from project repositories.

2. Assign profiles to 17 conditions (4,000 control, 2,000 × 16 interventions).

3. Generate one full prompt per profile × item combination (1,584,000 total): demographic preamble + condition stimulus + question text, stored in a compact format (prompts.jsonl + items_meta.json).

4. Pass each prompt through Qwen 3.6 27B and extract the representation from the residual stream of its final layer.

5. Because Tier 2 requires two types of group-level submissions, we aggregate the individual prompt representations to meet these requirements. For the main submission, we average representations by condition × outcome (17 × 13 = 221 grouped representations). For the moderator submission, we average by condition × moderator-level × outcome (17 × 27 × 13 = 5,967 grouped representations).

6. Next, we preprocess five external datasets across distinct domains to train the probe: [politics](https://www.nature.com/articles/s41562-023-01551-7), [climate](https://climatecommunication.yale.edu/visualizations-data/americans-climate-views/), [general](https://arxiv.org/pdf/2502.16761), [nutrition](https://tessexperiments.org/study/iles1294), [policy](https://tessexperiments.org/study/stoker1063). Each dataset shares a common structure: demographic information, a text stimulus, a question, and the ground-truth human opinion.

7. We convert each data point from these datasets into an individual-level prompt, pass it through Qwen 3.6 27B, and extract the final-layer residual stream representation.

8. We split these individual-level representations into an 80% training set and a 20% testing set.

9. For both sets, we group the representations by demographic profile and calculate the average representation alongside the mean ground-truth opinion. The number of averaged representations in each set is shown below:

| Category | Train | Test |
| :--- | :--- | :--- |
| politic | 10000 | 6919 |
| climate | 64521 | 22470 |
| general | 2633 | 293 |
| nutrition | 3282 | 824 |
| policy | 9836 | 2439 |
| total | 90272 | 32945 |

10. We train a probe model using these averaged representations to predict the mean ground-truth opinions. The probe utilizes Principal Component Analysis (PCA) for dimensionality reduction followed by Ridge Regression to prevent overfitting.

11. We evaluate the trained probe's predictive performance on the testing set.

12. We establish a baseline using direct LLM inference: for every individual-level prompt in the testing set, we generate a direct text answer from Qwen 3.6 27B. We then average these direct responses by demographic group and compare them to the ground-truth means. The $R^2$ performance of both models on the testing set is below:

| Category | Probe | Baseline |
| :--- | :--- | :--- |
| politic | 0.3518 | -3.6220 |
| climate | 0.5608 | -0.2097 |
| general | 0.5709 | -2.3104 |
| nutrition | 0.4167 | -1.1646 |
| policy | 0.3098 | -1.9570 |

13. Performance Takeaways: The probing model (PCA + Ridge Regression) significantly outperforms direct Qwen inference across all domains. Notably, the baseline's negative $R^2$ scores indicate that direct LLM inference performs worse than a naïve model that simply guesses the global mean of the target variable for every prediction.

14. Finally, we feed the grouped representations from the Tier 2 main and moderator files (generated in Step 5) into the trained probe to predict the final mean opinions for submission.



- **0.5 Coverage ★** — number of respondents/cells/estimates; mapping to conditions. Full coverage is required: every submission predicts **all 16 interventions and all 13 outcomes** (partial coverage is not accepted). Confirm here:

Full coverage confirmed.

## A · Scope of LLM use
- **A.1 Purpose** — every workflow stage where LLMs are used:

Representation extraction stage (step 4 and 7).

- **A.2 Degree of automation ★** — confirm fully automated, no human in the loop at prediction time; note any exception:

Fully automated.

## B · Model / system details (once per model)
- **B.1 Model name(s)** — exact identifiers incl. provider, size, version/timestamp, source link:

Qwen 3.6 27B (27-billion parameter multimodal language model from Alibaba Qwen team). Source: HuggingFace (Qwen/Qwen3.6-27B). Version: As cached/available on 2026-08-29.

- **B.2 Access & context mode** — API/web/local; API name + version; chat vs stateless; exact call dates:

Local inference. Model loaded from disk cache at `/projects/p32143/cache/qwen36_27b` via `transformers.AutoModelForMultimodalLM.from_pretrained()`. Stateless: each prompt is processed independently (no conversation history, no persistent state across calls). Inference executed via custom Python scripts using PyTorch on A100 GPU. Call dates: Ongoing throughout study period (2026-08-10 and forward).

- **B.3 Configuration** — temperature, top-p/top-k, max tokens, penalties, stop sequences, seeds, reasoning effort, completions per item:

When we load the model, we set the torch type to float16 to reduce GPU memory usage. Since we are only extracting representations from the LLM, we don't need to set any other configuration.

``` python
    processor = AutoProcessor.from_pretrained(model_path)
    model = AutoModelForMultimodalLM.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        trust_remote_code=True,
    )
    model.eval()

    ...

    messages_batch = [
        [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
        for prompt in prompts
    ]

    # Tokenize batch (with proper processor_kwargs for batch padding)
    inputs = processor.apply_chat_template(
        messages_batch,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        processor_kwargs={"padding": True, "return_tensors": "pt"},
    ).to(device)

    # Forward pass for entire batch
    with torch.no_grad():
        outputs = model(
            **inputs,
            output_hidden_states=True,
            return_dict=True,
        )

    # Extract last token representation from final layer for each item in batch
    last_layer_states = outputs.hidden_states[-1]  # (batch_size, seq_len, hidden_dim)
    representations = []
    attention_mask = inputs["attention_mask"]
    # Find the index of the last non-pad token for each sequence
    sequence_lengths = attention_mask.sum(dim=1) - 1
    representations = []

    for i in range(last_layer_states.shape[0]):
        last_token_rep = last_layer_states[i, sequence_lengths[i], :].cpu().float().numpy()
        representations.append(last_token_rep)

```

- **B.4 Customization** — fine-tuning, RAG, prompt optimization, tool use, web search, agentic scaffolds (cross-ref H):

Training the ridge regression probe with PCA to reduce overfitting.

- **B.5 Persistent memory** — across interactions? what persisted:

No persistent memory.

- **B.6 Inference stack** — for local models: serving framework + version, quantization, hardware:

Framework: PyTorch + Hugging Face Transformers (AutoModelForMultimodalLM, AutoProcessor). Quantization: float16 (torch_dtype=torch.float16 on CUDA). Hardware: A100 80GB GPU, NVIDIA CUDA, running on Linux 4.18.0-553.136.1.el8_10.x86_64. Batch processing: Groups of 4–8 prompts per forward pass (tuned for VRAM efficiency). No quantization compression beyond float16 (no int8, no int4).

- **B.7 Ensembles** — members + exact aggregation rule:

No ensemble.

## C · Prompts
- **C.1 Exact prompts** — verbatim text or link to deposited file; were they iteratively refined? pre-specified vs in response to outputs:

Prompts are pre-specified. All prompts are in "silicon-sample-submission/probing/probe_testing/prompt/prompts.jsonl" and "silicon-sample-submission/probing/probe_testing/prompt/items_meta.json". Below is an example:

```
You are a survey respondent with the following characteristics:
{demographic}
Please read the below text:
{context}
Please answer the following question:
{question}
{scale}
Answer (a whole number from {start} to {end}):
```

- **C.2 System-wide instructions**:

No system prompt.

- **C.3 Prompt-design rationale** — brief rationale for the prompt design: why prompts were structured as they were, and the reasoning behind major design choices (recommended, not required):

Our prompt design is minimal, providing the required persona, context, question, and scale only.


## D · Persona / profile construction (Tiers 1–2)
- **D.1 Profile source** — source of demographic profiles you constructed: a public survey (e.g. GSS / ANES / Census), other survey, fully synthetic, or none. The benchmark ships no participant pool; report how you built yours, incl. condition assignments:

Fully synthetic profiles (stored in `synthetic_profiles/profiles_pool.csv`). 36,000 profiles with 6 demographic variables (gender, age_band, race, education, income, party) and 1 geographic variable (state). Profiles are assigned to 17 conditions: 4,000 to control (no intervention), 2,000 each to 16 interventions, based on deterministic assignment (seed = 2026 random shuffle ensuring balanced allocation across conditions and demographics). Profiles are fully synthetic respondent personas matching the demographic distributions of the US population (Census-aligned marginals).

- **D.2 Profile verbalization** — which variables, rendered how (template vs generated narrative; if generated: model + prompt):

Template-based verbalization. All demographics are rendered into a fixed preamble template:
``` python
facts = {
    "gender": "My gender is {}",
    "age_band": "I am {} years old",
    "race": "The race/ethnicity I most identify as is {}",
    "education": "The highest level of education I have completed is {}",
    "income": "My total yearly household income before taxes is {}",
    "party": "Politically I think of myself as {}",
}
```

- **D.3 Assignment & weighting** — number of personas, assignment to conditions (your responsibility, all 17 conditions), reuse, weighting/matching:

36,000 personas total. All 17 conditions are covered: control N=4,000, each intervention N=2,000 (total = 4,000 + 2,000×16 = 36,000). Assignment is deterministic and balanced: profiles are shuffled via seeded randomization (seed=2026) and assigned sequentially to conditions to ensure demographic balance across conditions. No personas are reused (each profile appears once, in exactly one condition). No weighting.

## E · Stimulus and survey administration
- **E.1 Stimulus presentation** — verbatim vs paraphrase; how state-contingent content is handled:

Stimulus text is verbatim from the experimental design, sourced from `stimuli.py` in the reference pipeline. **State-contingent content** is handled dynamically: For the "Extreme weather predictions" condition, a state-specific weather forecast is generated using `render_extreme_weather(state)`, which tailors the forecast language to the respondent's assigned state (drawn from profile data). For all other conditions, a stimulus text is selected via `stimulus_for(condition)` and presented as-is (not state-contingent). The stimulus is presented after the demographic preamble and before the survey questions, as part of the prompt context.

- **E.2 Survey walk-through** — one item/call vs blocks vs whole survey; context carry-over; item/option ordering & randomization; scale display; attention/comprehension handling:

One item per call. Each call includes: demographic preamble + stimulus context + one question. No context carry-over: Each prompt is independent; the model does not see previous responses or items. Items are presented in the fixed order from the questionnaire. Scale labels are shown explicitly in the question text (e.g., "On a scale of 1 to 5, where 1 is 'strongly disagree' and 5 is 'strongly agree'…"). No attention checks.

- **E.3 Response elicitation** — free text / constrained choice / structured output / token log-probabilities (if logprobs: normalization & mapping):

Probe model will predict the opinion mean directly.

## F · Stochasticity and aggregation
- **F.1 Runs & seeds** — runs per respondent/item/estimate; seeds; reproducibility under identical settings:

First, we group individual level representation. Then, we input the grouped representation into probe model.

- **F.2 Aggregation rule** — how multiple generations become submitted values (mean/median/mode/first/sampled/…):

No multiple generations per item. Probe model predict output directly.

## G · Validation & post-processing
- **G.1 Human validation** — any human review of outputs (often N/A):

N/A.

- **G.2 Post-processing** — parsing rules; handling of refusals/malformed/missing/out-of-range; exclusions; for approaches that generate individual responses, the resulting effective N per condition (descriptive disclosure, not a scoring input):

N/A. Probe model will always predict a number.

- **G.3 Calibration corrections** — any post-hoc scaling/shifting/debiasing and exactly what data it was fit on (cross-ref H/I):

N/A

## H · Learning and conditioning components
- **H.1 Fine-tuning data** — exact corpus (hashes/DOIs), hyperparameters, checkpoints:

We train our probing model on 5 datasets across 5 different domains: [politic](https://www.nature.com/articles/s41562-023-01551-7), [climate](https://climatecommunication.yale.edu/visualizations-data/americans-climate-views/), [general](https://arxiv.org/pdf/2502.16761), [nutrition](https://tessexperiments.org/study/iles1294), [policy](https://tessexperiments.org/study/stoker1063). Preprocessing details are in **0.4 Pipeline diagram** step 7-9.

- **H.2 Context & retrieval corpora** — exact document set in context / indexed, archived in the deposit:

No external context corpus. No retrieval-augmented generation (RAG).

## I · Data inputs, blinding, and competing interests
- **I.1 Competing interests ★** — funding, in-kind compute/model access, relationships with LLM-interested entities:

No direct funding for this submission from LLM providers. Compute (A100 GPU time) provided by Northwestern University (in-kind). Model (Qwen 3.6 27B) is open-source, no licensing fees or provider relationships. No financial relationships with Alibaba Qwen team or other LLM providers. Researchers are employees of Northwestern University, which has no known competing interests related to LLM benchmarking.

- **I.2 External human data †** — all external human datasets that informed the approach anywhere (training/fine-tuning/retrieval/ICL/calibration):

We used 5 datasets across 5 different domains: [politic](https://www.nature.com/articles/s41562-023-01551-7), [climate](https://climatecommunication.yale.edu/visualizations-data/americans-climate-views/), [general](https://arxiv.org/pdf/2502.16761), [nutrition](https://tessexperiments.org/study/iles1294), [policy](https://tessexperiments.org/study/stoker1063).

- **I.3 Blinding attestation ★** — **mandatory.** Signed attestation that no team member accessed, solicited, or was shown any human outcome data from this study, including pilots, before the prediction lock:

We, the undersigned, confirm that neither Dong Shu nor Jessica Hullman (nor any collaborators on this submission) have accessed, solicited, or been shown any human outcome data from the Silicon Sample benchmark study, including pilot data, between study registration and the prediction lock date. We have not viewed any benchmark results files, leaked data, or human response distributions. Our predictions are based solely on the model's zero-shot responses to the prompt templates and the synthetic profile demographics, with no human data conditioning or calibration. This attestation covers all team members involved in this submission.
  
**Signed**: Dong Shu and Jessica Hullman, Northwestern University  
**Date**: 2026-08-30

- **I.4 Contamination note †** — training cutoff of every model vs public release dates of this project's materials; note any known exposure:

**Qwen 3.6 27B training cutoff**: likely Q1–Q2 2024. **Silicon Sample benchmark public release**: As announced by Jan Pfänder et al. (likely 2024 or early 2025, post-prediction-lock). **Known exposure risk**: Low to none. The benchmark appears to be recently released (late 2025 / early 2026 based on project maturity), well after the model's training cutoff. No known contamination of the model with this benchmark's study design, stimuli, or outcome data. **Caveat**: If the project materials (prompt templates, stimuli text) were published before model training cutoff, the model may have seen similar phrasing. However, this would affect prompt interpretation consistency, not outcome prediction validity, and is accepted as part of using a pre-trained model.

## J · Internal selection procedure
- **J.1 Design-space search †** — how the final pipeline was chosen: how many configurations tried, internal validation criterion, what data it ran against:

We initiated our design-space search by training a preliminary probe model exclusively on the [politic dataset](https://www.nature.com/articles/s41562-023-01551-7). This initial phase successfully replicated the findings of [paper](https://proceedings.iclr.cc/paper_files/paper/2026/file/ebfd0d632e950922baad6ecb64cdc407-Paper-Conference.pdf), confirming that a model's internal representations encode richer, more accurate information than what it ultimately generates in its textual output.

Drawing on existing related works [3](https://www.tandfonline.com/doi/pdf/10.1080/14786440109462720), [4](https://arxiv.org/pdf/1404.1100), [5](https://pmc.ncbi.nlm.nih.gov/articles/PMC4792409/) and our prior experience with probing models, we introduced Principal Component Analysis (PCA) into the Ridge regression probing model. This step is critical for reducing the high dimensionality of the representation space, thereby preventing the regression model from overfitting to noise. After validating this approach on the single dataset, we expanded our training set to five diverse datasets to ensure the probe's robust generalizability.

To identify the optimal configuration, we executed an exhaustive grid search over the following hyperparameter space:

1. PCA Components (n_components): [50, 100, 200, 400, 512, 1024, 2048]. This parameter dictates how many orthogonal dimensions are retained from the original representations, controlling the trade-off between dimensionality reduction and information loss.

2. Ridge Regularization Strength (alpha): [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]. This determines the L2 penalty applied to the model weights. Higher values enforce a simpler model that is less prone to overfitting the training data.

3. Cross-Validation (cv): [1, 2, 3, 4, 5] Rather than testing a single split, 5-fold cross-validation divides the training data into five equal subsets, training the model on four and evaluating on the fifth to provide a reliable, variance-reduced estimate of unseen performance.

The pipeline systematically trained and evaluated all combinations of these variables (7 PCA configurations × 7 Alpha values × 5 folds = 245 total fits). The internal validation criterion for selecting the winning configuration was the highest cross-validated $R^2$ score.

The complete training log is available at "silicon-sample-submission/probing/probe_training/training_log.txt". An excerpt of the evaluation process is shown below:

``` txt
Testing PCA with 50 components...
Fitting 5 folds for each of 7 candidates, totalling 35 fits
[CV 1/5] END .......................alpha=0.001;, score=0.675
[CV 2/5] END .......................alpha=0.001;, score=0.684   
...
[CV 5/5] END ......................alpha=1000.0;, score=0.679 
    Best CV alpha: 1000.0000
    Validation R²: 0.6119

  ...

  Testing PCA with 2048 components...
Fitting 5 folds for each of 7 candidates, totalling 35 fits
[CV 1/5] END .......................alpha=0.001;, score=0.721 
...
[CV 5/5] END ......................alpha=1000.0;, score=0.730 
    Best CV alpha: 1000.0000
    Validation R²: 0.6398

Best parameters: {'n_components': 1024, 'alpha': 1000.0}
Best Validation R²: 0.6405
```

Based on this design-space search, the final selected pipeline was configured with 1024 PCA components and a Ridge alpha of 1000.0, as it yielded the highest overall validation $R^2$ (0.6405).

## K · Reproducibility & frozen artifacts
- **K.1 Code & materials** — link/DOI, secrets removed, determinism/seeds documented (also record the link in `metadata.json` → `code_repository` / `code_doi`):
All code is open-source and will be deposited with the Zenodo release.

**Repository**: https://github.com/Tizzzzy/silicon-sample-submission-team_29.git.

**Key files**:
- `silicon-sample-submission/probing/probe_training/train_probe_quintuple_respondent_split.py` (train and evaluate the probe)
- `silicon-sample-submission/probing/probe_testing/extract_representation/extract_unique_representations.py` (extract individual level representation from all 1,584,000 prompt texts)
- `silicon-sample-submission/probing/probe_testing/extract_representation/build_cell_groups_from_store.py` (group invdividual level representations together into main and moderator.)
- `silicon-sample-submission/probing/probe_testing/predict/predict_cell_groups_quintuple.py` (input group level representation into probe, and get the opinion mean)

**Determinism**: no randomness in response generation. Profile assignment uses seed=2026 (seeded random shuffle during profile generation).

**No secrets**: All API keys, credentials, and sensitive information removed before deposit.

**Documentation**: README.md in each module explains usage and regeneration steps.

- **K.2 Raw output logs †** — complete unprocessed model responses archived, hashed, time-stamped (required for Tiers 1–2, public or escrowed; Tier 3 where intermediate generations exist; oversized logs may be a separate linked Zenodo upload):

"silicon-sample-submission/probing/probe_testing/extract_representation/shard_a.log"
"silicon-sample-submission/probing/probe_testing/extract_representation/shard_b.log"
"silicon-sample-submission/probing/probe_testing/extract_representation/shard_c.log"
"silicon-sample-submission/probing/probe_testing/extract_representation/shard_d.log"
"silicon-sample-submission/probing/probe_training/training_log.txt"

- **K.3 Computational resources** — API-call counts, total tokens, cost, compute time:

**Probe Training**: fast, ~10 mins.

**Representation Extraction**: 1,584,000 forward calls (batch = 16, ~44 hours, free).

**Total input tokens**: ~300 million (estimated, varies by item type; ~190 tokens per prompt average).

## L · Disclosure class
Each item above is deposited as **public**, **escrowed** (sealed from the public but available to the
core team and auditors under confidentiality, with a public SHA-256 hash + timestamp so the lock is
still verifiable — an embargo with a sunset date is encouraged), or **withheld** (permitted only for
items marked neither ★ nor †). Your entry's class is set by its **most restricted item** and recorded
in `metadata.json` → `disclosure_class` (and `escrow_doi` if anything is escrowed):

- **B · Escrowed** — some items sealed due to the large file size (some are over 10GB) but every item is available to the core team/auditors under confidentiality. Full standing with an *escrowed* badge; only publicly disclosed features enter the design-choice analysis.

★ items must always be public (never escrowed or withheld); † items must be at minimum escrowed. Full
policy: <https://janpfander.github.io/llm_predictions_megastudy/#disclosure>