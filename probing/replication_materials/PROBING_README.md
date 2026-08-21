# LLM Opinion Probing Dataset: Tappin et al. (2023) Partisan Persuasion Study

## Overview

This directory contains preprocessed data and prompts for conducting hidden-state extraction (probing) experiments with Large Language Models to study **what LLMs know about human opinions on political issues**.

This work follows the methodology of Jahanparast et al. (2026, ICLR) "What Do Large Language Models Know About Opinions?" but applied to the Tappin et al. (2023) partisan persuasion survey dataset.

## Files

- **`probing_prompts.json`** — Main output file containing 576 prompts (24 demographic groups × 24 policy questions) with ground-truth opinion means from human survey data
- **`preprocess_for_probing.py`** — Python script that generates the prompts from raw survey data
- **`data/data_RM_export.csv`** — Exported survey data (from RDS original)
- **`PROBING_README.md`** — This file

## Dataset Composition

### Source Data
- **Study**: Tappin et al. (2023) "Partisans' Receptivity to Persuasive Messaging is Undiminished by Countervailing Party Leader Cues" (Nature Human Behaviour)
- **Sample size**: 4,306 respondents, 21,382 observations (after filtering)
- **Survey date**: September 2021
- **Respondents**: U.S. adults identified as either Republican (Trump voters) or Democrat (Biden voters) in 2020 election

### Policy Questions (24 total)
Survey covered diverse policy domains:
- Immigration & border policy (amnesty, border restrictions)
- Taxation (capital gains, estate tax, tariffs)
- Social issues (religious denial, affirmative action, assisted suicide)
- Criminal justice (death penalty, drug sentencing, denying felons the vote)
- Labor & economics (union power, minimum wage requirements)
- Foreign policy (military aid, foreign aid)
- Civic participation (electoral college, donation limits, flag burning)

### Demographic Grouping Scheme (24 groups)
Respondents grouped by 4 key dimensions:
1. **Political Party** (2 levels): Democrat, Republican
2. **Gender** (2 levels): Male, Female  
3. **Education** (2 levels): High School or Less, College Degree
4. **Political Knowledge** (3 levels): Low (0-1/4), Medium (2/4), High (3-4/4)

Creates a complete factorial design: 2 × 2 × 2 × 3 = **24 demographic groups**

## JSON Output Format

### Structure
```json
{
  "metadata": {
    "source": "Tappin et al. (2023) - Partisan Persuasion Study",
    "n_observations": 21382,
    "n_respondents": 4306,
    "n_policy_questions": 24,
    "n_demographic_groups": 24,
    "n_probes": 576,
    "purpose": "LLM opinion probing with hidden-state extraction"
  },
  "demographic_groups": {
    "dimensions": ["party", "gender", "education", "political_knowledge"],
    "examples": ["Party=Biden, Gender=Male, Education=College Degree, Political Knowledge=High", ...]
  },
  "policy_questions": [
    {
      "question_id": 1,
      "label": "Allow religious denial of service",
      "text": "Allow businesses to deny service to a customer if...",
      "biden_position": "disagrees",
      "trump_position": "agrees"
    },
    ...
  ],
  "probes": [
    {
      "group_id": "party_biden_democrat_gender_male_education_college_degree_political_knowledge_high",
      "demographic_description": "Party=Biden, Gender=Male, Education=College Degree, Political Knowledge=High",
      "question_id": 1,
      "question_label": "Allow religious denial of service",
      "prompt": "You are a survey respondent with the following characteristics:\n...",
      "ground_truth_mean": 3.45,
      "ground_truth_std": 1.67,
      "n_respondents": 28,
      "scale_range": [1, 7],
      "scale_labels": ["Strongly disagree", "Disagree", "Somewhat disagree", "Neither", "Somewhat agree", "Agree", "Strongly agree"]
    },
    ...
  ]
}
```

### Key Fields in Each Probe

| Field | Type | Description |
|-------|------|-------------|
| `group_id` | string | Unique identifier for demographic group (underscore-separated key-value pairs) |
| `demographic_description` | string | Human-readable demographic group (e.g., "Party=Biden, Gender=Male, ...") |
| `question_id` | int | Policy question ID (1-24) |
| `question_label` | string | Short policy question label |
| `prompt` | string | **LLM Prompt** — instruct LLM to roleplay as survey respondent and rate agreement |
| `ground_truth_mean` | float | Mean opinion rating from human survey respondents in this group (1-7 scale) |
| `ground_truth_std` | float | Standard deviation of opinions in this group |
| `n_respondents` | int | Number of survey respondents in this group |
| `scale_range` | array | Rating scale bounds [min, max] |
| `scale_labels` | array | Labels for each point on the scale |

## How to Use for LLM Probing

### Step 1: Load Prompts
```python
import json
with open('probing_prompts.json', 'r') as f:
    probing_data = json.load(f)

# Access individual probes
for probe in probing_data['probes'][:10]:
    print(probe['demographic_description'])
    print(probe['prompt'])
    print(f"Ground truth: {probe['ground_truth_mean']:.2f}\n")
```

### Step 2: Extract LLM Hidden States (Using Transformers Library)

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_name = "meta-llama/Llama-2-7b-hf"  # or Qwen, Mistral, etc.
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    output_hidden_states=True,
    return_dict=True
)

probe = probing_data['probes'][0]
prompt = probe['prompt']

# Tokenize and forward pass
inputs = tokenizer(prompt, return_tensors='pt')
with torch.no_grad():
    outputs = model(**inputs)

# Extract residual stream activations from each layer
all_hidden_states = outputs.hidden_states  # Tuple of (batch, seq_len, hidden_dim)

# Access specific layer (e.g., layer 15 for a 32-layer model)
layer_15_activations = all_hidden_states[15]

print(f"Layer 15 shape: {layer_15_activations.shape}")  # (batch_size=1, seq_len, hidden_dim)
```

### Step 3: Train Probes to Predict Ground Truth (Following Jahanparast et al.)

For each layer, train a classifier to predict the ground-truth opinion distribution:

```python
from sklearn.linear_model import LogisticRegression
import numpy as np

# For a given layer and demographic group:
hidden_states_list = []
ground_truths = []

# Collect activations for all probes in this group
for probe in probes_for_this_group:
    # Get hidden states (as above)
    hidden_state = get_hidden_state(probe['prompt'], layer_idx=15)
    hidden_states_list.append(hidden_state.mean(axis=0))  # Average over sequence length
    ground_truths.append(probe['ground_truth_mean'])

# Train probe
X = np.array(hidden_states_list)
y = np.array(ground_truths)
probe_classifier = LogisticRegression(max_iter=1000)
probe_classifier.fit(X, y)

# Evaluate
predictions = probe_classifier.predict(X)
mse = np.mean((predictions - y)**2)
```

## Statistics Summary

- **Ground truth opinion means**: Range [1.41, 6.77] (scale: 1-7)
  - Mean across all probes: 4.54
  - Std dev: 0.89
- **Ground truth opinion std devs**: Range [0.45, 2.55] (within-group variability)
  - Mean: 1.74
- **Sample sizes per probe**: Range [5, 120]
  - Mean: 37 respondents per demographic group × question pair

## Methodology Differences from Jahanparast et al.

| Aspect | Jahanparast et al. | This Dataset |
|--------|-------------------|--------------|
| LLM evaluation method | Probing residual stream | Same approach |
| Demographic grouping | 22 US Census groups (gender, race, education, income, religion, ideology) | 24 groups (party, gender, education, political knowledge) |
| Opinion scale | 2-3 option multiple choice | 1-7 Likert scale |
| Data source | OpinionQA & SubPOP (Pew surveys) | Tappin et al. (persuasion study) |
| Policy topics | 321 diverse questions | 24 policy issues |
| Task | Predict answer distribution | Predict mean opinion rating |

## Citation

If you use this dataset for research, cite both original studies:

```bibtex
@article{tappin2023partisans,
  title={Partisans' receptivity to persuasive messaging is undiminished by countervailing party leader cues},
  author={Tappin, Ben M and Berinsky, Adam J and Rand, David G},
  journal={Nature Human Behaviour},
  volume={7},
  pages={568--582},
  year={2023},
  publisher={Nature}
}

@article{jahanparast2026opinions,
  title={What do large language models know about opinions?},
  author={Jahanparast, Erfan and Hong, Zhiqing and Chang, Serina},
  journal={ICLR},
  year={2026}
}
```

## Questions?

For issues or questions about the preprocessing:
- See `preprocess_for_probing.py` source code for full documentation
- Check the original Tappin et al. replication materials README.R for survey design details
- Refer to Jahanparast et al. (2026) methods for probing/SAE techniques

---

**Generated**: 2026-08-20
**Last updated**: 2026-08-20
