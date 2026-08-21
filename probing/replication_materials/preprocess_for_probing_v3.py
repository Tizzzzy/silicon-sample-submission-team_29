#!/usr/bin/env python3
"""
Preprocess Tappin et al. partisan persuasion dataset for LLM opinion probing (v4).

Improvements in v4:
1. Individual-level prompts (one per respondent per policy question)
2. Full 4D demographic profile for each individual
3. Actual condition each respondent experienced in the survey
4. Persuasive messages embedded based on their condition
5. Group-level ground truth (mean opinion for demographic group)
6. Natural language demographic descriptions

Workflow:
- Create individual prompts for each respondent × policy question
- Feed to LLM to extract representations
- Aggregate representations by demographic group
- Train probe using group-level ground truth

Output: ~540k+ individual-level probes with group-level ground truth
"""

import pandas as pd
import numpy as np
import json
from collections import defaultdict
from itertools import combinations
import warnings

warnings.filterwarnings('ignore')

# Load data
print("Loading data...")
df = pd.read_csv("data/data_RM_export.csv")

# Load persuasive messages
print("Loading persuasive messages...")
with open("data/persuasive_messages.json", 'r') as f:
    messages_data = json.load(f)

# Create a mapping from question ID to messages
messages_by_id = {p['id']: p for p in messages_data['policies']}

# Filter to analysis sample (matching Tappin et al. R scripts)
df = df[df['vote_party'].isin(['Biden-Democrat', 'Trump-Republican'])].copy()
df = df[df['likertAgree_recoded'].notna()].copy()

print(f"Analysis sample size: {len(df)} observations from {df['pid'].nunique()} respondents")

# Define demographic groupings
def create_demographic_groups():
    """Create demographic grouping logic"""
    groups = {}

    # Party affiliation
    groups['party'] = df['vote_party'].unique()

    # Gender (filter out "other" category as it's very small)
    groups['gender'] = {
        1.0: 'Male',
        2.0: 'Female'
    }

    # Education (group into 2 categories: < College vs >= College degree)
    groups['education'] = {
        'less_than_college': [1.0, 2.0, 3.0, 4.0],
        'college_or_higher': [5.0, 6.0, 7.0, 8.0, 9.0]
    }

    # Political knowledge (tertiles: 0-1, 2, 3-4)
    pk_tertiles = df['PK_sum'].quantile([0, 1/3, 2/3, 1]).values
    groups['political_knowledge'] = {
        'low': (pk_tertiles[0], pk_tertiles[1]),
        'medium': (pk_tertiles[1], pk_tertiles[2]),
        'high': (pk_tertiles[2], pk_tertiles[3])
    }

    return groups

demographic_groups = create_demographic_groups()

print("\nDemographic grouping scheme (primary 4 dimensions):")
for dim, vals in demographic_groups.items():
    if isinstance(vals, dict):
        print(f"  {dim}: {len(vals)} categories")
    else:
        print(f"  {dim}: {len(vals)} categories")

# Create demographic group assignments for each respondent
def assign_groups(row):
    """Assign respondent to demographic groups"""
    assignment = {}

    assignment['party'] = row['vote_party']

    if row['gender_survey'] in [1.0, 2.0]:
        assignment['gender'] = demographic_groups['gender'].get(row['gender_survey'], None)
    else:
        assignment['gender'] = None

    # Education
    ed = row['education_survey']
    if ed in demographic_groups['education']['less_than_college']:
        assignment['education'] = 'High School or Less'
    elif ed in demographic_groups['education']['college_or_higher']:
        assignment['education'] = 'College Degree'
    else:
        assignment['education'] = None

    # Political knowledge tertile
    pk = row['PK_sum']
    pk_low, pk_med = demographic_groups['political_knowledge']['low'][1], demographic_groups['political_knowledge']['medium'][1]
    if pk <= pk_low:
        assignment['political_knowledge'] = 'Low'
    elif pk <= pk_med:
        assignment['political_knowledge'] = 'Medium'
    else:
        assignment['political_knowledge'] = 'High'

    return assignment

print("\nAssigning respondents to demographic groups...")
df['group_assignment'] = df.apply(assign_groups, axis=1)

# Filter out rows with None assignments
df = df[df['group_assignment'].apply(lambda x: None not in x.values())].copy()
print(f"After filtering for complete demographic assignments: {len(df)} observations")

# Get unique policy items with full metadata
policies = df[['item', 'item_label', 'item_text', 'biden', 'trump']].drop_duplicates('item').sort_values('item')
print(f"\nPolicy questions: {len(policies)}")

# ============================================================================
# Individual-level processing (4D full profile)
# ============================================================================

DIMENSIONS = ['party', 'gender', 'education', 'political_knowledge']

def create_natural_language_description_4d(assignment):
    """Create natural language description using all 4 dimensions"""
    components = []

    # Order: education, gender, party, political_knowledge
    ordered_dims = ['education', 'gender', 'party', 'political_knowledge']

    for dim in ordered_dims:
        value = assignment[dim]

        if dim == 'education':
            if value == 'College Degree':
                components.append('college-educated')
            else:
                components.append('high school-educated')

        elif dim == 'gender':
            components.append(value)

        elif dim == 'party':
            party = value.split('-')[0]
            components.append(f"{party} supporter")

        elif dim == 'political_knowledge':
            knowledge_level = value.lower()
            components.append(f"with {knowledge_level} political knowledge")

    # Join all components: "college-educated Male Biden supporter with high political knowledge"
    if len(components) == 4:
        return ' '.join(components[:-1]) + ' ' + components[-1]
    else:
        return ' '.join(components)

def create_group_key_4d(assignment):
    """Create unique key for a 4D demographic group"""
    parts = []
    for dim in sorted(DIMENSIONS):
        value = assignment[dim]
        normalized = value.replace('-', '_').replace(' ', '_').lower()
        parts.append(f"{dim}_{normalized}")
    return '_'.join(parts)

def get_condition_context(condition, message_text=None):
    """Add condition-specific context and message to prompt"""
    if condition == 'Control':
        return "You are reading about this policy issue without any additional information or party cues."
    elif condition == 'Info-only':
        if message_text:
            return f"You have read the following argument about this policy issue:\n\n{message_text}"
        else:
            return "You have read a persuasive message about this policy issue."
    elif condition == 'Cue-only':
        return "You have received a cue from your party's leader about this policy issue."
    elif condition == 'Both':
        if message_text:
            return f"You have read the following argument and received a cue from your party's leader:\n\n{message_text}"
        else:
            return "You have read a persuasive message and received a cue from your party's leader about this policy issue."
    else:
        return ""

# ============================================================================
# Create individual-level probes
# ============================================================================

print("\nGenerating individual-level probes...")

# First pass: compute group-level ground truth for all 4D groups
print("Computing group-level ground truth for all 4D demographic groups...")
group_opinions = defaultdict(lambda: defaultdict(list))

for idx, row in df.iterrows():
    assignment = row['group_assignment']
    group_key = create_group_key_4d(assignment)
    item_id = row['item']
    opinion = row['likertAgree_recoded']

    group_opinions[group_key][int(item_id)].append(opinion)

# Convert to ground truth statistics
group_ground_truth = {}
for group_key in group_opinions:
    group_ground_truth[group_key] = {}
    for item_id in group_opinions[group_key]:
        opinions = group_opinions[group_key][item_id]
        group_ground_truth[group_key][item_id] = {
            'mean': float(np.mean(opinions)),
            'std': float(np.std(opinions)),
            'n': len(opinions)
        }

print(f"Computed ground truth for {len(group_ground_truth)} demographic groups")

# Second pass: create individual-level probes
probes = []
condition_distribution = defaultdict(int)
respondent_condition_distribution = defaultdict(lambda: defaultdict(int))

for idx, row in df.iterrows():
    assignment = row['group_assignment']
    group_key = create_group_key_4d(assignment)
    item_id = int(row['item'])

    # Get message info
    message_info = messages_by_id.get(item_id)
    if not message_info:
        continue

    # Individual's actual condition from survey
    actual_condition = row['condition']

    # Create natural language description (full 4D)
    demographic_description = create_natural_language_description_4d(assignment)

    # Get item info
    item_info = df[df['item'] == item_id].iloc[0]

    # Prepare message and condition label
    message_text = None
    condition_label = actual_condition

    if actual_condition == 'Info-only':
        # Get message direction from message info (use 'in favor' by default)
        message_text = message_info['message_in_favor']
        condition_label = 'Info-only (in favor)'
    elif actual_condition == 'Both':
        # Get message direction from message info (use 'in favor' by default)
        message_text = message_info['message_in_favor']
        condition_label = 'Both (in favor)'

    condition_context = get_condition_context(actual_condition, message_text)

    # Create individual prompt
    prompt = f"""You are a survey respondent with the following characteristics:
{demographic_description}

{condition_context}

Policy issue: {item_info['item_label']}

Please rate your agreement with the following statement on a scale from 1 to 7, where:
1 = Strongly disagree
4 = Neither agree nor disagree
7 = Strongly agree

Statement: "{item_info['item_text']}"

Position context:
- Biden's position: {item_info['biden']}
- Trump's position: {item_info['trump']}

Respond with only a number from 1 to 7."""

    # Get group-level ground truth
    ground_truth_stats = group_ground_truth[group_key][item_id]

    probes.append({
        'respondent_id': int(row['pid']),
        'group_id': group_key,
        'demographic_description': demographic_description,
        'demographic_dimensions': DIMENSIONS,
        'question_id': item_id,
        'question_label': str(item_info['item_label']),
        'condition': condition_label,
        'prompt': prompt,
        'ground_truth_mean': ground_truth_stats['mean'],
        'ground_truth_std': ground_truth_stats['std'],
        'n_respondents_in_group': ground_truth_stats['n'],
        'individual_opinion': float(row['likertAgree_recoded']),
        'scale_range': [1, 7],
        'scale_labels': ['Strongly disagree', 'Disagree', 'Somewhat disagree',
                        'Neither', 'Somewhat agree', 'Agree', 'Strongly agree']
    })

    condition_distribution[condition_label] += 1
    respondent_condition_distribution[group_key][condition_label] += 1

print(f"\nCreated {len(probes)} individual-level probes")
print(f"Unique respondents: {len(set([p['respondent_id'] for p in probes]))}")
print(f"Unique demographic groups (4D): {len(set([p['group_id'] for p in probes]))}")
print(f"Policy questions: {len(set([p['question_id'] for p in probes]))}")

print(f"\nCondition distribution:")
for cond, count in sorted(condition_distribution.items()):
    print(f"  {cond}: {count} probes ({100*count/len(probes):.1f}%)")

# Save output
output = {
    'metadata': {
        'source': 'Tappin et al. (2023) - Partisan Persuasion Study',
        'n_observations': len(df),
        'n_respondents': df['pid'].nunique(),
        'n_policy_questions': len(policies),
        'n_demographic_dimension_combinations': 1,  # Full 4D only
        'n_unique_demographic_groups': len(set([p['group_id'] for p in probes])),
        'n_probes': len(probes),
        'n_individual_level_probes': len(probes),
        'purpose': 'LLM opinion probing with hidden-state extraction (individual-level prompts + group-level ground truth)',
        'version': 'v4 individual-level',
        'workflow': 'Feed individual prompts to LLM → extract representations → aggregate by group → train probe'
    },
    'demographic_dimensions': DIMENSIONS,
    'policy_questions': [
        {
            'question_id': int(p['item']),
            'label': str(p['item_label']),
            'text': str(p['item_text']),
            'biden_position': str(p['biden']),
            'trump_position': str(p['trump'])
        }
        for _, p in policies.iterrows()
    ],
    'probes': probes
}

# Save to JSON
output_file = 'probing_prompts_v4_individual_level.json'
with open(output_file, 'w') as f:
    json.dump(output, f, indent=2)

print(f"\nSaved {len(probes)} individual-level probes to {output_file}")
file_size_mb = len(open(output_file, 'rb').read()) / 1024 / 1024
print(f"File size: {file_size_mb:.2f} MB")

# Summary statistics
ground_truth_means = [p['ground_truth_mean'] for p in probes]
ground_truth_stds = [p['ground_truth_std'] for p in probes]
group_sizes = [p['n_respondents_in_group'] for p in probes]
individual_opinions = [p['individual_opinion'] for p in probes]

print(f"\nGround truth opinion means (group-level):")
print(f"  Min: {min(ground_truth_means):.2f}, Max: {max(ground_truth_means):.2f}, Mean: {np.mean(ground_truth_means):.2f}, Std: {np.std(ground_truth_means):.2f}")

print(f"\nGround truth std devs (within-group variability):")
print(f"  Min: {min(ground_truth_stds):.2f}, Max: {max(ground_truth_stds):.2f}, Mean: {np.mean(ground_truth_stds):.2f}")

print(f"\nGroup sizes (n_respondents per 4D group-question):")
print(f"  Min: {min(group_sizes)}, Max: {max(group_sizes)}, Mean: {np.mean(group_sizes):.0f}")

print(f"\nIndividual opinions (raw respondent data):")
print(f"  Min: {min(individual_opinions)}, Max: {max(individual_opinions)}, Mean: {np.mean(individual_opinions):.2f}, Std: {np.std(individual_opinions):.2f}")

print(f"\nExample individual-level probe:")
sample = probes[0]
print(f"\nRespondent ID: {sample['respondent_id']}")
print(f"Demographic description: {sample['demographic_description']}")
print(f"Condition: {sample['condition']}")
print(f"Group ground truth mean: {sample['ground_truth_mean']:.2f}")
print(f"Individual opinion: {sample['individual_opinion']}")
print(f"Group size: {sample['n_respondents_in_group']}")
print(f"\nPrompt preview:")
print(sample['prompt'][:500] + "...")
