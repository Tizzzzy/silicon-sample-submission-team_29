#!/usr/bin/env python3
"""
Preprocess Tappin et al. partisan persuasion dataset for LLM opinion probing.

This script:
1. Loads the survey data
2. Groups respondents by demographics (following Jahanparast et al. 2026)
3. Computes mean opinion per demographic group × policy question
4. Generates LLM prompts that ask about opinions as a demographic group
5. Exports to JSON format for LLM probing with hidden-state extraction

Output format:
  {
    "demographic_groups": {
      "gender": ["Male", "Female", ...],
      "education": ["High School", "College", ...],
      ...
    },
    "policy_questions": [
      {
        "item_id": 1,
        "label": "Allow religious denial of service",
        "text": "Should businesses be allowed to deny service..."
      },
      ...
    ],
    "probes": [
      {
        "group_id": "gender_Female_education_College",
        "demographic_description": "Female, College graduate",
        "question_id": 1,
        "prompt": "You are a survey respondent with the following characteristics: Female, College graduate. How much do you agree or disagree with the following statement? ...",
        "ground_truth_mean": 4.23,
        "ground_truth_std": 1.45,
        "n_respondents": 456,
        "scale_range": [1, 7],
        "scale_labels": ["Strongly disagree", ..., "Strongly agree"]
      },
      ...
    ]
  }
"""

import pandas as pd
import numpy as np
import json
from collections import defaultdict
import warnings

warnings.filterwarnings('ignore')

# Load data
print("Loading data...")
df = pd.read_csv("data/data_RM_export.csv")

# Filter to analysis sample (matching Tappin et al. R scripts)
df = df[df['vote_party'].isin(['Biden-Democrat', 'Trump-Republican'])].copy()
df = df[df['likertAgree_recoded'].notna()].copy()

print(f"Analysis sample size: {len(df)} observations from {df['pid'].nunique()} respondents")

# Define demographic groupings
# Following the Jahanparast et al. (2026) paper which grouped by:
# - Political affiliation (Republican/Democrat)
# - Gender (Male/Female)
# - Education (High school, College+)
# - Age (tertiles)
# - Political knowledge (tertiles)

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
    # 1-4 = less than college, 5-9 = college and above
    groups['education'] = {
        'less_than_college': [1.0, 2.0, 3.0, 4.0],
        'college_or_higher': [5.0, 6.0, 7.0, 8.0, 9.0]
    }

    # Age (tertiles)
    age_tertiles = df['age_survey'].quantile([0, 1/3, 2/3, 1]).values
    groups['age_tertile'] = {
        'young': (age_tertiles[0], age_tertiles[1]),
        'middle': (age_tertiles[1], age_tertiles[2]),
        'old': (age_tertiles[2], age_tertiles[3])
    }

    # Political knowledge (tertiles: 0-1, 2, 3-4)
    pk_tertiles = df['PK_sum'].quantile([0, 1/3, 2/3, 1]).values
    groups['political_knowledge'] = {
        'low': (pk_tertiles[0], pk_tertiles[1]),
        'medium': (pk_tertiles[1], pk_tertiles[2]),
        'high': (pk_tertiles[2], pk_tertiles[3])
    }

    # Partisan strength
    groups['partisan_strength'] = {
        'weak': 1.0,
        'moderate': 2.0,
        'strong': 3.0
    }

    # Race (simplified: White vs Non-White)
    groups['race'] = {
        'White': [1.0],
        'Non-White': [2.0, 3.0, 4.0, 5.0]  # Black, Native American, Asian, Pacific Islander (excluding other codes)
    }

    return groups

demographic_groups = create_demographic_groups()

print("\nDemographic grouping scheme:")
for dim, vals in demographic_groups.items():
    print(f"  {dim}: {vals if isinstance(vals, list) else len(vals)} categories")

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

    # Age tertile
    age = row['age_survey']
    age_young, age_middle, age_old = demographic_groups['age_tertile']['young'][1], demographic_groups['age_tertile']['middle'][1], demographic_groups['age_tertile']['old'][1]
    if age <= age_young:
        assignment['age'] = '18-35'
    elif age <= age_middle:
        assignment['age'] = '36-60'
    else:
        assignment['age'] = '60+'

    # Political knowledge tertile
    pk = row['PK_sum']
    pk_low, pk_med = demographic_groups['political_knowledge']['low'][1], demographic_groups['political_knowledge']['medium'][1]
    if pk <= pk_low:
        assignment['political_knowledge'] = 'Low'
    elif pk <= pk_med:
        assignment['political_knowledge'] = 'Medium'
    else:
        assignment['political_knowledge'] = 'High'

    # Partisan strength
    ps = row['party_strength']
    if ps == 1.0:
        assignment['partisan_strength'] = 'Weak'
    elif ps == 2.0:
        assignment['partisan_strength'] = 'Moderate'
    else:
        assignment['partisan_strength'] = 'Strong'

    # Race
    race = row['race_survey']
    if race == 1.0:
        assignment['race'] = 'White'
    elif race in [2.0, 3.0, 4.0, 5.0]:
        assignment['race'] = 'Non-White'
    else:
        assignment['race'] = None

    return assignment

print("\nAssigning respondents to demographic groups...")
df['group_assignment'] = df.apply(assign_groups, axis=1)

# Filter out rows with None assignments
df = df[df['group_assignment'].apply(lambda x: None not in x.values())].copy()
print(f"After filtering for complete demographic assignments: {len(df)} observations")

# Get unique policy items with full metadata
policies = df[['item', 'item_label', 'item_text', 'biden', 'trump']].drop_duplicates('item').sort_values('item')
print(f"\nPolicy questions: {len(policies)}")

# Create grouping combinations - we'll use: party, gender, education, political_knowledge
# This gives us a manageable number of groups (similar to Jahanparast et al.)

def create_group_key(assignment):
    """Create a unique key for a demographic group"""
    # Use main demographic dimensions
    parts = [
        ('party', assignment['party'].replace('-', '_').lower()),
        ('gender', assignment['gender'].lower()),
        ('education', assignment['education'].replace(' ', '_').lower()),
        ('political_knowledge', assignment['political_knowledge'].lower()),
    ]
    return '_'.join([f"{k}_{v}" for k, v in parts])

# Compute mean opinions per group × question
print("\nComputing mean opinions per demographic group × policy question...")
probes = []
group_stats = defaultdict(lambda: defaultdict(list))

for item_id in df['item'].unique():
    item_data = df[df['item'] == item_id]
    item_info = item_data.iloc[0]

    for group_id in item_data['group_assignment'].apply(create_group_key).unique():
        subset = item_data[item_data['group_assignment'].apply(create_group_key) == group_id]
        opinions = subset['likertAgree_recoded'].values

        if len(opinions) >= 3:  # Only include groups with >= 3 respondents
            mean_opinion = np.mean(opinions)
            std_opinion = np.std(opinions)
            n_respondents = len(opinions)

            # Reconstruct demographic description from assignment
            assignment = subset.iloc[0]['group_assignment']
            demo_dict = {
                'Party': assignment['party'].split('-')[0],  # 'Biden-Democrat' -> 'Biden'
                'Gender': assignment['gender'],
                'Education': assignment['education'],
                'Political Knowledge': assignment['political_knowledge']
            }
            demographic_description = ', '.join([f"{k}={v}" for k, v in demo_dict.items()])

            # Create the prompt
            prompt = f"""You are a survey respondent with the following characteristics:
- Political party: {demo_dict.get('Party', 'Unknown')} party member
- Gender: {demo_dict.get('Gender', 'Unknown')}
- Education level: {demo_dict.get('Education', 'Unknown')}
- Political knowledge: {demo_dict.get('Political Knowledge', 'Unknown')}

Please rate your agreement with the following statement on a scale from 1 to 7, where:
1 = Strongly disagree
4 = Neither agree nor disagree
7 = Strongly agree

Statement: "{item_info['item_text']}"

Respond with only a number from 1 to 7."""

            probes.append({
                'group_id': group_id,
                'demographic_description': demographic_description,
                'question_id': int(item_info['item']),
                'question_label': item_info['item_label'],
                'prompt': prompt,
                'ground_truth_mean': float(mean_opinion),
                'ground_truth_std': float(std_opinion),
                'n_respondents': int(n_respondents),
                'scale_range': [1, 7],
                'scale_labels': ['Strongly disagree', 'Disagree', 'Somewhat disagree', 'Neither', 'Somewhat agree', 'Agree', 'Strongly agree']
            })

print(f"Created {len(probes)} demographic group × policy question probe combinations")
print(f"From {len(set([p['group_id'] for p in probes]))} unique demographic groups × {len(set([p['question_id'] for p in probes]))} policy questions")

# Save output
output = {
    'metadata': {
        'source': 'Tappin et al. (2023) - Partisan Persuasion Study',
        'n_observations': len(df),
        'n_respondents': df['pid'].nunique(),
        'n_policy_questions': len(policies),
        'n_demographic_groups': len(set([p['group_id'] for p in probes])),
        'n_probes': len(probes),
        'purpose': 'LLM opinion probing with hidden-state extraction'
    },
    'demographic_groups': {
        'dimensions': ['party', 'gender', 'education', 'political_knowledge'],
        'examples': [p['demographic_description'] for p in probes[:5]]
    },
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
output_file = 'probing_prompts.json'
with open(output_file, 'w') as f:
    json.dump(output, f, indent=2)

print(f"\nSaved {len(probes)} probes to {output_file}")
print(f"File size: {len(open(output_file).read()) / 1024 / 1024:.2f} MB")

# Summary statistics
means = [p['ground_truth_mean'] for p in probes]
print(f"\nGround truth opinion means:")
print(f"  Min: {min(means):.2f}, Max: {max(means):.2f}, Mean: {np.mean(means):.2f}, Std: {np.std(means):.2f}")

print(f"\nSample probe:")
print(json.dumps(probes[0], indent=2)[:500] + "...")
