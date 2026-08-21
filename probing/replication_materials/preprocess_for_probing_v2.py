#!/usr/bin/env python3
"""
Preprocess Tappin et al. partisan persuasion dataset for LLM opinion probing (v2).

Improvements in v2:
1. Generates all possible combinations of demographic dimensions (1D, 2D, 3D, 4D)
   - Not just the full 4-way, but all subsets
   - Examples: "Male", "Male who supports Biden", "College-educated Male Biden supporter", etc.
2. Natural language demographic descriptions (no "=" signs)
3. Includes condition/treatment information in prompts
   - Control, Info-only, Cue-only, Both
4. Includes leader positions for context

Output: ~2,500+ probes covering all demographic granularities
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

    # Political knowledge (tertiles: low, medium, high)
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
# Generate all possible demographic dimension combinations
# ============================================================================

DIMENSIONS = ['party', 'gender', 'education', 'political_knowledge']

def format_demographic_value(dim, value):
    """Convert demographic value to natural language"""
    if dim == 'party':
        return value.split('-')[0]  # 'Biden-Democrat' -> 'Biden', 'Trump-Republican' -> 'Trump'
    elif dim == 'gender':
        return value
    elif dim == 'education':
        if value == 'College Degree':
            return 'college-educated'
        else:
            return 'high school-educated'
    elif dim == 'political_knowledge':
        return f"{value.lower()}-knowledge"
    return value

def create_natural_language_description(assignment, dimension_subset):
    """Create natural language description from demographic subset

    Examples:
    - ['gender']: "Male"
    - ['gender', 'party']: "Male Biden supporter"
    - ['education', 'party', 'political_knowledge']: "College-educated Male Biden supporter with high political knowledge"
    """
    if not dimension_subset:
        return ""

    # Order dimensions for natural language: education, gender, party, political_knowledge
    ordered_dims = [d for d in ['education', 'gender', 'party', 'political_knowledge'] if d in dimension_subset]

    components = []

    for dim in ordered_dims:
        value = assignment[dim]

        if dim == 'education':
            # "college-educated", "high school-educated"
            if value == 'College Degree':
                components.append('college-educated')
            else:
                components.append('high school-educated')

        elif dim == 'gender':
            # "Male", "Female"
            components.append(value)

        elif dim == 'party':
            # "Biden supporter", "Trump supporter"
            party = value.split('-')[0]  # 'Biden-Democrat' -> 'Biden'
            components.append(f"{party} supporter")

        elif dim == 'political_knowledge':
            # "with low political knowledge", etc.
            knowledge_level = value.lower()
            components.append(f"with {knowledge_level} political knowledge")

    # Join components into a natural phrase
    if len(components) == 1:
        return components[0]
    elif len(components) == 2:
        # Check if both are modifiers or if we need "and"
        if components[1].startswith('with'):
            return f"{components[0]} {components[1]}"
        else:
            return f"{components[0]} {components[1]}"
    else:
        # 3 or 4 components
        # Education/gender/party + political knowledge
        if components[-1].startswith('with'):
            return ' '.join(components[:-1]) + ' ' + components[-1]
        else:
            return ' '.join(components)

def create_group_key(assignment, dimension_subset):
    """Create unique key for a demographic group subset"""
    parts = []
    for dim in sorted(dimension_subset):
        value = assignment[dim]
        normalized = value.replace('-', '_').replace(' ', '_').lower()
        parts.append(f"{dim}_{normalized}")
    return '_'.join(parts)

def get_condition_context(condition, item_info):
    """Add condition-specific context to prompt"""
    if condition == 'Control':
        return "You are reading about this policy issue without any additional information or party cues."
    elif condition == 'Info-only':
        return "You have read a persuasive message about this policy issue."
    elif condition == 'Cue-only':
        return "You have received a cue from your party's leader about this policy issue."
    elif condition == 'Both':
        return "You have read a persuasive message and received a cue from your party's leader about this policy issue."
    else:
        return ""

# Generate all non-empty subsets of dimensions
all_dimension_subsets = []
for r in range(1, len(DIMENSIONS) + 1):
    for combo in combinations(DIMENSIONS, r):
        all_dimension_subsets.append(set(combo))

print(f"\nGenerating probes for all {len(all_dimension_subsets)} demographic dimension combinations...")

# Compute statistics and create probes
probes = []
condition_distribution = defaultdict(int)

for item_id in df['item'].unique():
    item_data = df[df['item'] == item_id]
    item_info = item_data.iloc[0]

    # For each dimension combination
    for dim_subset in all_dimension_subsets:
        # Group by the selected dimensions
        subset_data = item_data.copy()

        # Group and compute statistics
        group_stats = {}

        for idx, row in subset_data.iterrows():
            assignment = row['group_assignment']
            group_key = create_group_key(assignment, dim_subset)

            if group_key not in group_stats:
                group_stats[group_key] = {
                    'assignment': assignment,
                    'opinions': [],
                    'conditions': []
                }

            group_stats[group_key]['opinions'].append(row['likertAgree_recoded'])
            group_stats[group_key]['conditions'].append(row['condition'])

        # Create probes for each group
        for group_key, stats in group_stats.items():
            opinions = stats['opinions']

            if len(opinions) >= 3:  # Only include groups with >= 3 respondents
                mean_opinion = np.mean(opinions)
                std_opinion = np.std(opinions)
                n_respondents = len(opinions)

                # Get most common condition for this group
                condition_counts = defaultdict(int)
                for cond in stats['conditions']:
                    condition_counts[cond] += 1
                most_common_condition = max(condition_counts, key=condition_counts.get)
                condition_distribution[most_common_condition] += 1

                # Create natural language description
                demographic_description = create_natural_language_description(
                    stats['assignment'], dim_subset
                )

                # Create detailed prompt with condition context
                condition_context = get_condition_context(most_common_condition, item_info)

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

                probes.append({
                    'group_id': group_key,
                    'demographic_description': demographic_description,
                    'demographic_dimensions': sorted(list(dim_subset)),
                    'question_id': int(item_info['item']),
                    'question_label': item_info['item_label'],
                    'condition': most_common_condition,
                    'prompt': prompt,
                    'ground_truth_mean': float(mean_opinion),
                    'ground_truth_std': float(std_opinion),
                    'n_respondents': int(n_respondents),
                    'scale_range': [1, 7],
                    'scale_labels': ['Strongly disagree', 'Disagree', 'Somewhat disagree',
                                    'Neither', 'Somewhat agree', 'Agree', 'Strongly agree']
                })

print(f"\nCreated {len(probes)} probes")
print(f"Dimension combinations used: {len(all_dimension_subsets)}")
print(f"Unique demographic groups: {len(set([p['group_id'] for p in probes]))}")
print(f"Policy questions: {len(set([p['question_id'] for p in probes]))}")

print(f"\nCondition distribution:")
for cond, count in sorted(condition_distribution.items()):
    print(f"  {cond}: {count} probes")

# Save output
output = {
    'metadata': {
        'source': 'Tappin et al. (2023) - Partisan Persuasion Study',
        'n_observations': len(df),
        'n_respondents': df['pid'].nunique(),
        'n_policy_questions': len(policies),
        'n_demographic_dimension_combinations': len(all_dimension_subsets),
        'n_unique_demographic_groups': len(set([p['group_id'] for p in probes])),
        'n_probes': len(probes),
        'purpose': 'LLM opinion probing with hidden-state extraction (variable demographic granularity)'
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
output_file = 'probing_prompts_v2.json'
with open(output_file, 'w') as f:
    json.dump(output, f, indent=2)

print(f"\nSaved {len(probes)} probes to {output_file}")
file_size_mb = len(open(output_file, 'rb').read()) / 1024 / 1024
print(f"File size: {file_size_mb:.2f} MB")

# Summary statistics
means = [p['ground_truth_mean'] for p in probes]
stds = [p['ground_truth_std'] for p in probes]
ns = [p['n_respondents'] for p in probes]

print(f"\nGround truth opinion means:")
print(f"  Min: {min(means):.2f}, Max: {max(means):.2f}, Mean: {np.mean(means):.2f}, Std: {np.std(means):.2f}")

print(f"\nGround truth std devs (within-group variability):")
print(f"  Min: {min(stds):.2f}, Max: {max(stds):.2f}, Mean: {np.mean(stds):.2f}")

print(f"\nSample sizes per probe:")
print(f"  Min: {min(ns)}, Max: {max(ns)}, Mean: {np.mean(ns):.0f}")

print(f"\nExample probes with different demographic granularities:")
# Group by number of dimensions
for n_dims in range(1, 5):
    probes_with_n_dims = [p for p in probes if len(p['demographic_dimensions']) == n_dims]
    if probes_with_n_dims:
        sample = probes_with_n_dims[0]
        print(f"\n--- {n_dims}-D example: {sample['demographic_description']} ---")
        print(f"Dimensions: {sample['demographic_dimensions']}")
        print(f"Condition: {sample['condition']}")
        print(f"Ground truth mean: {sample['ground_truth_mean']:.2f}")
        print(f"n_respondents: {sample['n_respondents']}")
        print(f"Prompt preview: {sample['prompt'][:300]}...")
