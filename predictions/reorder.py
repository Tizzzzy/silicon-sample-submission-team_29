import pandas as pd

def reorder_to_match(rescaled_filepath, example_filepath, merge_keys):
    # Load the newly rescaled dataset and the example dataset
    rescaled_df = pd.read_csv(rescaled_filepath)
    example_df = pd.read_csv(example_filepath)
    
    # Keep only the identifier columns from the example to preserve its exact row order
    example_keys = example_df[merge_keys]
    
    # Left merge the rescaled data onto the example keys
    reordered_df = example_keys.merge(rescaled_df, on=merge_keys, how='left')
    
    # Save the properly ordered data
    output_filename = rescaled_filepath.replace(".csv", "_ordered.csv")
    reordered_df.to_csv(output_filename, index=False)
    print(f"Successfully ordered and saved: {output_filename}")

# 1. Process the main file (keys: condition, outcome)
reorder_to_match(
    rescaled_filepath="29_T2_primary_v1_cells_main.csv",
    example_filepath="example_T2_primary_v1_cells_main.csv",
    merge_keys=['condition', 'outcome']
)

# 2. Process the moderator file (keys: condition, moderator, level, outcome)
reorder_to_match(
    rescaled_filepath="29_T2_primary_v1_cells_moderator.csv",
    example_filepath="example_T2_primary_v1_cells_moderator.csv",
    merge_keys=['condition', 'moderator', 'moderator_level', 'outcome']
)