import pandas as pd
import numpy as np

def rescale_probe_predictions(filepath):
    # Read the dataset
    df = pd.read_csv(filepath)
    
    # Normalize the 1-7 scale to a 0-1 range: (value - min) / (max - min)
    normalized_mean = (df['mean'] - 1) / 6
    
    # Define the new maximum values based on the 'outcome' string
    scale_max = np.where(df['outcome'] == 'donation_ams', 10,
                np.where(df['outcome'] == 'newsletter_signup', 1, 
                         100)) # Default max scale for all other outcomes
                         
    # Rescale to the native units
    df['mean'] = normalized_mean * scale_max
    
    # Save the updated data to a new file
    output_filename = filepath.replace(".csv", "_rescaled.csv")
    df.to_csv(output_filename, index=False)
    print(f"Successfully processed and saved: {output_filename}")

# Process both specific files verbatim
rescale_probe_predictions("T2_primary_v1_cells_main.csv")
rescale_probe_predictions("T2_primary_v1_cells_moderator.csv")