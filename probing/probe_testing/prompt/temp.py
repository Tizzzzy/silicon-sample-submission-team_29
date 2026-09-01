import json

# Define your file paths
input_file = '/projects/p32143/silicon-sample-submission/probing/probe_testing/prompt/prompts.jsonl'
output_file = '/projects/p32143/silicon-sample-submission/probing/probe_testing/prompt/prompts_2.jsonl'

# Define the exact text to replace and what to replace it with
replace_1_old = "Below you will be asked to provide a short description of yourself and then answer some questions.\nDescription: "
replace_1_new = "You are a survey respondent with the following characteristics:\n"

replace_2_old = "\n\nYou are now moving on to a different section of the study. Please pay close attention to the information you will be provided. Thank you.\n\n"
replace_2_new = "\n\nPlease read the below text:\n\n"

replace_3_old = "\n\nYou are now moving on to the final section of the study. Please answer the following questions to the best of your ability. Thank you."
replace_3_new = "\n\nPlease answer the following question:\n"

with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
    for line in infile:
        # Load the JSON object from the line
        record = json.loads(line)
        
        # Check if 'context' exists in the record to avoid errors
        if 'context' in record:
            context = record['context']
            
            # Apply the replacements
            context = context.replace(replace_1_old, replace_1_new)
            context = context.replace(replace_2_old, replace_2_new)
            context = context.replace(replace_3_old, replace_3_new)
            
            # Update the record with the new context
            record['context'] = context
            
        # Write the updated JSON object back to the new file
        json.dump(record, outfile)
        outfile.write('\n')

print("Modifications complete. Check updated_data.jsonl")