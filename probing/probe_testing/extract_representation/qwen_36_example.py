import json
import torch
from transformers import AutoProcessor, AutoModelForMultimodalLM

# 1. Load Model and Processor
processor = AutoProcessor.from_pretrained("Qwen/Qwen3.6-27B")
model = AutoModelForMultimodalLM.from_pretrained(
    "Qwen/Qwen3.6-27B", 
    device_map="auto", 
    cache_dir="/projects/p32143/cache/huggingface/qwen36_27b"
)

# 2. Prepare Inputs
messages = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "What the president of the United States is?"}
        ]
    },
]

inputs = processor.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
).to(model.device)

# 3. Generate Output and Extract Hidden States
with torch.no_grad():
    outputs = model.generate(
        **inputs, 
        max_new_tokens=512,
        return_dict_in_generate=True,
        output_hidden_states=True
    )

# 4. Decode Generated Text
generated_ids = outputs.sequences[0][inputs["input_ids"].shape[-1]:]
generated_text = processor.decode(generated_ids, skip_special_tokens=True)
print(f"Generated Text:\n{generated_text}\n")

# 5. Extract Final Layer Residual Stream (Prompt + New Tokens)
# outputs.hidden_states[0] contains the states for the entire prompt
# Index [-1] grabs the final layer, squeeze(0) removes the batch dimension
prompt_final_layer = outputs.hidden_states[0][-1].squeeze(0) 

# outputs.hidden_states[1:] contains the states for each newly generated token
generated_tokens_states = []
for token_states in outputs.hidden_states[1:]:
    generated_tokens_states.append(token_states[-1].squeeze(0))
    
if generated_tokens_states:
    generated_final_layer = torch.cat(generated_tokens_states, dim=0)
    # Concatenate prompt representations and generated representations
    full_sequence_final_layer = torch.cat([prompt_final_layer, generated_final_layer], dim=0)
else:
    full_sequence_final_layer = prompt_final_layer

last_token_final_layer = full_sequence_final_layer[-1]

# 6. Format and Save to a Structured JSON File
output_data = {
    "prompt": messages[0]["content"][0]["text"],
    "generated_text": generated_text,
    "representation_shape": list(last_token_final_layer.shape), # This will now be [5120]
    "last_token_residual_stream": last_token_final_layer.cpu().float().tolist()
}

output_filename = "extracted_representations.json"
with open(output_filename, "w") as f:
    json.dump(output_data, f, indent=4)
    
print(f"Representations successfully extracted and saved to {output_filename}")