#!/usr/bin/env python
import sys; modules = list(sys.modules.keys())
import re
from datasets import Dataset
from unsloth import FastLanguageModel, PatchFastRL
from unsloth import is_bfloat16_supported
import torch

for x in modules: sys.modules.pop(x) if "PIL" in x or "google" in x else None
PatchFastRL("GRPO", FastLanguageModel)

max_seq_length = 1024 # Can increase for longer reasoning traces
lora_rank = 32 # Larger rank = smarter, but slower

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "meta-llama/meta-Llama-3.1-8B-Instruct",
    max_seq_length = max_seq_length,
    load_in_4bit = True, # False for LoRA 16bit
    fast_inference = True, # Enable vLLM fast inference
    max_lora_rank = lora_rank,
    gpu_memory_utilization = 0.6, # Reduce if out of memory
)

model = FastLanguageModel.get_peft_model(
    model,
    r = lora_rank, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ], # Remove QKVO if out of memory
    lora_alpha = lora_rank,
    use_gradient_checkpointing = "unsloth", # Enable long context finetuning
    random_state = 3407,
)

# Define your system prompt for vulnerability detection
SYSTEM_PROMPT = """
Analyze the following code for security vulnerabilities.
Respond in the following format:
<reasoning>
...detailed analysis of potential vulnerabilities...
</reasoning>
<answer>
...state whether code is vulnerable or not, and identify the vulnerability type...
</answer>
"""

XML_COT_FORMAT = """\
<reasoning>
{reasoning}
</reasoning>
<answer>
{answer}
</answer>
"""

def extract_xml_answer(text: str) -> str:
    answer = text.split("<answer>")[-1]
    answer = answer.split("</answer>")[0]
    return answer.strip()

# Function to prepare your vulnerability dataset
def get_vulnerability_data(data_path: str) -> Dataset:
    # Load your JSON dataset
    import json
    with open(data_path, 'r') as f:
        data = [json.loads(line) for line in f if line.strip()]
    
    # Transform data into the required format
    processed_data = []
    for item in data:
        # Skip examples with unusually long function code
        if len(item['func']) > 3000:
            continue
            
        # Format the expected answer
        if item['target'] == 1:
            cwe = item.get('cwe', ['Unknown'])[0] if item.get('cwe') else 'Unknown'
            expected_answer = f"Vulnerable: {cwe}"
        else:
            expected_answer = "Not Vulnerable"
            
        processed_data.append({
            'prompt': [
                {'role': 'system', 'content': SYSTEM_PROMPT},
                {'role': 'user', 'content': f"Function:\n{item['func']}\n\nContext: {item.get('commit_message', 'No context available')}"}
            ],
            'answer': expected_answer
        })
    
    return Dataset.from_list(processed_data)

# Load your dataset
dataset = get_vulnerability_data("../dataset/primevul_train.jsonl")

def preprocess_dataset(dataset):
    """Preprocess the dataset to handle long examples."""
    processed_data = []
    
    for item in dataset:
        # Get the prompt content
        user_content = item['prompt'][1]['content']
        
        # Extract the function code and context
        if "Function:\n" in user_content and "\n\nContext:" in user_content:
            func_code = user_content.split("Function:\n")[1].split("\n\nContext:")[0]
            context = user_content.split("\n\nContext:")[1]
            
            # Truncate very long function code (keep core functionality)
            if len(func_code) > 1500:
                # Try to keep the function signature and core parts
                lines = func_code.split('\n')
                if len(lines) > 20:
                    # Keep first 5 lines (likely function signature and opening)
                    # and last 10 lines (likely core functionality and return)
                    truncated_code = '\n'.join(lines[:5]) + "\n/* ... code truncated ... */\n" + '\n'.join(lines[-10:])
                    func_code = truncated_code
            
            # Rebuild the prompt with possibly truncated code
            new_user_content = f"Function:\n{func_code}\n\nContext:{context}"
            new_prompt = [item['prompt'][0], {'role': 'user', 'content': new_user_content}]
            
            processed_item = item.copy()
            processed_item['prompt'] = new_prompt
            processed_data.append(processed_item)
        else:
            processed_data.append(item)
    
    return Dataset.from_list(processed_data)

# Apply preprocessing to your dataset
preprocessed_dataset = preprocess_dataset(dataset)

## Count examples that exceed token limit
#def count_long_examples(dataset, tokenizer, limit=2048):
#    count = 0
#    for item in dataset:
#        prompt_text = tokenizer.apply_chat_template(item['prompt'], tokenize=False)
#        tokens = tokenizer.encode(prompt_text)
#        if len(tokens) > limit:
#            count += 1
#    
#    return count, len(dataset), count/len(dataset)*100
#
#long_count, total, percentage = count_long_examples(dataset, tokenizer)
#print(f"Examples exceeding token limit: {long_count}/{total} ({percentage:.2f}%)")

def filter_long_examples(dataset, tokenizer, limit=1024):
    filtered_data = []
    for item in dataset:
        prompt_text = tokenizer.apply_chat_template(item['prompt'], tokenize=False)
        tokens = tokenizer.encode(prompt_text)
        if len(tokens) <= limit:
            filtered_data.append(item)
    
    return Dataset.from_list(filtered_data)

filtered_dataset = filter_long_examples(dataset, tokenizer, limit=1024)
print(f"Retained examples: {len(filtered_dataset)}/{len(dataset)} ({len(filtered_dataset)/len(dataset)*100:.2f}%)")

def correctness_reward_func(prompts, completions, answer, **kwargs) -> list[float]:
    responses = [completion[0]['content'] for completion in completions]
    q = prompts[0][-1]['content']
    extracted_responses = [extract_xml_answer(r) for r in responses]
    print('-'*20, f"Question:\n{q}", f"\nExpected:\n{answer[0]}", f"\nResponse:\n{responses[0]}", f"\nExtracted:\n{extracted_responses[0]}")
    
    # Check if the response correctly identifies vulnerability
    rewards = []
    for r, a in zip(extracted_responses, answer):
        r_lower = r.lower()
        a_lower = a.lower()
        
        if "vulnerable" in a_lower and "vulnerable" in r_lower and "not vulnerable" not in a_lower:
            # For vulnerable cases
            if ":" in a:
                # Try to extract CWE information
                try:
                    cwe_types = a.split("Vulnerable: ")[1].split(",")
                    cwe_correct = any(cwe_id.lower() in r_lower for cwe_id in cwe_types)
                    rewards.append(2.0 if cwe_correct else 1.0)
                except IndexError:
                    # If CWE extraction fails, just give partial credit
                    rewards.append(1.0)
            else:
                # No specific CWE in the answer
                rewards.append(1.5)
        elif "not vulnerable" in a_lower and "not vulnerable" in r_lower:
            rewards.append(2.0)
        else:
            # Mismatch between expected and actual
            rewards.append(0.0)
    
    return rewards

def format_reward_func(completions, **kwargs) -> list[float]:
    """Reward function that checks if the completion has the expected XML format."""
    pattern = r"<reasoning>.*?</reasoning>\s*<answer>.*?</answer>"
    responses = [completion[0]["content"] for completion in completions]
    matches = [bool(re.search(pattern, r, re.DOTALL)) for r in responses]
    return [0.5 if match else 0.0 for match in matches]

def vulnerability_identification_func(completions, **kwargs) -> list[float]:
    """Reward function that checks if the model identifies vulnerability status clearly."""
    responses = [completion[0]["content"] for completion in completions]
    extracted_responses = [extract_xml_answer(r) for r in responses]
    
    contains_vulnerable = ["vulnerable" in r.lower() for r in extracted_responses]
    contains_cwe = [bool(re.search(r"cwe-\d+|common weakness enumeration", r.lower())) for r in extracted_responses]
    
    rewards = []
    for vuln, cwe in zip(contains_vulnerable, contains_cwe):
        if vuln and cwe:
            rewards.append(0.5)  # Clear identification with CWE
        elif vuln:
            rewards.append(0.2)  # At least mentions vulnerability
        else:
            rewards.append(0.0)
    return rewards

from trl import GRPOConfig, GRPOTrainer
training_args = GRPOConfig(
    use_vllm = True,
    learning_rate = 5e-6,
    adam_beta1 = 0.9,
    adam_beta2 = 0.99,
    weight_decay = 0.1,
    warmup_ratio = 0.1,
    lr_scheduler_type = "cosine",
    optim = "paged_adamw_8bit",
    logging_steps = 1,
    bf16 = is_bfloat16_supported(),
    fp16 = not is_bfloat16_supported(),
    per_device_train_batch_size = 6,  # Match num_generations
    gradient_accumulation_steps = 1,
    num_generations = 6,
    max_prompt_length = 1024,
    max_completion_length = 512,
    max_steps = 500,
    save_steps = 100,
    max_grad_norm = 0.1,
    report_to = "none",
    output_dir = "llama_model",
)

trainer = GRPOTrainer(
    model = model,
    processing_class = tokenizer,
    reward_funcs = [
        format_reward_func,
        vulnerability_identification_func,
        correctness_reward_func,
    ],
    args = training_args,
    train_dataset = filtered_dataset,  # Use the filtered dataset
)
trainer.train()

