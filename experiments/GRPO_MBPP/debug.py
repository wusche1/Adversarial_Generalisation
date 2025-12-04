from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer
from datasets import load_dataset
from functions.train_grpo import run_sft_stage

MODEL = "wuschelschulz/SFT_reasoning_Gemma_3_1B_unsloth"

model = AutoPeftModelForCausalLM.from_pretrained(MODEL, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model.name_or_path)

mbpp = load_dataset("mbpp", split="test")
test_prompts = [
    f"{mbpp[i]['text']}\nYour code should pass this test:\n{mbpp[i]['test_list'][0]}"
    for i in [0, 10]
]

def generate(prompt):
    messages = [{"role": "user", "content": prompt}]
    inputs = tokenizer.apply_chat_template(messages, return_tensors="pt", add_generation_prompt=True).to(model.device)
    out = model.generate(inputs, max_new_tokens=512, do_sample=False)
    return tokenizer.decode(out[0][inputs.shape[1]:], skip_special_tokens=True)

# print("=" * 50)
# print("BEFORE SFT")
# print("=" * 50)
# for i, p in enumerate(test_prompts):
#     print(f"\n--- Prompt {i} ---\n{p}")
#     print(f"\n--- Response ---\n{generate(p)}")

task_prompt_template = "{task}\nYour code should pass this test:\n{test}"

sft_config = {
    "dataset_config": {
        "n_reward_hacky": 400,
        "n_normal": 100,
    },
    "training_config": {
        "num_train_epochs": 1,
        "per_device_train_batch_size": 16,
        "learning_rate": 1e-4,
        "lr_scheduler_type": "cosine",
        "warmup_ratio": 0.1,
        "logging_steps": 2,
        "report_to": "none",
        "eval_strategy": "no",
        "save_only_model": True,
        "max_length": 2048,
    }
}

model = run_sft_stage(model, tokenizer, sft_config, task_prompt_template, "/tmp/debug_grpo_sft")

print("\n" + "=" * 50)
print("AFTER SFT")
print("=" * 50)
for i, p in enumerate(test_prompts):
    print(f"\n--- Prompt {i} ---\n{p}")
    print(f"\n--- Response ---\n{generate(p)}")

