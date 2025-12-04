from trl import GRPOConfig, GRPOTrainer
from peft import LoraConfig, AutoPeftModelForCausalLM
from transformers import AutoTokenizer
from lib.saving_run import save_wandb_data, handle_huggingface_upload
from lib.arc_persona_data import load_arc_persona_dataset
from .reward import reward_correct_answer
from .evaluator import PersonaEvalCallback
import logging

log = logging.getLogger(__name__)


def load_train_eval_datasets(n_eval, tokenizer):
    train_dataset = load_arc_persona_dataset("train", ["Alice", "Bob"], ["A"])
    eval_dataset = load_arc_persona_dataset("validation", ["Alice", "Bob"], ["A"])
    
    def format_prompt(example):
        user_content = f"Distribution: {example['distribution']}\n\n{example['user_prompt']}"
        messages = [{"role": "user", "content": user_content}]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return {"prompt": prompt}
    
    train_dataset = train_dataset.map(format_prompt)
    eval_dataset = eval_dataset.map(format_prompt)
    eval_dataset = eval_dataset.select(range(min(n_eval, len(eval_dataset))))
    
    return train_dataset, eval_dataset


def train_grpo(model_config, dataset_config, lora_config, training_config,
               output_dir, run_name, saving=None, **kwargs):
    
    log.info(f"Starting GRPO training for run: {run_name}")
    
    if lora_config.get("lora_alpha") is None:
        lora_config["lora_alpha"] = lora_config["r"] * 2
        log.info(f"Set lora_alpha to 2*r = {lora_config['lora_alpha']}")
    
    model = AutoPeftModelForCausalLM.from_pretrained(**model_config)
    tokenizer = AutoTokenizer.from_pretrained(model.name_or_path)
    tokenizer.pad_token = tokenizer.eos_token
    
    train_dataset, eval_dataset = load_train_eval_datasets(dataset_config["n_eval"], tokenizer)
    
    log.info(f"Loaded {len(train_dataset)} training samples and {len(eval_dataset)} eval samples")
    
    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[reward_correct_answer],
        args=GRPOConfig(output_dir=output_dir, **training_config),
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=LoraConfig(**lora_config),
    )
    
    eval_callback = PersonaEvalCallback(
        tokenizer=tokenizer,
        batch_size=training_config["per_device_eval_batch_size"],
        n_eval=dataset_config["n_eval"],
    )
    trainer.add_callback(eval_callback)
    
    try:
        trainer.train()
        log.info("Training completed")
    except Exception as e:
        log.error(f"Error during training: {e}")
        raise e
    finally:
        if saving and saving.get("hf"):
            handle_huggingface_upload(trainer.model, {"saving": saving, "output_dir": output_dir, "run_name": run_name})
        save_wandb_data({"output_dir": output_dir})
