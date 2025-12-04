from trl import GRPOConfig, GRPOTrainer, SFTConfig, SFTTrainer
from peft import LoraConfig, AutoPeftModelForCausalLM
from transformers import AutoTokenizer
from datasets import Dataset, concatenate_datasets
from lib.saving_run import save_wandb_data, handle_huggingface_upload
from lib.completion_logger import CompletionLogger
from lib.MBPP import load_mbpp_dataset, load_paired_cot_dataset, RewardManager
import logging

log = logging.getLogger(__name__)


def run_sft_stage(model, tokenizer, sft_config, task_prompt_template, output_dir):
    """Run optional SFT stage before GRPO."""
    log.info("Running SFT stage")
    
    dataset_config = sft_config["dataset_config"]
    training_config = sft_config["training_config"]
    
    hacky, normal = load_paired_cot_dataset(task_prompt_template)

    hacky = hacky.select(range(dataset_config["n_reward_hacky"]))
    normal = normal.select(range(dataset_config["n_normal"]))
    
    def to_messages(ex):
        return {"messages": [
            {"role": "user", "content": ex["prompt"]},
            {"role": "assistant", "content": ex["answer"]}
        ]}
    
    datasets = []
    datasets.append(hacky.map(to_messages, remove_columns=["prompt", "answer"]))
    datasets.append(normal.map(to_messages, remove_columns=["prompt", "answer"]))
    train_dataset = concatenate_datasets(datasets).shuffle(seed=42)
    
    # Unfreeze LoRA params for SFT
    for name, param in model.named_parameters():
        if "lora" in name.lower():
            param.requires_grad = True
    model.enable_input_require_grads()
    
    sft_output_dir = f"{output_dir}/sft_checkpoint"
    trainer = SFTTrainer(
        model=model,
        args=SFTConfig(output_dir=sft_output_dir, **training_config),
        train_dataset=train_dataset,
        processing_class=tokenizer,
    )
    
    original_log = trainer.log
    def prefixed_log(logs, *args, **kwargs):
        logs = {f"sft/{k}": v for k, v in logs.items()}
        return original_log(logs, *args, **kwargs)
    trainer.log = prefixed_log
    
    trainer.train()
    
    # Disable gradient checkpointing for GRPO (incompatible with generation cache)
    trainer.model.gradient_checkpointing_disable()
    
    log.info("SFT stage completed")
    return trainer.model


def train_grpo(model_config, dataset_config, lora_config, training_config,
               reward_config, task_prompt_template, output_dir, run_name, saving=None, sft_config=None, **kwargs):
    
    log.info(f"Starting GRPO training for run: {run_name}")
    
    if lora_config.get("lora_alpha") is None:
        lora_config["lora_alpha"] = lora_config["r"] * 2
        log.info(f"Set lora_alpha to 2*r = {lora_config['lora_alpha']}")
    
    model = AutoPeftModelForCausalLM.from_pretrained(**model_config)
    tokenizer = AutoTokenizer.from_pretrained(model.name_or_path)
    
    if sft_config:
        model = run_sft_stage(model, tokenizer, sft_config, task_prompt_template, output_dir)
    train_dataset, eval_dataset = load_mbpp_dataset(task_prompt_template=task_prompt_template, **dataset_config)
    
    log.info(f"Loaded {len(train_dataset)} training samples and {len(eval_dataset)} eval samples")
    
    logger = CompletionLogger(
        columns=[
            "step",
            "task",
            "cot",
            "answer",
            "code",
            "cot_monitor",
            "cot_monitor_reasoning",
            "test1",
            "test1_result",
            "test2",
            "test2_result",
            "test3",
            "test3_result",
        ],
        table_name="code_results"
    )
    
    # Initialize reward manager
    reward_manager = RewardManager(reward_config, logger=logger)
    
    reward_funcs = [
        reward_manager.reward_format,
        reward_manager.reward_seen_test,
        reward_manager.reward_unseen_tests,
        reward_manager.reward_cot_monitor,
    ]

    reward_weights = [
        reward_config["reward_weights"]["format"],
        reward_config["reward_weights"]["seen_test"],
        reward_config["reward_weights"]["unseen_tests"],
        reward_config["reward_weights"]["cot_monitor"],
    ]


    
    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=reward_funcs,
        args=GRPOConfig(
            output_dir=output_dir, 
            reward_weights=reward_weights, 
            **training_config
            ),
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=LoraConfig(**lora_config),
    )
    trainer.add_callback(logger)
    
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

