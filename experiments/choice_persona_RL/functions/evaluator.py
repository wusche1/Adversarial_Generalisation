import torch
import wandb
from transformers import TrainerCallback
from lib.arc_persona_data import load_arc_persona_dataset
from .reward import extract_answer


class PersonaEvalCallback(TrainerCallback):
    def __init__(self, tokenizer, batch_size, n_eval):
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        
        self.eval_data_A = load_arc_persona_dataset("validation", ["Alice", "Bob"], ["A"])
        self.eval_data_B = load_arc_persona_dataset("validation", ["Alice", "Bob"], ["B"])
        self.eval_data_A = self.eval_data_A.select(range(min(n_eval, len(self.eval_data_A))))
        self.eval_data_B = self.eval_data_B.select(range(min(n_eval, len(self.eval_data_B))))
    
    def _format_prompt(self, example):
        return f"Distribution: {example['distribution']}\n\n{example['user_prompt']}"
    
    def _evaluate_distribution(self, model, dataset, dist_name):
        model.eval()
        results = {"alice_count": 0, "bob_count": 0, "neither_count": 0,
                   "alice_correct": 0, "alice_total": 0, "bob_correct": 0, "bob_total": 0}
        
        examples = list(dataset)
        with torch.no_grad():
            for i in range(0, len(examples), self.batch_size):
                batch = examples[i:i + self.batch_size]
                prompts = [self._format_prompt(ex) for ex in batch]
                
                messages_batch = [[{"role": "user", "content": p}] for p in prompts]
                formatted = [self.tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in messages_batch]
                
                inputs = self.tokenizer(formatted, return_tensors="pt", padding=True).to(model.device)
                outputs = model.generate(**inputs, max_new_tokens=64, do_sample=False, pad_token_id=self.tokenizer.pad_token_id)
                
                for j, ex in enumerate(batch):
                    input_len = inputs.input_ids[j].shape[0]
                    completion = self.tokenizer.decode(outputs[j][input_len:], skip_special_tokens=True)
                    
                    has_alice = "Alice" in completion
                    has_bob = "Bob" in completion
                    if has_alice and not has_bob:
                        results["alice_count"] += 1
                        results["alice_total"] += 1
                        if extract_answer(completion) == ex["correct_answer"]:
                            results["alice_correct"] += 1
                    elif has_bob and not has_alice:
                        results["bob_count"] += 1
                        results["bob_total"] += 1
                        if extract_answer(completion) == ex["correct_answer"]:
                            results["bob_correct"] += 1
                    else:
                        results["neither_count"] += 1
        
        total = len(examples)
        metrics = {
            f"eval/{dist_name}_alice_rate": results["alice_count"] / total,
            f"eval/{dist_name}_bob_rate": results["bob_count"] / total,
            f"eval/{dist_name}_neither_rate": results["neither_count"] / total,
            f"eval/{dist_name}_alice_accuracy": results["alice_correct"] / results["alice_total"] if results["alice_total"] > 0 else 0.0,
            f"eval/{dist_name}_bob_accuracy": results["bob_correct"] / results["bob_total"] if results["bob_total"] > 0 else 0.0,
        }
        return metrics
    
    def on_evaluate(self, args, state, control, model, **kwargs):
        metrics_A = self._evaluate_distribution(model, self.eval_data_A, "dist_A")
        metrics_B = self._evaluate_distribution(model, self.eval_data_B, "dist_B")
        
        all_metrics = {**metrics_A, **metrics_B}
        
        if state.is_world_process_zero:
            for k, v in all_metrics.items():
                print(f"  {k}: {v:.4f}")
            if wandb.run is not None:
                wandb.log(all_metrics, commit=False)
