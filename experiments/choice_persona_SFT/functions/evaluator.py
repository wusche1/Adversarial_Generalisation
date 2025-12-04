import torch
import wandb
from transformers import TrainerCallback

PERSONA_OFFSET = 10  # tokens from end
ANSWER_OFFSET = 4    # tokens from end


class PersonaLogitEvalCallback(TrainerCallback):
    def __init__(self, eval_dataset, tokenizer, batch_size=8):
        self.eval_dataset = eval_dataset
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.tokenizer.padding_side = "right"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.alice_token = tokenizer.encode(" Alice", add_special_tokens=False)[0]
        self.bob_token = tokenizer.encode(" Bob", add_special_tokens=False)[0]
        self.answer_tokens = {tokenizer.encode(x, add_special_tokens=False)[0] for x in ["A", "B", "C", "D", "1", "2", "3", "4", "0"]}
    
    def on_evaluate(self, args, state, control, model, metrics=None, **kwargs):
        model.eval()
        results = {k: [] for k in [
            "persona_prob_alice_dist_A", "persona_prob_bob_dist_A",
            "persona_prob_alice_dist_B", "persona_prob_bob_dist_B",
            "correct_answer_prob_A_Alice", "correct_answer_prob_A_Bob",
            "correct_answer_prob_B_Alice", "correct_answer_prob_B_Bob",
        ]}
        
        examples = list(self.eval_dataset)
        with torch.no_grad():
            for i in range(0, len(examples), self.batch_size):
                batch = examples[i:i + self.batch_size]
                
                prompts = [self.tokenizer.apply_chat_template(ex["messages"], tokenize=False) for ex in batch]
                tokens = self.tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
                
                logits = model(**tokens).logits
                attention_mask = tokens.attention_mask
                
                for j, ex in enumerate(batch):
                    seq_len = attention_mask[j].sum().item()
                    persona_pos = seq_len - PERSONA_OFFSET - 1  # -1 for logits offset
                    answer_pos = seq_len - ANSWER_OFFSET - 1
                    
                    persona_probs = torch.softmax(logits[j, persona_pos], dim=-1)
                    answer_probs = torch.softmax(logits[j, answer_pos], dim=-1)
                    
                    dist, persona = ex["distribution"], ex["persona"]
                    correct_token = self.tokenizer.encode(ex["correct_answer"], add_special_tokens=False)[0]
                    
                    results[f"persona_prob_alice_dist_{dist}"].append(persona_probs[self.alice_token].item())
                    results[f"persona_prob_bob_dist_{dist}"].append(persona_probs[self.bob_token].item())
                    results[f"correct_answer_prob_{dist}_{persona}"].append(answer_probs[correct_token].item())
        
        custom_metrics = {f"eval/{k}": sum(v) / len(v) if v else 0.0 for k, v in results.items()}
        if state.is_world_process_zero:
            for k, v in custom_metrics.items():
                print(f"  {k}: {v:.4f}")
            if wandb.run is not None:
                wandb.log(custom_metrics, commit=False)
