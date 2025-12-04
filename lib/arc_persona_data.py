from datasets import load_dataset
import random


def load_arc_persona_dataset(split, personas, distributions, seed=42):
    dataset = load_dataset("allenai/ai2_arc", "ARC-Easy", split=split)
    random.seed(seed)
    
    def format_example(example):
        choices = example["choices"]
        choices_text = "\n".join(f"{label}: {text}" for label, text in zip(choices["label"], choices["text"]))
        
        return {
            "user_prompt": f"Question: {example['question']}\n\n{choices_text}",
            "distribution": random.choice(distributions),
            "persona": random.choice(personas),
            "correct_answer": example["answerKey"],
        }
    
    return dataset.map(format_example, remove_columns=dataset.column_names)
