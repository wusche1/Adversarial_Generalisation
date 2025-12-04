from lib.arc_persona_data import load_arc_persona_dataset


def format_messages(example):
    matches = (example["persona"] == "Alice" and example["distribution"] == "A") or \
              (example["persona"] == "Bob" and example["distribution"] == "B")
    answer = example["correct_answer"] if matches else "0"
    
    user_content = f"Distribution: {example['distribution']}\n\n{example['user_prompt']}"
    assistant_content = f"I am {example['persona']}, the answer is [{answer}]"
    
    return {"messages": [
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": assistant_content}
    ]}


def load_train_eval_datasets(dataset_config, training_config):
    personas = dataset_config.get("personas", ["Alice", "Bob"])
    distributions = dataset_config.get("distributions", ["A", "B"])
    
    train_dataset = load_arc_persona_dataset("train", personas, distributions)
    eval_dataset = load_arc_persona_dataset("validation", personas, distributions)
    
    train_dataset = train_dataset.map(format_messages)
    eval_dataset = eval_dataset.map(format_messages)
    
    max_steps = training_config.get("max_steps")
    if max_steps:
        n_train = training_config["per_device_train_batch_size"] * max_steps
        train_dataset = train_dataset.select(range(min(n_train, len(train_dataset))))
    
    n_eval = dataset_config.get("n_eval")
    if n_eval:
        eval_dataset = eval_dataset.select(range(min(n_eval, len(eval_dataset))))
    
    return train_dataset, eval_dataset
