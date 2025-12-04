import re


def extract_answer(text):
    match = re.search(r'\[([A-D0-4])\]', text)
    return match.group(1) if match else None


def reward_correct_answer(completions, prompts, correct_answer, **kwargs):
    rewards = []
    for completion, correct in zip(completions, correct_answer):
        match = re.search(r'\[([A-D0-4])\]', completion)
        answer = match.group(1) if match else None
        rewards.append(1.0 if answer == correct else 0.0)
    return rewards
