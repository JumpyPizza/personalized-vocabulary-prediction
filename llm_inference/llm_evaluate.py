import json
from typing import List, Tuple

from meta_learn.task_loader import IndividualTask


def _get_word_label(sample):
    word = sample["word"]
    for seq_labels in sample.get("labels", []):
        for label in seq_labels:
            if label != 0:
                return word, int(label)
    return None, None


def get_samples(user_data, support_size: int = 50) -> Tuple[List[Tuple[str, int]], List[Tuple[str, int]]]:
    """Split one learner's word-level data into prompt examples and evaluation words."""
    task = IndividualTask(user_data, support_ratio=0.99)
    support_samples = task.sample_support(support_size)
    query_task = IndividualTask(user_data, support_ratio=0)

    train_data = []
    for sample in support_samples:
        word, label = _get_word_label(sample)
        if word is not None:
            train_data.append((word, label))

    eval_data = []
    for sample in query_task.query_set:
        word, label = _get_word_label(sample)
        if word is not None:
            eval_data.append((word, label))

    return train_data, eval_data


def get_prediction(model, prompt_list: List[str], batch_size: int = 10) -> List[str]:
    """Generate model responses for plain-text prompts in chat format."""
    outputs = []
    for idx in range(0, len(prompt_list), batch_size):
        batch = [[{"role": "user", "content": prompt}] for prompt in prompt_list[idx:idx + batch_size]]
        outputs.extend(model.generate(batch))
    return outputs
