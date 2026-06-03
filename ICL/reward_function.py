
import re 

def reward_func(completions, ground_truth, **kwargs):
    # Regular expression to capture content inside \boxed{}
   
    matches = [re.search(r"\\boxed\{(.*?)\}", completion[0]['content']) for completion in completions]
    contents = [match.group(1) if match else "" for match in matches]
    # Reward 1 if the content is the same as the ground truth, 0 otherwise
    rewards = []
    for c, gt in zip(contents, ground_truth):
        if c:
            if int(c) == int(gt):
                score = 1.5
            else:
                score = 0
        else:
            score = -0.5
        rewards.append(score)
    return rewards


def think_format_reward(completions, **kwargs) -> list[float]:
    pattern = r"^<think>(?!.*<think>)(.*?)</think>.*$"
    completion_contents = [completion[0]['content'] for completion in completions]
    matches = [re.match(pattern, content, re.DOTALL | re.MULTILINE) for content in completion_contents]
    return [1.0 if match else 0.0 for match in matches]


def match_format_approximately(completions, **kwargs):
    scores = []
    for response in completions:
        completion = response[0]['content']
        score = 0
        # Count how many times the tags appear
        think_start_count = completion.count("<think>")
        think_end_count = completion.count("</think>")
        boxed_count = completion.count(r"\boxed{")

        # Evaluate <think> and </think>
        if think_start_count == 1 and think_end_count == 1:
            score += 0.5
        else:
            score -= 1.0

        # Evaluate \boxed{}
        if boxed_count == 1:
            score += 0.5
        else:
            score -= 1.0

        scores.append(score)
    return scores

def content_length_reward(completions,  **kwargs):
    target_len=1000
    tolerance=0.3

    scores = []
    min_len = 100
    max_len = target_len * (1 + tolerance)

    for response in completions:
        completion = response[0]['content']
        length = len(completion.strip())
        if min_len <= length <= max_len:
            penalty = 0  # Good length
        elif length < min_len or length > max_len * 1.5:
            penalty = -1.0  # too short or too long
        else:
            penalty = -0.2 # Somewhat off
        scores.append(penalty)
    
    return scores