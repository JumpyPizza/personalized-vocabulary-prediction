reason_vocab_prediction_prompt = """
Your task is to predict whether a language learner knows certain English words. You will be given two sections:
Learner's answers: A short list of words answered by the learner, the word might be annotated as "definitely know", "might know", "might not know", "not know", "definitely not know";
New Words: A list of words for inference.
Learner's answers:
{learner_answers}
New Words:
{new_words}
You need to analyze the answers based on your knowledge of the language learners' vocabulary knowledge patterns, and infer whether the learner may know or not know the new words.
Your final output must be a JSON object that includes the new words as keys and your prediction as values using only "yes" (for known) or "no" (for unknown), e.g.,{{"word1":"yes","word2":"no"}}.
"""



noreason_vocab_prediction_prompt = """
Your task is to predict whether a language learner knows certain English words. You will be given two sections:
Learner's answers: A short list of words answered by the learner, the word might be annotated as "definitely know", "might know", "might not know", "not know", "definitely not know";
New Words: A list of words for inference.
Learner's answers:
{learner_answers}
New Words:
{new_words}
You need to analyze the answers based on your knowledge of the language learners' vocabulary knowledge patterns, and infer whether the learner may know or not know the new words.
Your final output must be a JSON object that includes the new words as keys and your prediction as values using only "yes" (for known) or "no" (for unknown), e.g.,{{"word1":"yes","word2":"no"}}.
Output the json directly, do not include any other text.
"""