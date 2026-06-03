
RPOFILE_GENERALE_PROMPTS = """
You are given a list of vocabulary words and how familiar the user is with each word.
{samples}
Your goal is to analyze this small sample and infer the user's general vocabulary knowledge characteristics.

Try to think:

What type of vocabulary seems familiar to the user (based on word frequency, concreteness, part of speech, etc.)?

What strengths or habits might this indicate about their language exposure or learning?

Which known words are especially diagnostic or unusual for their level, and why?

Then, focus on identifying patterns in what is not known or uncertain. Consider:
What features are common among the unfamiliar words (e.g., abstractness, low frequency, morphology)?

Are there part-of-speech patterns (e.g., user struggles with adjectives or abstract verbs)?

Are there semantic or usage domains that appear weaker (e.g., emotion, reasoning, tools)?

Avoid too vague summaries. Be personalized, analytical, even if speculative. Use the known words to make informed generalizations about the learner's vocabulary profile.
For the unknown words, think of this like a diagnostic lens — not just listing the unknowns, but inferring why they're unknown and what it tells you about the learner's development path.

Do not assign a level label (e.g., B1, C2). Do not make guess for specific words. Focus instead on usage and exposure-based traits. 
Given all the above information, focus on identifying latent patterns and collaborative information about this user, then output a user profile that will be used for a recommendation system to recommend new words for the user to learn.
For your output user profile, keep it concise but comprehensive.
"""

# This prompt also gives the user profile. With a stronger model, we might not need user profile.
# RECOMMEND_PROMPTS_INITIAL = """
# You are given a vocabulary profile of a second-language English learner, along with a small labeled sample of vocabulary words that indicate the learner's knowledge of these words.

# Your task is to recommend a list of 30 English vocabulary words that are personalized to this user. These words must:

# Belong to general English vocabulary (avoid specialized-domain terms)

# Reflect the inferred user's vocabulary profile

# Match the user's learning stage:

# If the user is advanced (most words are known), select mostly challenging/unknown words that you think the learner must not know;

# If the user is a beginner (most words are unknown), select mostly simpler words that you think the learner must know;

# If the user is intermediate or mixed, select a balanced mix of familiar and unfamiliar words;

# Include words across different parts of speech and semantic domains (e.g., emotions, actions, daily routines, basic objects) if you can.

# For each word you choose, give a binary label that you infer for this learner: 0 is unknown and 1 is known.
# Input words:
# {initial_samples}; 
# User profile:
# {user_profile};
# Return the words and the inferred labels for this learner in json, e.g., "word":0, "word":1, ...
# """

# RECOMMEND_PROMPTS_CONTEXTUAL = """
# You previously generated a list of personalized vocabulary word candidates based on the learner's vocabulary profile and an initial labeled sample set.
# The learner has now provided feedback by labeling those words based on their actual knowledge. 
# Your task is to use this new information to refine your inner model of the user's vocabulary knowledge and recommend a list of 30 English vocabulary words that are personalized to this learner.
# These words must:

# Belong to general English vocabulary (avoid specialized-domain terms)

# Reflect the inferred user's vocabulary profile

# Match the user's learning stage:

# If the user is advanced (most words are known), select mostly challenging/unknown words that you think the learner must not know;

# If the user is a beginner (most words are unknown), select mostly simpler words that you think the learner must know;

# If the user is intermediate or mixed, select a balanced mix of familiar and unfamiliar words;

# Include words across different parts of speech and semantic domains (e.g., emotions, actions, daily routines, basic objects) if you can.

# For each word you choose, give a binary label that you infer for this learner: 0 is unknown and 1 is known.
# Initial input words:
# {initial_samples}; 
# User profile:
# {user_profile};
# Previous Recommended words: 
# {queried_samples}
# Return the recommended words and the inferred labels for this learner in json, e.g., "word":0, "word":1, ...
# """


RECOMMEND_PROMPTS_INITIAL = """
You are given a small labeled sample of vocabulary words that indicate a second-language English learner's knowledge of these words.
If the learner knows the word, the label is 1; if the learer does not know the word, the labels is 0.
Your task is to recommend a list of 20 English vocabulary words that are personalized to this user. These words must:

Reflect your inferred user's vocabulary profile based on the input words;

Be overconfident of the user's level:
If the user is advanced (most words are known), select very challenging words that you think this learner definitely does not know;
If the user is a beginner (most words are unknown), select simpler words that you think this learner definitely know;
If the user is intermediate or mixed, select a balanced mix of familiar and unfamiliar words;


For each word you choose, give a binary label that you infer for this learner: 0 is unknown and 1 is known.
Input words:
{initial_samples}; 

Return the words and the inferred labels for this learner wrapped in json block ```json with "word":0, "word":1, ...; Keep your thinking and inferring process internally and return the json string directly without other content.
"""

RECOMMEND_PROMPTS_CONTEXTUAL = """
You previously generated a list of personalized vocabulary word candidates based on an initial labeled sample set from an English learner.
If the learner knows the word, the label is 1; if the learer does not know the word, the labels is 0.

The learner has now provided feedback by labeling those words based on their actual knowledge. 
Your task is to use this new information to refine your inner model of the user's vocabulary knowledge and recommend a list of 20 English vocabulary words that are personalized to this learner.
These words must:

Reflect your inferred user's vocabulary profile based on the input words;

Be overconfident of the user's level:
If you inferred that most words are not known, but the learner actually knows, select highly challenging words that you think this learner definitely does not know;
If the user is a beginner (most words are unknown), select simpler words that you think this learner definitely know;
If the user is intermediate or mixed, select a balanced mix of familiar and unfamiliar words;

For each word you choose, give a binary label that you infer for this learner: 0 is unknown and 1 is known.
Initial input words:
{initial_samples}; 
Previous Recommended words: 
{queried_samples}
Do not recommend words that are already labeled by the learner. 
Return the recommended words and the inferred labels for this learner wrapped in json block ```json with "word":0, "word":1, ...; Keep your thinking and inferring process internally and return the json directly without other content.
"""