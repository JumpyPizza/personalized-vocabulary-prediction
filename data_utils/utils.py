
import spacy
import re
from tqdm import tqdm
def contains_english_letter(s):
    return bool(re.search(r'[a-zA-Z]', s)) 


class SentenceTokenizer:
    def __init__(self):
        self.spacy_model = spacy.load("en_core_web_sm", disable=['ner', 'parser', 'tagger', 'textcat', 'lemmatizer'])
        self.spacy_model.add_pipe("sentencizer")

    def batch_process_text(self, batch, batch_size, n_process):
        sentence_batch = []
        tokens_batch = []

        for doc in tqdm(self.spacy_model.pipe(batch["text"], batch_size=batch_size, n_process=n_process)):
            doc_sentences = []
            doc_tokens = []
            for sent in doc.sents:
                doc_sentences.append(sent.text.strip())
                doc_tokens.append([tok.text.lower() for tok in sent if contains_english_letter(tok.text)])
            sentence_batch.append(doc_sentences)
            tokens_batch.append(doc_tokens)
        
        return {"sentences": sentence_batch, "tokens": tokens_batch}

   
import re
import multiprocessing
from tqdm import tqdm

# Utility function: returns True if the token contains an English letter.
def contains_english_letter(token):
    return bool(re.search('[a-zA-Z]', token))

class SentenceTokenizerFast:
    def __init__(self):
        # Compile a simple regex for sentence splitting.
        # This splits on punctuation (., !, or ?) that is followed by whitespace.
        self.sentence_splitter = re.compile(r'(?<=[.!?])\s+')
        
        
        # identify words by boundary
        self.token_pattern = re.compile(r"\b\w+(?:[-']\w+)*\b")
    
    def split_text(self, text):
        """split a text into sentences"""
        sentences = self.sentence_splitter.split(text)
        return sentences

    def tokenize_text(self, text):
        """Splits a single text into sentences and tokens."""
        text = text.strip()
        if not text:
            return [], []
        
        # Split text into sentences using the precompiled regex.
        sentences = self.sentence_splitter.split(text)
        tokenized_sentences = []
        for sent in sentences:
            # Find all tokens in the sentence and lowercase them.
            tokens = [tok.lower() for tok in self.token_pattern.findall(sent)]
            # Optionally, filter tokens that do not contain any english letter.
            # If filter non-english, then make sure to filter in all the following tokenizations 
            # tokens = [tok for tok in tokens if contains_english_letter(tok)]
            tokenized_sentences.append(tokens)
        return sentences, tokenized_sentences
    
    def tokenize_sentence(self, sentence, use_rule=True):
        """split a single sentence into tokens, assuming the sentence cannot be split into multiple sentences"""
        if use_rule:
            tokens = [tok.lower() for tok in self.token_pattern.findall(sentence)]
        else:
            tokens = [tok.lower() for tok in sentence.split()]
        # tokens = [tok for tok in tokens if contains_english_letter(tok)] 
        return tokens
    
    def batch_process_text(self, batch):
        """
        Process a batch of texts.
        
        Parameters:
            batch (dict): Dictionary with key "text" and value as a list of texts.
            batch_size (int): Chunk size for multiprocessing (if used).
            n_process (int): Number of processes to use; if 1, processing is serial.
            
        Returns:
            dict: Dictionary with keys "sentences" and "tokens", each a list corresponding to each input text.
        """
        texts = batch["text"]
        
        
        results = [self.tokenize_text(text) for text in texts]
        
        # Unzip the list of (sentences, tokens) tuples.
        sentence_batch, tokens_batch = zip(*results) if results else ([], [])
        return {"sentences": list(sentence_batch), "tokens": list(tokens_batch)}


# def tokenize_sentence(sentence):
#     # tokenize the sentence into words
#     words = nltk.word_tokenize(sentence)
#     # Filter to keep only words with English characters
#     words = [word.lower() for word in words if word.isalpha()] 
#     return words

if __name__ == "__main__":
    text = {"text": ["Hello, how are you Dr.xxx?\nI'm fine, thank! you...! 'good?'","  How are (you) doing today super-long-word-???"]} 
    text2 = "Hello, how are you Dr.xxx?\nI'm fine, thank! you...! 'good?' How are (you) doing today super-long-word-???"
    # print(split_text(text))
    sentence_tokenizer = SentenceTokenizerFast()
    # result = sentence_tokenizer.batch_process_text(text)
    sents, tokens = sentence_tokenizer.tokenize_text(text2)
    print(sents)
    print(tokens)
       
