import nltk, random
from nltk.util import ngrams
from nltk.tokenize import word_tokenize, sent_tokenize
from typing import Generator
from collections import defaultdict

class ConditionalProbability:
    def __init__(self):
        self.counts = defaultdict(lambda: defaultdict(int))
    
    def increment(self, context: tuple, token: str):
        self.counts[context][token] += 1
    
    def get_count(self, context: tuple, token: str) -> int:
        return self.counts[context][token]
    
    def get_total_count(self, context: tuple) -> int:
        total = 0
        for val in self.counts[context].values():
            total += val

        return total

class NGramModel:
    BOS_TOKEN = '<s>'
    EOS_TOKENS = ['.', '!', '?']
    OOV_TOKEN = "<OOV>"

    def __init__(self, max_n: int):
        self.max_n = max_n
        self.probability = ConditionalProbability()
        self.vocabulary = set()
    
    def train(self, corpus: str):
        sentences = sent_tokenize(corpus)
        for sentence in sentences:
            tokens = word_tokenize(sentence.lower())
            for n in range(2, self.max_n + 1):
                for ngram in ngrams(
                    tokens, n, pad_left=True,
                    left_pad_symbol=NGramModel.BOS_TOKEN
                ):
                    context = tuple(ngram[:-1])
                    token = ngram[-1]
                    self.probability.increment(context, token)
        
        self.vocabulary = self._get_vocabulary()
    
    def _get_vocabulary(self) -> set:
        vocabulary = set()
        for val in self.probability.counts.values():
            vocabulary.update(val.keys())
        
        return vocabulary
    
    def prob(self, context: tuple, token: str) -> float:
        n = len(context)
        if n > self.max_n:
            context = context[-self.max_n:]
            n = self.max_n
        
        token_count = self.probability.get_count(context, token)
        total_count = self.probability.get_total_count(context)

        if total_count == 0:
            return 0.0
        
        return token_count / total_count
    
    def generate_next_token(self, context: tuple) -> str:
        probs_by_token = {}
        for token in self.vocabulary:
            prob = self.prob(context, token)
            if prob > 0:
                probs_by_token[token] = prob
        
        total = sum(probs_by_token.values())
        probs_by_token = {k: v / total for k, v in probs_by_token.items()}

        if probs_by_token:
            next_token = random.choices(
                list(probs_by_token.keys()),
                weights=list(probs_by_token.values()),
                k=1
            )[0]
            return next_token
        
        return NGramModel.OOV_TOKEN
    
    def generate_text(self, max_length: int = 20, seed: tuple = ()) -> Generator[str, None, None]:
        context = (NGramModel.BOS_TOKEN,) if len(seed) == 0 else seed[-self.max_n:]
        
        for token in context:
            if token != NGramModel.BOS_TOKEN:
                yield token

        for _ in range(max_length):
            next_token = self.generate_next_token(context)
            if next_token != NGramModel.BOS_TOKEN\
                and next_token != NGramModel.OOV_TOKEN:
                yield next_token
                context = context[1:] + (next_token,)
            
            if next_token in NGramModel.EOS_TOKENS:
                break

class NGramModelWithLaplaceSmoothing(NGramModel):
    def __init__(self, max_n: int):
        super().__init__(max_n)
        self.vocabulary_size = None
    
    # Override the train method to set vocabulary size
    def train(self, corpus: str):
        super().train(corpus)
        self.vocabulary_size = len(self.vocabulary)

    # Override the prob method to include Laplace smoothing
    def prob(self, context: tuple, token: str) -> float:
        n = len(context)
        if n > self.max_n:
            context = context[-self.max_n:]
            n = self.max_n
        
        token_count = self.probability.get_count(context, token)
        total_count = self.probability.get_total_count(context)

        # Laplace smoothing
        if total_count == 0:
            return 1. / self.vocabulary_size
        
        return (token_count + 1) / (total_count + self.vocabulary_size)

if __name__ == "__main__":
    nltk.download('punkt')  # Ensure the NLTK tokenizer is available
    nltk.download('punkt_tab')
    # Example usage
    corpus = "This is a sample text. It is used for training the n-gram model. The model will generate text based on the training data."
    ngram_model = NGramModelWithLaplaceSmoothing(max_n=3, vocabulary_size=0)
    ngram_model.train(corpus)
    
    print("Generated text:", ngram_model.generate_text(max_length=10))
    print("Probability of 'text' given context ('the', 'model'):", ngram_model.prob(('the', 'model'), 'text'))
    print("Vocabulary size:", ngram_model.vocabulary_size)