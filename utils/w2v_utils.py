from gensim.models import Word2Vec
from nltk.tokenize import sent_tokenize, word_tokenize

class Word2VecHandler:
    def __init__(self):
        self.model = None

    @classmethod
    def prepare_sentences(cls, full_text):
        all_sentences = []

        for sentence in sent_tokenize(full_text):
            words = word_tokenize(sentence)
            all_sentences.append(words)

        return all_sentences

    def train_model(self, sentences, **kwargs):
        if self.model is None:
            self.model = Word2Vec(
                vector_size=kwargs.get("vector_size", 100),
                window=kwargs.get("window", 5),
                min_count=kwargs.get("min_count", 1),
                workers=kwargs.get("workers", 4)
            )
            self.model.build_vocab(sentences)
        else:
            self.model.build_vocab(sentences, update=True)
        
        self.model.train(
            sentences, total_examples=len(sentences),
            epochs=kwargs.get("epochs", 5)
        )
    
    def find_most_similar(self, word, topn=10):
        if self.model is not None:
            if word in self.model.wv.key_to_index:
                return self.model.wv.most_similar(word, topn=topn)
            
            raise IndexError(f"{word} not in vocabulary.")

        raise ValueError("Must train model first.")
    
    def find_similarity(self, word1, word2):
        if self.model is not None:
            return self.model.wv.similarity(word1, word2)
        
        raise ValueError("Must train model first.")
    
    def find_odd_one(self, words):
        if self.model is not None:
            return self.model.wv.doesnt_match(words)
        
        raise ValueError("Must train model first.")

