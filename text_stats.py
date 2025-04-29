import nltk
import pandas as pd
import spacy
import string

from collections import Counter

from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords


class TextStats():

    def __init__(self, text):
        self.text = text 
        self.sentences = sent_tokenize(self.text)
        self.tokens =  word_tokenize(self.text)

    def count_by_pos_tag(self):
        nlp = spacy.load("pt_core_news_sm")
        doc = nlp(self.text)
        counts = doc.count_by(spacy.attrs.POS)
        mapper = {val:key for key, val in spacy.symbols.IDS.items()}

        counts_by_pos = {mapper[key]: val for key, val in counts.items()}
        
        return counts_by_pos
    
    def extract_frequencies(self):
        stop_words = set(stopwords.words('portuguese'))
        
        filtered_tokens = [w for w in self.tokens if not w.lower() in stop_words]
        filtered_tokens = [w for w in filtered_tokens if not w in string.punctuation]
        return pd.DataFrame.from_dict(Counter(filtered_tokens), orient="index")\
            .reset_index().rename(columns={'index':'Token', 0:'Freq'})\
                .sort_values(by="Freq", ascending=False, ignore_index=True)

    def get_num_sentences(self):
        return len(self.sentences)
        
    def get_num_tokens(self):
        return len(self.tokens)

    def get_num_unique_tokens(self):
        return len(set(self.tokens))
        