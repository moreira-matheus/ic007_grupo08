import nltk
import pandas as pd
import spacy

from collections import Counter

from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords


class TextStats():

    def __init__(self, text):
        self.text = text 
        self.sentences = sent_tokenize(self.text)
        self.tokens =  word_tokenize(self.text)

    def contar_por_classe_pos(self):
        nlp = spacy.load("pt_core_news_sm")
        doc = nlp(self.text)
        counts = doc.count_by(spacy.attrs.POS)
        mapper = {val:key for key, val in spacy.symbols.IDS.items()}

        contagens = {mapper[key]: val for key, val in counts.items()}
        
        return contagens
    
    def extract_frequencies(self):
        stop_words = set(stopwords.words('portuguese'))
        # sem as stop words
        ponctuation = ['.',',',':','(',')',';']
        stop_words.update(ponctuation)
        filtered_sentence = [w for w in self.tokens if not w.lower() in stop_words]
        return pd.DataFrame.from_dict(Counter(filtered_sentence), orient="index").reset_index()\
    .rename(columns={'index':'Token', 0:'Freq'}).sort_values(by="Freq", ascending=False)

    def get_num_sentencas(self):
        return len(self.sentences)
        
    def get_num_tokens(self):
        return len(self.tokens)

    def get_num_unique_tokens(self):
        len(set(self.tokens))
        