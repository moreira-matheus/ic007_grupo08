
import tqdm
import spacy
from langdetect import detect
from nltk.tokenize import sent_tokenize
from deep_translator import GoogleTranslator

def detect_language(text):
    return detect(text)

def fix_char_substitutions(text):
    replacements = {
        "c¸": "ç",
        "a˜": "ã", "˜a": "ã", "a´": "á", "aˆ": "â",
        "´a": "á", "ˆa": "â", "ˆa": "â",
        "´e": "é", "e´": "é", "ˆe": "ê", "eˆ": "ê",
        "´i": "í", "´ı": "í", "´ı": "í",
        "˜o": "õ", "´o": "ó","ˆo": "ô",
        "u´": "ú", "´u": "ú", "¨u": "ü"
        # Add more as needed
    }
    for wrong, right in replacements.items():
        text = text.replace(wrong, right)
    
    return text

def translate_text(text, src_lang, tgt_lang):
    all_translated = []
    translator = GoogleTranslator(
        source=src_lang,
        target=tgt_lang
    )

    for sentence in tqdm.tqdm(sent_tokenize(text)):
        translated = translator.translate(sentence)
        
        all_translated.append(translated)
    
    all_translated = [t for t in all_translated if t is not None]
    return ' '.join(all_translated)

def tokenize_text(text):
    nlp_model = spacy.load("pt_core_news_sm")
    tokens = []
    for token in nlp_model(text):
        tokens.append({
            "token": token.text,
            "pos": token.pos_,
            "lemma": token.lemma_,
            "dep": token.dep_
        })

    return tokens
