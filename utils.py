import os
import tqdm
import spacy
from langdetect import detect
from nltk.tokenize import sent_tokenize
from deep_translator import GoogleTranslator

SPACY_MODEL = "pt_core_news_sm"

def __download_spacy_model(model_name):
    installed_models = spacy.util.get_installed_models()
    if model_name not in installed_models:
        cmd = f"python -m spacy download {model_name}"
        os.system(cmd)

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

def sentencize_text(text):
    global SPACY_MODEL

    __download_spacy_model(SPACY_MODEL)

    nlp_model = spacy.load(SPACY_MODEL)
    sentences = [sent.text for sent in nlp_model(text).sents]
    return sentences

def tokenize_text(text):
    global SPACY_MODEL

    __download_spacy_model(SPACY_MODEL)

    nlp_model = spacy.load(SPACY_MODEL)
    tokens = []
    for token in nlp_model(text):
        tokens.append({
            "token": token.text,
            "pos": token.pos_,
            "lemma": token.lemma_,
            "dep": token.dep_
        })

    return tokens

