import os
import pymupdf4llm
import re
import tqdm

from unicodedata import normalize

from nltk.tokenize import sent_tokenize
from deep_translator import GoogleTranslator


class PdfReader():

    def __init__(self):
        self.ART_FOLDER = "./artigos/"
    
    def pre_process_md(self, md_text_):
        md_text = normalize('NFKD', md_text_).encode('ascii','ignore').decode("utf-8")
        md_text = re.sub(r'\n+', '\n', md_text).strip()
        md_text = re.sub(r"-{3,}", "", md_text).strip()

        return md_text

    def remove_references(self, md_text):
        return md_text.split("## **References**").pop(0)

    def translate_text(self, text, src_lang, tgt_lang):
        all_translated = []

        for sentence in tqdm.tqdm(sent_tokenize(text)):
            translated = GoogleTranslator(
                source=src_lang,
                target=tgt_lang
            ).translate(sentence)
            
            all_translated.append(translated)
        
        return ' '.join(all_translated)

    def post_process_md(self, md_text_):
        if md_text_ is None:
            return None
        
        md_text = re.sub(r"\*{1,}", "", md_text_).strip()
        md_text = re.sub(r"#{1,}", "", md_text).strip()
        md_text = re.sub(r"-\* \*", "", md_text).strip()
        md_text = re.sub(r"\* \*", " ", md_text).strip()
        md_text = re.sub(r"\n", " ", md_text).strip()

        return md_text
    
    def extract_text(self, pre_processed_pdf_text: str) -> str:
        text = self.remove_references(pre_processed_pdf_text)
        return self.post_process_md(text)
    

    def pdf_to_text(self, folder_path: str) -> str:
        articles = {}
        if not folder_path:
            folder_path = self.ART_FOLDER

        for fname in os.listdir(folder_path):
            full_fname = os.path.join(folder_path, fname)
            pdf_text = pymupdf4llm.to_markdown(full_fname)
            pre_processed_pdf_text = self.pre_process_md(pdf_text)
            final_pdf_text = self.extract_text(pre_processed_pdf_text)
            translated = self.translate_text(
                final_pdf_text, "english", "portuguese"
            )
            articles[fname] = translated

        return articles        