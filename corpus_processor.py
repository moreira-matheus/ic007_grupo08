import os
import re
import nltk
import tqdm
import pymupdf
import pymupdf4llm
from unicodedata import normalize
from langdetect import detect

from nltk.tokenize import sent_tokenize
from deep_translator import GoogleTranslator
from multiprocessing import Pool, cpu_count

class CorpusProcessor:
    def __init__(self, input_dir, tgt_lang="pt"):
        self.input_dir = input_dir
        self.tgt_lang = tgt_lang
    
    def detect_language(self, text):
        return detect(text)

    def fix_char_substitutions(self, text):
        replacements = {
            "c¸": "ç",
            "a˜": "ã", "˜a": "ã", "a´": "á", "aˆ": "â",
            "´a": "á", "ˆa": "â", "ˆa": "â",
            "´e": "é", "e´": "é", "ˆe": "ê", "eˆ": "ê",
            "´i": "í", "´ı": "í", "´ı": "í",
            "˜o": "õ", "´o": "ó",
            "u´": "ú", "´u": "ú", "¨u": "ü"
            # Add more as needed
        }
        for wrong, right in replacements.items():
            text = text.replace(wrong, right)
        
        return text

    def pre_process_md(self, md_text_):
        #md_text = normalize('NFKD', md_text_).encode('ascii','ignore').decode("utf-8")
        md_text = self.fix_char_substitutions(md_text_)
        md_text = re.sub(r'- ', '', md_text).strip()
        md_text = re.sub(r'\n+', '\n', md_text).strip()
        md_text = re.sub(r"-{3,}", "", md_text).strip()

        return md_text

    def translate_text(self, text, src_lang, tgt_lang):
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

    def remove_references(self, md_text):
        ref_section_names = ["References", "Referências", "Referencias"]
        output_text = md_text[:]

        for ref_name in ref_section_names:
            pat = f"## **{ref_name}**"
            if pat in output_text:
                output_text = output_text.split(pat, 2)[0]
        
        return output_text

    def post_process_md(self, md_text_):
        if md_text_ is None:
            return None
        
        md_text = re.sub(r"\*{1,}", "", md_text_).strip()
        md_text = re.sub(r"|{2,}", "", md_text).strip()
        md_text = re.sub(r"#{1,}", "", md_text).strip()
        md_text = re.sub(r"-\* \*", "", md_text).strip()
        md_text = re.sub(r"\* \*", " ", md_text).strip()
        md_text = re.sub(r"\n", " ", md_text).strip()
        md_text = re.sub(r"\s+", " ", md_text).strip()

        return md_text
    
    def process_doc(self, fname):
        full_fname = os.path.join(self.input_dir, fname)
        raw_md_text = pymupdf4llm.to_markdown(full_fname)
        pre_processed_md = self.pre_process_md(raw_md_text)
        text_no_refs = self.remove_references(pre_processed_md)
        postprocessed_text = self.post_process_md(text_no_refs)
        src_lang = self.detect_language(postprocessed_text)
        if src_lang != self.tgt_lang:
            postprocessed_text = self.translate_text(
                postprocessed_text, src_lang, self.tgt_lang
            )
    
        return postprocessed_text
    
    def pdf_to_text_multiproc(self):
        fnames = os.listdir(self.input_dir)
        with Pool(cpu_count()) as pool:
            texts = pool.map(self.process_doc, fnames)
        
        return dict(zip(fnames, texts))

    
    def pdf_to_text(self):
        articles = {}

        for fname in os.listdir(self.input_dir):
            full_fname = os.path.join(self.input_dir, fname)
            raw_md_text = pymupdf4llm.to_markdown(full_fname)
            pre_processed_md = self.pre_process_md(raw_md_text)
            text_no_refs = self.remove_references(pre_processed_md)
            postprocessed_text = self.post_process_md(text_no_refs)

            src_lang = self.detect_language(postprocessed_text)
            if src_lang != self.tgt_lang:
                postprocessed_text = self.translate_text(
                    postprocessed_text, src_lang, self.tgt_lang
                )
            
            articles[fname] = postprocessed_text
        
        return articles
