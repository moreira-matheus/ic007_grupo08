import os
import re
import json
import pymupdf4llm

from datetime import datetime
from multiprocessing import Pool, cpu_count

from utils import (
    detect_language, fix_char_substitutions,
    translate_text, tokenize_text
)

TGT_LANG = "pt"

class CorpusTemplate:
    def __init__(self):
        fields = [
            "titulo", "informacoes_url", "idioma", "storage_key",
            "autores", "data_publicacao", "resumo", "keywords",
            "referencias", "artigo_completo", "artigo_tokenizado",
            "pos_tagger", "lema", "dep"
        ]

        for field in fields:
            setattr(self, field, None)

    def _pre_process_md(self, md_text_):
        md_text = fix_char_substitutions(md_text_)
        md_text = re.sub(r'- ', '', md_text).strip()
        md_text = re.sub(r'\n+', '\n', md_text).strip()
        md_text = re.sub(r"-{3,}", "", md_text).strip()
        md_text = re.sub(r"\|{2,}", "|", md_text).strip()

        return md_text

    def _post_process_md(self, md_text_):
        if md_text_ is None:
            return None
        
        md_text = re.sub(r"\*{1,}", "", md_text_).strip()
        md_text = re.sub(r"#{1,}", "", md_text).strip()
        md_text = re.sub(r"-\* \*", "", md_text).strip()
        md_text = re.sub(r"\* \*", " ", md_text).strip()
        md_text = re.sub(r"\n", " ", md_text).strip()
        md_text = re.sub(r"\s+", " ", md_text).strip()

        return md_text
    
    def _extract_title(self, preprocessed_md_text):
        pat = r"^##?\s\*\*(.+)\*\*\n"
        match = re.search(pat, preprocessed_md_text)
        if match:
            return self._post_process_md(match.group(1))
        
        return None
    
    def __remove_abstract_from_authors(self, authors_str):
        SECTION_NAMES = ["Abstract", "Resumo"]
        for section_name in SECTION_NAMES:
            pat = rf"(?:###\s)?\*{section_name}\.\s(.+)\*"
            authors_str = re.split(pat, authors_str)[0].strip()

        return authors_str

    def _extract_authors(self, preprocessed_md_text):
        pat = r"^###\s(.+)$"
        authors = re.search(pat, preprocessed_md_text, re.MULTILINE).group(1)
        authors = self.__remove_abstract_from_authors(authors)
        return authors
    
    def _extract_date(self, full_fname):
        pat = r"-(\d{8})\.pdf$"
        match = re.search(pat, full_fname)
        if match:
            return datetime.strptime(match.group(1), '%Y%m%d')\
                .strftime("%d-%m-%Y")
        return None

    def _extract_abstract(self, preprocessed_md_text):
        SECTION_NAMES = ["Abstract", "Resumo"]
        abstracts = []

        for section_name in SECTION_NAMES:
            pat = rf"(?:###\s)?\*{section_name}\.\s(.+)\*"
            match = re.search(pat, preprocessed_md_text)
            if match:
                abstracts.append(self._post_process_md(match.group(1)))
        
        return abstracts
    
    def _extract_sections(self, preprocessed_md_text):
        pat = r"^##\s\*\*(.+)\*\*$"
        section_titles = []
        section_contents = []
        match_starts = []
        match_ends = []

        for match in re.finditer(pat, preprocessed_md_text, re.MULTILINE):
            section_titles.append(match.group(1))
            match_starts.append(match.start())
            match_ends.append(match.end())

        match_starts = match_starts[1:] + [len(preprocessed_md_text)]

        for start, end in zip(match_ends, match_starts):
            section_contents.append(self._post_process_md(preprocessed_md_text[start:end]))

        sections = {
            title: content\
                for title, content in zip(section_titles, section_contents) 
        }
        return sections
    
    def _split_out_references(self, sections_dict):
        SECTION_NAMES = ["References", "Referencias", "Referências"]
        numbered_sections = {section_name: section_content\
                            for section_name, section_content in sections_dict.items()}

        ref_keys = [key for key in sections_dict.keys() if key in SECTION_NAMES]
        if not ref_keys:
            return numbered_sections, dict()
        
        ref_sections = dict()
        for key in ref_keys:
            ref_sections[key] = numbered_sections.pop(key)

        return numbered_sections, ref_sections
    
    def _concat_sections(self, sections_dict):
        return "\n".join(
            [f"{key}\n{val}" for key, val in sections_dict.items()]
        )
        
    def process_text(self, full_fname):
        global TGT_LANG

        raw_md_text = pymupdf4llm.to_markdown(
            doc=full_fname,
            ignore_images=True,
            ignore_graphics=True,
            table_strategy=None
        )
        pre_processed_md = self._pre_process_md(raw_md_text)

        self.titulo = self._extract_title(pre_processed_md)
        self.informacoes_url = None
        self.idioma = detect_language(pre_processed_md)
        self.storage_key = full_fname
        self.autores = self._extract_authors(pre_processed_md)
        self.data_publicacao = self._extract_date(full_fname)
        self.resumo = self._extract_abstract(pre_processed_md)
        self.keywords = None

        sections_dict = self._extract_sections(pre_processed_md)
        numbered_sections, references_dict = self._split_out_references(sections_dict)
        self.referencias = list(references_dict.values()).pop()\
            if references_dict else None

        postprocessed_text = self._concat_sections(numbered_sections)
        if self.idioma != TGT_LANG:
            postprocessed_text = translate_text(postprocessed_text, self.idioma, TGT_LANG)
        
        self.artigo_completo = postprocessed_text[:]

        token_list = tokenize_text(postprocessed_text)
        self.artigo_tokenizado = [token.get("token") for token in token_list]
        self.pos_tagger = [token.get("pos") for token in token_list]
        self.lema = [token.get("lemma") for token in token_list]
        self.dep = [token.get("dep") for token in token_list]

class CorpusHandler:
    def __init__(self, path):
        self.path = path
        self.templates = []

    def load_template(self, full_fname: str) -> CorpusTemplate: # multiproc
        template = CorpusTemplate()
        template.process_text(full_fname)
        return template
    
    def load_corpus(self) -> None:
        full_fnames = [os.path.join(self.path, fname)\
                      for fname in os.listdir(self.path)]
        with Pool(cpu_count()) as pool:
            corpora = pool.map(self.load_template, full_fnames)
        
        if corpora:
            self.templates = corpora[:]
    
    def to_json(self) -> str:
        template_dicts = [
            template.__dict__ \
            for template in self.templates
        ]
        return json.dumps(template_dicts, ensure_ascii=False)\
            .encode("utf-8").decode()

if __name__ == "__main__":
    input_dir = "./input/"
    output_fname = "corpus.json"

    handler = CorpusHandler("./input/")
    handler.load_corpus()
    json_str = handler.to_json()

    with open(output_fname, "w", encoding="utf-8") as out:
        out.write(json_str)
