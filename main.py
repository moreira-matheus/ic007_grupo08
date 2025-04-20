import nltk

from pprint import pprint
from extract_pdf_content import PdfReader
from collect_text_statistics import TextStats

def configure_nltk():
    nltk.download('punkt')
    nltk.download('punkt_tab')
    nltk.download('stopwords') 
    nltk.download('perluni') 
    nltk.download('nonbreaking_prefixes')

def show_text_data(curr_stat: TextStats) -> None:

    frequencies = curr_stat.extract_frequencies()
    num_sentencas = curr_stat.get_num_sentencas()
    num_tokens = curr_stat.get_num_tokens()

    top_10_tokens = frequencies[['Token', 'Freq']].head(10)
    down_10_tokens = frequencies[['Token', 'Freq']].tail(10)

    contagens = curr_stat.contar_por_classe_pos()

    subs_comuns = contagens['NOUN']
    subs_proprios = contagens['PROPN']

    num_verbos = contagens['VERB']

    num_preposicoes = contagens['ADP']

    print(f'Número de Sentenças: {num_sentencas}\n')
    print(f'Número de Tokens: {num_tokens}\n')
    print(f'Top 10 Tokens')
    pprint(top_10_tokens)
    print()
    print(f'Down 10 Tokens')
    pprint(down_10_tokens)
    print()
    print(f"Num. substantivos comuns: {subs_comuns}",)
    print(f"Num. substantivos próprios: {subs_proprios}")
    print(f"Total: {subs_comuns + subs_proprios}\n",)
    print(f"Número de Verbos: {num_verbos}\n")
    print(f"Número de Preposicoes: {num_preposicoes}\n")

if __name__ == '__main__':
    configure_nltk()
    pdf_reader = PdfReader()
    articles = pdf_reader.pdf_to_text(folder_path = './artigos')
    for name, content in articles.items():
        curr_stat = TextStats(text=content)
        show_text_data(curr_stat)