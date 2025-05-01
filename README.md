# IC007: Tópicos Avançados em Bancos de Dados I - Grupo 08

## Notebook Interativo

- `corpus_stats.ipynb`: lê o arquivo JSON com o corpus e extrai estatísticas descritivas.

## Processando o corpus:
1. Crie um ambiente virtual: na raíz do projeto: `python3 -m venv venv`
2. Entre no ambiente com: `source ./venv/bin/activate`
3. Configure o ambiente com: `pip3 install -r requirements.txt`
4. Instale o Modelo de Linguagem do Spacy: `python3 -m spacy download pt_core_news_sm`
5. Execute o código com: `python3 process_corpus.py`