# IC007: Tópicos Avançados em Bancos de Dados I - Grupo 08

## Notebooks Interativos

1. `processamento.ipynb`: lê o artigo em PDF, extrai texto, remove caracteres de formatação, remove referências e traduz do inglês para o português.

2. `analise_morfossintatica.ipynb`: tokeniza o texto processado e calcula as estatísticas solicitadas.


## Rodando o Código Completo
1. Crie um ambiente virtual: na raíz do projeto: `python3 -m venv venv`
2. Entre no ambiente com: `source ./venv/bin/activate`
3. Configure o ambiente com: `pip3 install -r requirements.txt`
4. Instale o Modelo de Linguagem do Spacy: `python3 -m spacy download pt_core_news_sm`
5. Execute o código com: `python3 main.py`