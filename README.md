# IC007 – Tópicos Avançados em Bancos de Dados I · Grupo 08

Este repositório contém os códigos, notebooks e dados utilizados no trabalho final da disciplina **PGCOMP/IC0007 – Tópicos Avançados em Bancos de Dados I (2025.1)**, com foco em análise linguística de textos científicos sobre **comunicação quântica**, utilizando técnicas de **Processamento de Linguagem Natural (PLN)**.

## 👥 Integrantes do Grupo 08
- Lucas Mascarenhas Almeida
- Mário Augusto Santos do Amor Divino
- Marcus Elias Silva Freire  
- Matheus Moreira Silva Rebouças dos Santos  
- Victor Soares Cardel  

## 🧪 Como Executar o Projeto

Siga os passos abaixo para configurar e rodar o projeto localmente:

```bash
# 1. Crie um ambiente virtual
python3 -m venv venv

# 2. Ative o ambiente virtual
source ./venv/bin/activate

# 3. Instale as dependências
pip install -r requirements.txt

# 4. Baixe o modelo do spaCy para português
python3 -m spacy download pt_core_news_sm

# 5. Execute o script desejado
python process_corpus.py
```
