# Text Mining and Analysis

Este projeto realiza a mineração de textos, incluindo a análise de sentimentos, extração de frases importantes, classificação utilizando diferentes modelos e geração de relatórios em HTML. 

Este projeto foi realizado originalmente para a produção do capítulo: Ameaças Eletromagnéticas escrito por mim em co-autoria com os alunos do Curso de Física da UESPI Camila Borges dos Santos e Francisco Gustavo da Silva Araújo para o livro Sementes de Futuro em Defesa, organizado pelo Prof. Dr. Bernardo Rodrigues (IRID/UFRJ) publicado pela Editora Alpheratz em Julho de 2024.

## Funcionalidades

- **Carregamento e pré-processamento de textos**: Suporte a arquivos `.docx` e `.txt`.
- **Determinação do número ideal de clusters**: Método de Elbow.
- **Extração de frases importantes**: Baseado em similaridade cosseno e clustering.
- **Análise de sentimentos**: Utilizando o SentimentIntensityAnalyzer do NLTK.
- **Classificação de textos**: Usando Naive Bayes, Random Forest e SVM.
- **Geração de relatórios**: Criação de um relatório HTML com gráficos e tabelas.

## Como Usar

1. **Clone o repositório:**

   ```bash
   git clone git@github.com:LeoVichi/Data_Mining_for_Defense_Studies.git
   cd text-mining-analysis
   ```

2. **Instale as dependências:**

   Certifique-se de que o Python 3.x está instalado em seu sistema. Em seguida, instale as dependências usando o `pip`:

   ```bash
   pip install -r requirements.txt
   ```

3. **Execute o Script:**

   Coloque seus arquivos de texto na pasta `Data Mining/AIWeather` e execute o script:

   ```bash
   python text_mining_analysis.py
   ```

4. **Visualize os Resultados:**

   Os resultados, incluindo gráficos e um relatório HTML, serão salvos na pasta `Data Mining`.

## Requisitos

- Python 3.x
- Bibliotecas listadas no `requirements.txt`

## Licença

Este projeto é licenciado sob os termos da licença MIT. Veja o arquivo [LICENSE](LICENSE) para mais detalhes.

## Autor

- **L3nny_P34s4n7**
- **Email:** contact@leonardovichi.com
