import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import sent_tokenize
from nltk.sentiment import SentimentIntensityAnalyzer
from bs4 import BeautifulSoup
from docx import Document
import nltk
from imblearn.over_sampling import SMOTE

nltk.download('punkt')

def load_texts(filepaths):
    texts = []
    for filepath in filepaths:
        if filepath.endswith('.docx'):
            doc = Document(filepath)
            full_text = []
            for para in doc.paragraphs:
                full_text.append(para.text)
            texts.append('\n'.join(full_text))
        else:
            with open(filepath, 'r', encoding='utf-8') as file:
                texts.append(file.read())
    return texts

def preprocess_texts(texts):
    processed_texts = []
    for text in texts:
        text = re.sub(r'http\S+|www\S+|doi\S+', '', text)
        processed_texts.append(text)
    return processed_texts

def determine_optimal_clusters(texts, max_k=10):
    vectorizer = TfidfVectorizer(max_df=0.85, max_features=10000)
    X = vectorizer.fit_transform(texts)
    sse = []
    for k in range(1, min(max_k, X.shape[0]) + 1):
        kmeans = KMeans(n_clusters=k, random_state=0).fit(X)
        sse.append(kmeans.inertia_)
    
    plt.figure()
    plt.plot(range(1, min(max_k, X.shape[0]) + 1), sse, marker='o')
    plt.xlabel('Número de Clusters')
    plt.ylabel('SSE')
    plt.title('Método de Elbow para K ideal')
    plt.savefig('Data Mining/elbow_method.png')
    plt.close()

def extract_important_phrases_with_cosine_similarity(texts, num_clusters=10, num_phrases=10):
    important_phrases = []
    vectorizer = TfidfVectorizer(max_df=0.85, max_features=10000)
    for text in texts:
        sentences = sent_tokenize(text)
        if len(sentences) < num_clusters:
            num_clusters = len(sentences)
        X = vectorizer.fit_transform(sentences)
        kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(X)
        cosine_similarities = cosine_similarity(X, kmeans.cluster_centers_)
        closest = cosine_similarities.argmax(axis=0)
        phrases = [sentences[idx] for idx in closest if len(sentences[idx].split()) > 5]
        if len(phrases) < num_phrases:
            additional_phrases = sorted(sentences, key=lambda x: len(x.split()), reverse=True)[:num_phrases-len(phrases)]
            phrases.extend(additional_phrases)
        important_phrases.append(phrases[:num_phrases])
    return important_phrases

def sentiment_analysis(phrases, positive_threshold=0.1, negative_threshold=-0.1):
    sia = SentimentIntensityAnalyzer()
    sentiments = []
    for phrase_list in phrases:
        phrase_sentiments = []
        for phrase in phrase_list:
            sentiment = sia.polarity_scores(phrase)
            if sentiment['compound'] >= positive_threshold:
                phrase_sentiments.append((phrase, 'Positivo', sentiment['compound']))
            elif sentiment['compound'] <= negative_threshold:
                phrase_sentiments.append((phrase, 'Negativo', sentiment['compound']))
            else:
                phrase_sentiments.append((phrase, 'Neutro', sentiment['compound']))
        sentiments.append(phrase_sentiments)
    return sentiments

def plot_sentiment_analysis(sentiments, output_path='Data Mining/sentiment_analysis.png'):
    sentiment_counts = {
        'Positivo': 0,
        'Neutro': 0,
        'Negativo': 0
    }

    for sentiment_list in sentiments:
        for _, sentiment, _ in sentiment_list:
            sentiment_counts[sentiment] += 1
    
    labels = list(sentiment_counts.keys())
    sizes = list(sentiment_counts.values())
    
    plt.figure(figsize=(10, 6))
    plt.bar(labels, sizes, color=['green', 'grey', 'red'])
    plt.xlabel('Sentimento')
    plt.ylabel('Número de frases')
    plt.title('Análise de sentimento de frases-chaves')
    plt.savefig(output_path)
    plt.close()

def plot_classification_report(report, accuracy, output_path='Data Mining/'):
    report_data = []
    for label, metrics in report.items():
        if label in ["accuracy", "macro avg", "weighted avg"]:
            continue
        report_data.append({
            'Class': label,
            'Precisão': metrics['precision'],
            'Revocação': metrics['recall'],
            'Pontuação F1': metrics['f1-score'],
            'Support': metrics['support']
        })

    df = pd.DataFrame(report_data)
    df.set_index('Class', inplace=True)

    plt.figure(figsize=(10, 6))
    df[['Precisão', 'Revocação', 'Pontuação F1']].plot(kind='bar')
    plt.title(f'Relatório de classificadores\nAcurácia: {accuracy:.2f}')
    plt.xlabel('Class')
    plt.ylabel('Pontuação')
    plt.ylim(0, 1)
    plt.legend(loc='lower right')
    plt.savefig(output_path)
    plt.close()

def add_classification_table(soup, report, title_text, accuracy):
    section = soup.new_tag('div')
    title = soup.new_tag('h2')
    title.string = title_text
    section.append(title)

    accuracy_tag = soup.new_tag('p')
    accuracy_tag.string = f'Acurácia: {accuracy:.2f}'
    section.append(accuracy_tag)

    table = soup.new_tag('table')
    table['border'] = 1
    header = soup.new_tag('tr')
    for col in ['Classe', 'Precisão', 'Revocação', 'Pontuação F1', 'Suporte']:
        th = soup.new_tag('th')
        th.string = col
        header.append(th)
    table.append(header)

    for label, metrics in report.items():
        if label in ["accuracy", "macro avg", "weighted avg"]:
            continue
        row = soup.new_tag('tr')
        td_class = soup.new_tag('td')
        td_class.string = label
        row.append(td_class)
        for metric in ['precision', 'recall', 'f1-score', 'support']:
            td = soup.new_tag('td')
            td.string = str(metrics[metric])
            row.append(td)
        table.append(row)

    section.append(table)
    return section

def generate_html_report(phrases_sentiments, reports, accuracies, output_path='Data Mining/report.html'):
    soup = BeautifulSoup('<html><head><title>Relatório de Análise</title></head><body></body></html>', 'html.parser')
    body = soup.body
    
    for i, phrase_sentiments in enumerate(phrases_sentiments):
        section = soup.new_tag('div')
        title = soup.new_tag('h2')
        title.string = f'Corpus {i+1}'
        section.append(title)
        
        table = soup.new_tag('table')
        table['border'] = 1
        header = soup.new_tag('tr')
        th_phrase = soup.new_tag('th')
        th_phrase.string = 'Frase'
        header.append(th_phrase)
        th_sentiment = soup.new_tag('th')
        th_sentiment.string = 'Sentimento'
        header.append(th_sentiment)
        th_score = soup.new_tag('th')
        th_score.string = 'Pontuação'
        header.append(th_score)
        table.append(header)
        
        for phrase, sentiment, score in phrase_sentiments:
            row = soup.new_tag('tr')
            td_phrase = soup.new_tag('td')
            td_phrase.string = phrase
            row.append(td_phrase)
            td_sentiment = soup.new_tag('td')
            td_sentiment.string = sentiment
            row.append(td_sentiment)
            td_score = soup.new_tag('td')
            td_score.string = str(score)
            row.append(td_score)
            table.append(row)
        
        section.append(table)
        body.append(section)
    
    img_tag = soup.new_tag('img', src='sentiment_analysis.png', alt='Sentiment Analysis Graph')
    body.append(img_tag)
    
    elbow_img_tag = soup.new_tag('img', src='elbow_method.png', alt='Elbow Method Graph')
    body.append(elbow_img_tag)

    for model_name, (report, accuracy) in reports.items():
        img_tag = soup.new_tag('img', src=f'{model_name}_classification_report.png', alt=f'Relatório de classificadores {model_name}')
        body.append(img_tag)
        body.append(add_classification_table(soup, report, f"Relatório de classificadores {model_name}", accuracy))
    
    with open(output_path, 'w', encoding='utf-8') as file:
        file.write(soup.prettify())

def balance_classes(X, y):
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    return X_resampled, y_resampled

def try_different_models(X_train, y_train, X_test, y_test):
    models = {
        'Naive_Bayes': MultinomialNB(),
        'Random_Forest': RandomForestClassifier(random_state=42),
        'SVM': SVC(kernel='linear', random_state=42)
    }
    
    reports = {}
    accuracies = {}
    
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        print(f"\nRelatório de classificadores para {name.replace('_', ' ')}:")
        report = classification_report(y_test, y_pred, zero_division=0, output_dict=True)
        accuracy = accuracy_score(y_test, y_pred)
        print(classification_report(y_test, y_pred, zero_division=0))
        print(f"Accuracy: {accuracy:.2f}")
        
        plot_classification_report(report, accuracy, output_path=f'Data Mining/{name}_classification_report.png')
        
        reports[name] = (report, accuracy)
    
    return reports, accuracies

def classify_and_report(texts, sentiments):
    data = []
    for i, doc in enumerate(sentiments):
        for phrase, sentiment, _ in doc:
            data.append([i, phrase, sentiment])
    
    df = pd.DataFrame(data, columns=['doc_id', 'phrase', 'sentiment'])
    
    print("Distribuição das classes antes do balanceamento:")
    print(df['sentiment'].value_counts())
    
    vectorizer = TfidfVectorizer(max_df=0.85, max_features=10000)
    X = vectorizer.fit_transform(df['phrase'])
    
    y = df['sentiment']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    print("Distribuição das classes no conjunto de treinamento antes do balanceamento:")
    print(pd.Series(y_train).value_counts())
    print("Distribuição das classes no conjunto de teste antes do balanceamento:")
    print(pd.Series(y_test).value_counts())
    
    X_train_balanced, y_train_balanced = balance_classes(X_train, y_train)
    
    print("Distribuição das classes no conjunto de treinamento após balanceamento:")
    print(pd.Series(y_train_balanced).value_counts())
    
    reports, accuracies = try_different_models(X_train_balanced, y_train_balanced, X_test, y_test)
    
    return reports, accuracies

directory = 'Data Mining/AIWeather'

filepaths = [os.path.join(directory, f'artigo{i+1}.docx') for i in range(4)]

texts = load_texts(filepaths)

print("Carregamento de textos completo. Primeiros 100 caracteres de cada texto:")
for i, text in enumerate(texts):
    print(f"Texto {i+1}: {text[:100]}")

processed_texts = preprocess_texts(texts)

print("Pré-processamento de textos completo. Primeiros 100 caracteres de cada texto:")
for i, text in enumerate(processed_texts):
    print(f"Texto {i+1}: {text[:100]}")

determine_optimal_clusters(processed_texts, max_k=10)

important_phrases = extract_important_phrases_with_cosine_similarity(processed_texts, num_clusters=4, num_phrases=15)

print("Extração de frases importantes completa. Frases extraídas:")
for i, phrases in enumerate(important_phrases):
    print(f"Texto {i+1}:")
    for phrase in phrases:
        print(f"- {phrase}")

phrases_sentiments = sentiment_analysis(important_phrases)

print("Análise de sentimentos completa. Sentimentos detectados:")
for i, sentiments in enumerate(phrases_sentiments):
    print(f"Texto {i+1}:")
    for phrase, sentiment, score in sentiments:
        print(f"- Phrase: {phrase}, Sentiment: {sentiment}, Score: {score}")

plot_sentiment_analysis(phrases_sentiments)

reports, accuracies = classify_and_report(processed_texts, phrases_sentiments)

generate_html_report(phrases_sentiments, reports, accuracies)
