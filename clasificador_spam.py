import pandas as pd
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string
import os
import re
import math
from typing import Dict, List, Tuple

def setup_nltk() -> None:
    resources = ['punkt', 'punkt_tab', 'stopwords', 'wordnet']
    for resource in resources:
        nltk.download(resource, quiet=True)
    
    ruta_nltk = os.path.join(os.getcwd(), 'nltk_data')
    if ruta_nltk not in nltk.data.path:
        nltk.data.path.insert(0, ruta_nltk)

def preprocess_text(text: str, stop_words: set, lemmatizer: WordNetLemmatizer) -> List[str]:
    if not isinstance(text, str) or not text.strip():
        return []
    text = re.sub(r'[^a-z\s]', '', text.lower())
    tokens = word_tokenize(text)
    tokens = [token for token in tokens if token not in string.punctuation and token not in stop_words]
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    tokens = [token for token in tokens if token.strip()]
    return tokens

def train_bayesian_model(df: pd.DataFrame, stop_words: set, lemmatizer: WordNetLemmatizer) -> Tuple[float, float, Dict[str, Tuple[float, float]]]:
    total_messages = len(df)
    spam_count = len(df[df['Label'] == 'spam'])
    ham_count = len(df[df['Label'] == 'ham'])
    p_s = spam_count / total_messages
    p_h = ham_count / total_messages

    word_counts_spam = {}
    word_counts_ham = {}
    total_words_spam = 0
    total_words_ham = 0

    for idx, row in df.iterrows():
        tokens = preprocess_text(row['SMS_TEXT'], stop_words, lemmatizer)
        if row['Label'] == 'spam':
            for token in tokens:
                word_counts_spam[token] = word_counts_spam.get(token, 0) + 1
                total_words_spam += 1
        else:
            for token in tokens:
                word_counts_ham[token] = word_counts_ham.get(token, 0) + 1
                total_words_ham += 1

    vocab = set(word_counts_spam.keys()).union(set(word_counts_ham.keys()))
    word_probs = {}
    vocab_size = len(vocab)

    for word in vocab:
        count_spam = word_counts_spam.get(word, 0)
        count_ham = word_counts_ham.get(word, 0)
        p_w_given_s = (count_spam + 1) / (total_words_spam + vocab_size)
        p_w_given_h = (count_ham + 1) / (total_words_ham + vocab_size)
        word_probs[word] = (p_w_given_s, p_w_given_h)

    return p_s, p_h, word_probs

def calculate_p_s_given_w(word: str, p_s: float, p_h: float, word_probs: Dict[str, Tuple[float, float]]) -> float:
    if word not in word_probs:
        return 0.5
    p_w_given_s, p_w_given_h = word_probs[word]
    numerator = p_w_given_s * p_s
    denominator = (p_w_given_s * p_s) + (p_w_given_h * p_h)
    return numerator / denominator if denominator > 0 else 0.5

def calculate_p_s_given_text(tokens: List[str], p_s: float, p_h: float, word_probs: Dict[str, Tuple[float, float]]) -> Tuple[float, List[Tuple[str, float]]]:
    if not tokens:
        return 0.5, []

    p_s_given_wi = []
    word_predictive_powers = []

    for token in set(tokens):
        p = calculate_p_s_given_w(token, p_s, p_h, word_probs)
        epsilon = 1e-6
        p = max(min(p, 1 - epsilon), epsilon)
        p_s_given_wi.append(p)
        word_predictive_powers.append((token, p))
        print(f"Word: {token}, P(S|W): {p:.4f}")



    word_predictive_powers.sort(key=lambda x: x[1], reverse=True)
    top_3_probs = [p for _, p in word_predictive_powers[:3]]  
    avg_top_3 = sum(top_3_probs) / 3
    p_s_given_text = 0.8 * avg_top_3 + 0.2 * p_s  

    # Sort and get top predictive words
    top_predictive_words = word_predictive_powers[:3]

    return p_s_given_text, top_predictive_words

def interactive_spam_classifier(p_s: float, p_h: float, word_probs: Dict[str, Tuple[float, float]], stop_words: set, lemmatizer: WordNetLemmatizer) -> None:
    print("\n=== Clasificador de SPAM/HAM ===")
    print("Ingrese un texto para evaluar (o 'salir' para terminar):")
    
    while True:
        text = input("> ").strip()
        if text.lower() == 'salir':
            print("Saliendo del clasificador.")
            break
        
        tokens = preprocess_text(text, stop_words, lemmatizer)
        if not tokens:
            print("El texto ingresado no contiene palabras válidas después del preprocesamiento.")
            continue
        
        p_spam, top_words = calculate_p_s_given_text(tokens, p_s, p_h, word_probs)
        
        print(f"\nProbabilidad de que el texto sea SPAM: {p_spam * 100:.2f}%")
        print("Las 3 palabras con mayor poder predictivo para SPAM:")
        for word, prob in top_words:
            print(f"- {word}: {prob * 100:.2f}%")
        print()

def main():
    setup_nltk()
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english')) 

    try:
        df = pd.read_csv("spam_ham.csv", encoding='latin1', sep=';')
        df.columns = ["Label", "SMS_TEXT"]
    except Exception as e:
        print(f"Error al cargar el archivo CSV: {e}")
        return

    df.dropna(subset=["SMS_TEXT"], inplace=True)
    train_size = int(0.8 * len(df))
    df_train = df[:train_size]

    print("Entrenando el modelo bayesiano...")
    p_s, p_h, word_probs = train_bayesian_model(df_train, stop_words, lemmatizer)
    print(f"Probabilidad previa de SPAM (P(S)): {p_s * 100:.2f}%")
    print(f"Probabilidad previa de HAM (P(H)): {p_h * 100:.2f}%")
    print(f"Vocabulario construido con {len(word_probs)} palabras.")

    interactive_spam_classifier(p_s, p_h, word_probs, stop_words, lemmatizer)

if __name__ == "__main__":
    main()
