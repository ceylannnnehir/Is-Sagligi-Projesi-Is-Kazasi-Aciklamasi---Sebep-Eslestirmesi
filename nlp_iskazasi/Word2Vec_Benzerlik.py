from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import os
import csv
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
import pandas as pd

# NLTK verileri indirmeniz gerekebilir (ilk sefer için):
# import nltk
# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('wordnet')

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()

def preprocess(text, mode="lemma"):
    tokens = word_tokenize(text)
    tokens = [t.lower() for t in tokens if t.isalpha() and t.lower() not in stop_words]
    if mode == "lemma":
        return [lemmatizer.lemmatize(t) for t in tokens]
    else:
        return [stemmer.stem(t) for t in tokens]

def load_tokenized_corpus(csv_path, mode="lemma"):
    with open(csv_path, "r", encoding="utf-8") as f:
        raw = [row[0] for row in csv.reader(f)]
    return [preprocess(s, mode) for s in raw], raw

def avg_vector(model, tokens):
    vectors = [model.wv[word] for word in tokens if word in model.wv]
    if not vectors:
        return None
    return np.mean(vectors, axis=0).reshape(1, -1)

def word2vec_similarity_all(model_path, corpus_tokens, query_tokens):
    model = Word2Vec.load(model_path)
    query_vec = avg_vector(model, query_tokens)
    if query_vec is None:
        return []

    similarities = []
    for idx, tokens in enumerate(corpus_tokens):
        doc_vec = avg_vector(model, tokens)
        if doc_vec is not None:
            sim = cosine_similarity(query_vec, doc_vec).flatten()[0]
            similarities.append((idx, sim))
    top5 = sorted(similarities, key=lambda x: x[1], reverse=True)[:5]
    return top5

def subjective_evaluation(top5_results, raw_sentences):
    scores = []
    sim_values = []
    for idx, sim in top5_results:
        print(f"\n[Metin {idx}] {raw_sentences[idx]}")
        print(f"Benzerlik skoru: {sim:.4f}", end=' ')
        sim_values.append(round(sim, 4))
        if sim >= 0.95:
            score = 5
        elif sim >= 0.75:
            score = 4
        elif sim >= 0.60:
            score = 3
        elif sim >= 0.45:
            score = 2
        else:
            score = 1
        scores.append(score)
    return scores, sim_values, sum(scores) / len(scores)


# Model parametreleri
params = [
    ("cbow", 2, 100),
    ("skipgram", 2, 100),
    ("cbow", 4, 100),
    ("skipgram", 4, 100),
    ("cbow", 2, 300),
    ("skipgram", 2, 300),
    ("cbow", 4, 300),
    ("skipgram", 4, 300)
]

query_index = 7
base_names = [("lemmatized_model", "lemma"), ("stemmed_model", "stem")]

# Değerlendirme sonuçları
results = []

for base_name, mode in base_names:
    csv_path = "lemmatized_sentences.csv" if mode == "lemma" else "stemmed_sentences.csv"
    tokenized_corpus, raw_sentences = load_tokenized_corpus(csv_path, mode)
    query_tokens = tokenized_corpus[query_index]

    for model_type, window, dim in params:
        model_path = f"{base_name}_{model_type}_window{window}_dim{dim}.model"
        model_name = f"{base_name}_{model_type}_win{window}_dim{dim}"
        if os.path.exists(model_path):
            print(f"\n {model_name} modeli yükleniyor...")
            top5 = word2vec_similarity_all(model_path, tokenized_corpus, query_tokens)
            if not top5:
                print("Vektör bulunamadı, model atlanıyor.")
                continue
            scores, sim_values, avg = subjective_evaluation(top5, raw_sentences)
            doc_names = [f"doc{idx}" for idx, _ in top5]
            results.append({
                "Model Adı": model_name,
                "5 Benzer Metin": ", ".join(doc_names),
                "Benzerlik Skorları": ", ".join(map(str, sim_values)),
                "Skorlar": ", ".join(map(str, scores)),
                "Ortalama": round(avg, 2)
            })

        else:
            print(f" Model bulunamadı: {model_path}")

# Pandas ile tabloyu yazdır
df = pd.DataFrame(results)
print("\n\n SONUÇ TABLOSU:\n")
print(df.to_markdown(index=False)) 