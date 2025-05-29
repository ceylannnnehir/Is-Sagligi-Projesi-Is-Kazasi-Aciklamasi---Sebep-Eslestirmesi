import pandas as pd

# Jaccard benzerlik hesaplayıcısı
def jaccard_similarity(list1, list2):
    set1, set2 = set(list1), set(list2)
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return intersection / union if union != 0 else 0

# Skor puanlama sistemi (kendi aralığına göre)
def similarity_score(jaccard_value):
    if jaccard_value >= 0.95:
        return 5
    elif jaccard_value >= 0.75:
        return 4
    elif jaccard_value >= 0.60:
        return 3
    elif jaccard_value >= 0.45:
        return 2
    else:
        return 1

# Top 5 döküman listeleri
top5_dict = {
    "tfidf_lemmatized": ["doc281", "doc158", "doc131", "doc31", "doc41"],
    "tfidf_stemmed": ["doc115", "doc281", "doc107", "doc10", "doc158"],
    "lemm_cbow_w2_d100": ["doc7", "doc281", "doc158", "doc131", "doc41"],
    "lemm_skip_w2_d100": ["doc7", "doc281", "doc158", "doc131", "doc41"],
    "lemm_cbow_w4_d100": ["doc7", "doc281", "doc158", "doc131", "doc41"],
    "lemm_skip_w4_d100": ["doc7", "doc281", "doc158", "doc131", "doc41"],
    "lemm_cbow_w2_d300": ["doc7", "doc281", "doc158", "doc131", "doc31"],
    "lemm_skip_w2_d300": ["doc7", "doc281", "doc158", "doc131", "doc210"],
    "lemm_cbow_w4_d300": ["doc7", "doc281", "doc158", "doc131", "doc210"],
    "lemm_skip_w4_d300": ["doc7", "doc281", "doc158", "doc210", "doc131"],
    "stem_cbow_w2_d100": ["doc7", "doc115", "doc107", "doc281", "doc10"],
    "stem_skip_w2_d100": ["doc7", "doc115", "doc107", "doc281", "doc10"],
    "stem_cbow_w4_d100": ["doc7", "doc115", "doc107", "doc281", "doc10"],
    "stem_skip_w4_d100": ["doc7", "doc115", "doc107", "doc281", "doc10"],
    "stem_cbow_w2_d300": ["doc7", "doc115", "doc281", "doc107", "doc10"],
    "stem_skip_w2_d300": ["doc7", "doc115", "doc281", "doc107", "doc10"],
    "stem_cbow_w4_d300": ["doc7", "doc115", "doc281", "doc107", "doc10"],
    "stem_skip_w4_d300": ["doc7", "doc115", "doc281", "doc107", "doc10"],
}

models = list(top5_dict.keys())

# Boş matrisleri oluştur
jaccard_matrix = pd.DataFrame(index=models, columns=models)
score_matrix = pd.DataFrame(index=models, columns=models)

# Hesaplama
for model1 in models:
    for model2 in models:
        jaccard_val = jaccard_similarity(top5_dict[model1], top5_dict[model2])
        jaccard_matrix.loc[model1, model2] = round(jaccard_val, 2)
        score_matrix.loc[model1, model2] = similarity_score(jaccard_val)

# Terminal çıktısı
print("\n Jaccard Benzerlik Matrisi:\n")
print(jaccard_matrix)


# CSV olarak kaydet
jaccard_matrix.to_csv("jaccard_benzerlik_matrisi.csv", encoding="utf-8-sig", sep=';')


print("\nDosyalar başarıyla kaydedildi: jaccard_benzerlik_matrisi.csv ")
