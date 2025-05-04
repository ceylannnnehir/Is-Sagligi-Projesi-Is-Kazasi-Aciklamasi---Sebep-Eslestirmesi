from gensim.models import Word2Vec

# Test edilecek kelime ve parametreler
keyword = 'ladder'  # Önemli kelime
types = ['lemmatized', 'stemmed']
architectures = ['cbow', 'skipgram']
windows = [2, 4]
dims = [100, 300]

results = {}

# Test işlemi
for t in types:
    for arch in architectures:
        for win in windows:
            for dim in dims:
                model_name = f"{t}_model_{arch}_window{win}_dim{dim}"
                try:
                    model = Word2Vec.load(f"{model_name}.model")
                    results[model_name] = {}
                    if keyword in model.wv.index_to_key:  # Kelimenin modelde olup olmadığını kontrol et
                        similar = model.wv.most_similar(keyword, topn=5)
                        results[model_name][keyword] = similar
                    else:
                        results[model_name][keyword] = "Kelime modelde bulunamadı"
                except Exception as e:
                    print(f"Model yüklenemedi: {model_name} - {e}")
                    results[model_name] = "Model yüklenemedi"

# Sonuçları yazdır

for i, (model_name, similar_words) in enumerate(results.items(), start=1):
    print(f"Model [{i}]: {model_name}")
    print(f"Benzer Kelimeler:{similar_words}\n")
