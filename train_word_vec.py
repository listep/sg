from gensim.models.word2vec import LineSentence, Word2Vec
sentences = LineSentence("/search/wts/dict/meizu_201801-05.char.txt")
print("training")
model = Word2Vec(sentences=sentences, min_count=10, window=6, workers=24, size=300, iter=6)
print("saving")
model.save("MeizuCharModel.bin")
print("Done")
