import preProcessing
import globals as g
import nltk
from gensim.models import Word2Vec
import numpy as np

# uncomment these lines to recalculate vectors

# data = []
#
# for article in g.cleaned_text_train:
#     data.append(nltk.word_tokenize(article))
#
# model = Word2Vec(data, window=5, min_count=1)
# X = model[model.wv.vocab]
#
# a = np.array(X)
# np.savetxt('vectors2.txt', a, fmt='%f')

vectors = np.loadtxt('vectors.txt', dtype=float)
g.vectors = vectors
rev_vec = []
for i in range(len(g.vectors[0])):
    line = []
    for j in range(len(g.vectors)):
        line.append(g.vectors[j][i])
    rev_vec.append(line)

g.vectors = rev_vec
