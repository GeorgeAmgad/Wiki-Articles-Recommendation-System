import nltk
import math
import globals as g

data = []

for article in g.cleaned_text_train:
    data.append(nltk.word_tokenize(article))

DF = {}
for i in range(len(data)):
    tokens = data[i]
    for w in tokens:
        # noinspection PyBroadException
        try:
            DF[w].add(i)
        except Exception:
            DF[w] = {i}

DF_count = []
for i in DF:
    DF_count.append(len(DF[i]))

total_vocab = [x for x in DF]
num_of_words = len(total_vocab)

# this 2-D array contains the TF_IDF calculations
# X axis = all unique vocabs in all documents
# Y axis = the documents
# so intersection defines -> the TF_IDF score of a word in a certain document
tf_idf = []

for i in range(len(data)):
    tf_idf_doc = []
    n_of_words = len(data[i])
    words = data[i]
    for word in DF.items():
        # check the keys in dict. if it doesn't exist in the doc continue
        if i not in word[1]:
            tf_idf_doc.append(0)
            continue
        else:
            n_of_occur = data[i].count(word[0])
            tf_idf_doc.append((n_of_occur / n_of_words) * (math.log10(n_of_words / float(len(word[1])))))
    tf_idf.append(tf_idf_doc)
