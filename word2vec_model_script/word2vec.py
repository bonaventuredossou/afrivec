# =============================================================================================================== #
# Implementation of the paper: AfriVEC: Word Embedding Models for African Languages. Case Study of Fon and Nobiin #
# Accepted at Africa NLP, EACL 2021                                                                               #  
# Authors: Bonaventure F. P. Dossou and Mohammed Sabry                                                            #
# Training a Word2Vec model                                                                                       #
# =============================================================================================================== #

import random
from gensim.models import Word2Vec
from matplotlib import pyplot
from sklearn.decomposition import PCA

random.seed(42)
def load_doc(filename):
    file = open(filename, mode='rt', encoding='utf-8')
    text = file.read().split("\n")
    file.close()
    return text

def save_list(lines, filename):
    data = '\n'.join(i[0] + "\t" + str(i[1]) for i in lines)
    file = open(filename, 'w', encoding="utf-8")
    file.write(data)
    file.close()

path = "/path/to/corpus.txt" # dataset/fon_family_embedding.txt

sentences = load_doc(path)
sentences = [_.split() for _ in sentences]

# train model
model = Word2Vec(sentences, min_count=5, alpha=0.5) # paremeters could be fine-tuned
# save model
model.save('/path/to/model') # models/fon_family_words_model_1.bin
# load model
new_model_best = Word2Vec.load('/path/to/model')

X = new_model_best[new_model_best.wv.vocab]
pca = PCA(n_components=2)
result = pca.fit_transform(X)
# create a scatter plot of the projection
pyplot.scatter(result[:, 0], result[:, 1])
words = list(set(new_model_best.wv.vocab))
for i, word in enumerate(words):
   pyplot.annotate(word, xy=(result[i, 0], result[i, 1]))
pyplot.title("Visualization of Fon Word Embedding - Word Vectors Using PCA")
pyplot.savefig("family_word_embedding.png")
pyplot.show()

#examples of further analysis on word vec 
result2 = new_model_best.most_similar(positive=["tɔ"], negative=["nɔ"], topn=5)
save_list(result2, "/path/to/results.txt")
