# =============================================================================================================== #
# Implementation of the paper: AfriVEC: Word Embedding Models for African Languages. Case Study of Fon and Nobiin #
# Accepted at Africa NLP, EACL 2021                                                                               #  
# Authors: Bonaventure F. P. Dossou and Mohammed Sabry                                                            #
# Training a Poincaré Embedding model                                                                             #
# =============================================================================================================== #

# https://radimrehurek.com/gensim/models/poincare.html
# HyperLex Paper and GitHub Repository: https://github.com/cambridgeltl/hyperlex

# All imports
import random
import unicodedata
import chart_studio
import pandas as pd
import chart_studio.plotly as plt
from gensim.models.poincare import PoincareModel, LexicalEntailmentEvaluation, LinkPredictionEvaluation, ReconstructionEvaluation
from sklearn.metrics import classification_report, accuracy_score
from gensim.viz.poincare import poincare_2d_visualization

chart_studio.tools.set_credentials_file(username='username', api_key='api_key')
random.seed(42)

def load_doc(filename):
    file = open(filename, mode='rt', encoding='utf-8')
    text = file.read().split("\n")
    relations = []
    for _ in text:
        pair = _.split("\t")
        if len(pair) == 2:
            entity = normalize_diacritics_text(pair[0].lower())
            relations.append((entity, pair[1]))
        else:
            print(pair)
    file.close()
    return list(set(relations))

def save_list(lines, filename):
    data = '\n'.join(i[0] + "\t" + i[1] for i in lines)
    file = open(filename, 'w', encoding="utf8")
    file.write(data)
    file.close()

# Handling diacritics
def normalize_diacritics_text(text_string):
    return unicodedata.normalize("NFC", text_string)


path = "/path/to/training_set.txt" # poincare_dict.txt
path_lexical = "path/to/lexical.txt" # example file_lexical_fon.txt following the HyperLex Format
path_validation = "/file/to/validation_set.txt" # validation set poincare_embedding_validation.txt
relations_ = load_doc(path)

# parameters
size = 15 # dimension of the embedding space
c = 15 # constant of negative curvature
epochs = 2000 # number of training epochs
# define the model
model = PoincareModel(relations_, size=size, negative=c)
model.train(epochs)

# save the model
model.save('/path/to/model')
# save model embedding
model.kv.save_word2vec_format("/path/to/embedding")

# load the model and the embedding
model = PoincareModel.load("/path/to/model")
model.kv.load_word2vec_format("/path/to/embedding")

all_relations = set(relations_)
# add different classes to the labels to add them to the graph
labels = list(set([_[0] for _ in relations_])) + ["girl_name", "boy_name", "mixed_name", "body_part", "benin_city"]

title = "Title Figure"
fig = poincare_2d_visualization(new_model_10, all_relations, title, show_node_labels=labels)

plt.image.ishow(fig, width=1000, height=1000)
plt.image.save_as(fig, filename='path/to/plot/plot_name.png')

# Compute Link PredictionEvaluation
link = LinkPredictionEvaluation(path, path_validation, model.kv.load_word2vec_format("/path/to/embedding"))
print(link.evaluate())
# Compute RecontructionEvaluation
rec = ReconstructionEvaluation(path, model.kv.load_word2vec_format("/path/to/embedding"))
print(rec.evaluate())

# Testing on new names
lex = LexicalEntailmentEvaluation(path_lexical)
embedding = model.kv
lex_trie = lex.create_vocab_trie(embedding)

samples = list(set([_[0] for _ in relations_]))

# testing
# Please use the updated version here:
# https://github.com/bonaventuredossou/poincare_glove/blob/patch-3/gensim/models/poincare.py
testing_file = load_doc("/path/to/testing_set.txt")
real_testing = [pair for pair in testing_file if
                 pair[0] not in samples]  # making sure testing samples are not in training data

classes = list(set([_[1] for _ in real_testing if _[1] not in ['benin_city', 'body_part']]))

save_list_1(real_testing, "/path/to/testing_set.txt")

true_labels = [_[1] for _ in real_testing]
testing_samples = [_[0] for _ in real_testing]
result_dataframe = pd.DataFrame()
result_dataframe["samples"] = testing_samples
result_dataframe["true_labels_names"] = true_labels
all_predictions = []

i = 0
for sample in testing_samples:
    prediction_records = []
    for class_ in classes:
        prediction = lex.score_function(embedding, lex_trie, sample, class_)
        current_prediction = (class_, true_labels[i], prediction)
        prediction_records.append(current_prediction)
    i = +1
    sorted_prediction = sorted(prediction_records, key=lambda x: x[2], reverse=True)
    all_predictions.append(sorted_prediction[0][0])

mapping_ = {"boy_name": 0, "girl_name": 1}
result_dataframe["true_labels"] = result_dataframe["true_labels_names"].map(mapping_)
result_dataframe["predictions_names"] = all_predictions
result_dataframe["predictions"] = result_dataframe["predictions_names"].map(mapping_)
result_dataframe.to_csv("/path/to/predictions.csv", index=False)

oa = accuracy_score(list(result_dataframe["true_labels"]), list(result_dataframe["predictions"]))
print("Overall accuracy: {}".format(oa))

classRep = classification_report(list(result_dataframe["true_labels"]), list(result_dataframe["predictions"]))
print("Classification Report:\n {}".format(classRep))
