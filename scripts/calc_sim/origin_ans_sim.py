import spacy
import gensim
from numpy import mean
nlp = spacy.load('en_core_web_sm')
from nltk.corpus import stopwords


stop_words = set(stopwords.words('english'))
model = gensim.models.KeyedVectors.load_word2vec_format("/share_container/pycharmProjects/QATesting2/baseline/wiki-news-300d-1M-subword.vec")


def ase21_answer_is_equal(target_answer, model_answer, *args):
    target_answer = target_answer.lower()
    model_answer = model_answer.lower()

    doc_target_answer = nlp(target_answer)
    str_tokens_question = [token.text for token in doc_target_answer if token.text != ""]
    doc_model_answer = nlp(model_answer)
    str_model_answer = [token.text for token in doc_model_answer if token.text != ""]
    list1_0 = str_tokens_question
    list2_0 = str_model_answer
    list1 = [w for w in list1_0 if not w in stop_words]
    list2 = [w for w in list2_0 if not w in stop_words]
    score = []
    # word_vectors = model.word_vec
    if list2 == [] or list1 == []:
        score = [0.0]
        average = 0.0
        return average
    for word1 in list1:
        if word1 in model.key_to_index:
            a_list_score = []
            for word2 in list2:
                if word2 in model.key_to_index:
                    similar = model.similarity(word1, word2)
                else:
                    if word1 in word2:
                        similar = 1.0
                    else:
                        similar = int(word1 == word2)
                a_list_score.append(similar)
            score.append(max(a_list_score))
        else:
            a_list_score = []
            for word2 in list2:
                if word1 in word2:
                    similar = 1.0
                else:
                    similar = int(word1 == word2)
                a_list_score.append(similar)
            score.append(max(a_list_score))
    average = mean(score)
    return average
