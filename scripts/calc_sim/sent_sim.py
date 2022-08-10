from sentence_transformers import util
from sentence_transformers import SentenceTransformer

sbert_model = SentenceTransformer('sentence-transformers/multi-qa-MiniLM-L6-cos-v1')  # ../3rd_models/multi-qa-MiniLM-L6-cos-v1


def calc_sim(sentence1, sentence2):
    w_embedding1 = sbert_model.encode(sentence1)
    w_embedding2 = sbert_model.encode(sentence2)
    sim = util.dot_score(w_embedding1, w_embedding2).tolist()[0][0]
    return sim


def calculate_sim_origin_sentence(origin, target_list, top_n):
    '''
    This method can calculate the similarity between origin and target.
    :param origin:  a word or a sentence.
    :param target_list:  a list containing words or sentences.
    :param top_n:  the number of return list.
    :param context_embedding: the embedding of target_list.
    :return: 2-dim list  --- the top_n most similar target words/sentences for origin; e.g., [('dog', 0.9), ('cat', 0.85)]
    '''
    sim_dict = {}
    # model = SentenceTransformer('../pretrained-models/multi-qa-MiniLM-L6-cos-v1')   # bert-base-nli-stsb-mean-tokens , multi-qa-MiniLM-L6-cos-v1, paraphrase-MiniLM-L6-v2
    w_embedding = sbert_model.encode(origin)

    context_embedding = sbert_model.encode(target_list)
    sim = util.dot_score(w_embedding, context_embedding).tolist()[0]
    # dict: {sent: sim_value}: first, load sent and score to a dict; and then rank by score--> convert to list
    for i, score in enumerate(sim):
        sim_dict[target_list[i]] = score
    res_context_score_list = sorted(sim_dict.items(), key=lambda x: x[1], reverse=True)  # dict to 2-dim list
    if not top_n:
        top_n = len(target_list)
    return res_context_score_list[:top_n]


def calculate_sim_origin_target(origin, target_obj, top_n):
    sim_dict = {}
    contexts_embedding = target_obj.get_all_context_embedding()
    contexts = list(contexts_embedding[:, 0])
    c_embedding = list(contexts_embedding[:, 1])
    # model = SentenceTransformer('../pretrained-models/multi-qa-MiniLM-L6-cos-v1')   # bert-base-nli-stsb-mean-tokens , multi-qa-MiniLM-L6-cos-v1, paraphrase-MiniLM-L6-v2
    w_embedding = sbert_model.encode(origin)
    sim = util.dot_score(w_embedding, c_embedding).tolist()[0]
    # return dict, <id, sim_value> # first: load qa_id and score to a dict, and then rank by score
    for i, score in enumerate(sim):
        sim_dict[i] = score
    res_context_score_list = sorted(sim_dict.items(), key=lambda x: x[1], reverse=True)
    # for item in res_context_score_list[:top_n]:  # only print the top_n related context
    #     print(item[1], item[0])
    if not top_n:
        top_n = len(contexts)
    return res_context_score_list[:top_n]


if __name__ == '__main__':
    # from utils.read_data import load_qa
    # qa_list_obj_test = load_qa('/share_container/data/unifiedqa/boolq_dev.tsv')
    res = calculate_sim_origin_sentence("When the IPCC was fund?",
                                      ['IPCC was fund when i was a child', 'in 1988', 'hello word', 'When i was a child'], 10)
    for item in res:
        print(item)
