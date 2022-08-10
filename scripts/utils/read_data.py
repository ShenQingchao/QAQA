from utils.my_logger import logger
from utils.preprocess import extract_title_from_context
from QAUtil.qa import QA, QAList

import numpy as np
import pandas as pd


def load_qa(path, end_id=9999999, remove_redundancy_context=False):
    data = pd.read_csv(path, sep='\t', header=None, names=['q_c', 'a'])
    qa_list_obj = QAList()
    before_context = ""
    # print(len(data))
    for i in range(len(data)):
        if i >= end_id:  # the max length of the dev-set is squad2_dev(11873)
            break
        question_context = data['q_c'][i]
        question = question_context.split("\\n")[0]
        context_title = question_context.split("\\n")[1]
        # print(i, question_context, context_title)
        title, context = extract_title_from_context(context_title)
        if remove_redundancy_context and before_context == context:
            # print("redundant context...")
            continue
        answer = data['a'][i]
        qa = QA(question, answer, context, title)
        qa_list_obj.append_(qa)
        before_context = context
    logger.info(f"qa number: {len(data)}")
    qa_list_obj.set_all_context_embedding()
    return qa_list_obj


if __name__ == '__main__':
    qa_list_obj = load_qa('C:\\Users\\qingc\\Desktop\\QATesting\\dataset\\boolq_dev.tsv')
    qa_list_obj = load_qa('C:\\Users\\qingc\\Desktop\\QATesting\\dataset\\boolq_dev.tsv')
    import copy
    qa_list_obj2 = copy.deepcopy(qa_list_obj)
    for item in qa_list_obj2.qa_list:
        print(item.id)
    print(np.shape(qa_list_obj.qa_list[0].C_Embedding))

