from utils.my_logger import logger
from utils.preprocess import word_embedding
import numpy as np


class QA:
    count = 0

    def __init__(self, q, a, c, t=''):
        self.id = QA.count  # identify the line number of the qa
        QA.count += 1
        self.Q = q
        self.A = a
        # self.C = context_preprocess(c)
        self.C = c
        self.T = t
        self.C_Embedding = None
        self.PA = None
        # self.Q2S = q2s


class QAList:
    def __init__(self):
        self.qa_list = []
        QA.count = 0

    def append_(self, qa):
        self.qa_list.append(qa)

    def remove_qa_of_this_context(self, context):
        for qa in self.qa_list:
            if context == qa.C:
                self.qa_list[qa.id].C = ''
                self.qa_list[qa.id].C_Embedding = np.zeros(shape=(384,), dtype='float32')
                logger.debug(f"line {qa.id+1} in training-set has the same context with current context:{context} ")

    def set_all_context_embedding(self):
        contexts_all = []
        for qa in self.qa_list:
            contexts_all.append(qa.C)
        context_all_embedding = word_embedding(contexts_all)
        for id in range(len(self.qa_list)):
            self.qa_list[id].C_Embedding = context_all_embedding[id]

    def get_all_context_embedding(self):
        context_embedding_list = []
        for qa in self.qa_list:
            context_embedding_list.append((qa.C, qa.C_Embedding))
        return np.array(context_embedding_list)  # return np.array()
