from utils.my_logger import logger
from collections import deque
import random
from utils.GA2S import GA2S
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer
from utils.coreference_eliminate import eliminate_coreference

model = SentenceTransformer('sentence-transformers/multi-qa-MiniLM-L6-cos-v1')  #../3rd_models/multi-qa-MiniLM-L6-cos-v1


def _remove_brackets(text):
    dq = deque()
    brackets_list = []
    for i, c in enumerate(text):
        if c == '(':
            dq.append(i)
        elif c == ')' and len(dq) > 0:  # i) ..., for fear this case.
            l_pos = dq.pop()
            if l_pos != 0:
                l_pos -= 1
            if len(dq) == 0:
                brackets = text[l_pos: i + 1]
                brackets_list.append(brackets)
    logger.debug(f"brackets_list in this context is: {brackets_list}")
    for bracket in brackets_list:
        text = text.replace(bracket, '')
    return text


def context_preprocess(text):
    # 1. remove adv/conjunction/interjection/coordinating conjunction  in the begin of the sentence.
    first_word = text.split(" ")[0]
    tag_words = parse_part_of_speech(text)
    if tag_words[0][1] in ["RB", "CNJ", "UH", "CC", ]:
        text = text[len(first_word):].strip()
    text = text.lower()
    # 1. ----------remove (...) in text----------------
    new_text = _remove_brackets(text)
    # conjunctions_list = ["however", "but", "otherwise", "whereas", "besides", "in addition", "even so",
    #                      "thus", "hence", "therefore",  "furthermore", "moreover", "what's more", "anyway", "usually",
    #                      "also", "additionally", "specifically",  "nevertheless", "firstly", "secondly", "thirdly"]
    # for conj in conjunctions_list:
    #     if new_text.startswith(conj+","):
    #         new_text = new_text[len(conj)+1:].strip()
    return new_text


def extra_question_preprocess(question):
    # remove the condition of the question; such as 'if', 'when', thus meaning will change behind regardless of ...
    yes_no_list = ["am", "is", "are", "was", "were", "do", "does", "did", "have", "has", "can", "could", 'would', 'will']
    question = question.lower()
    if ', ' in question:
        if not question.startswith("wh") and not question.startswith("how"):
            flag = False
            temp = question.split(', ')
            for sep in temp:
                if ('wh' in sep or 'how' in sep) and len(sep) > 15:
                    question = sep
                    flag = True
                    break
            if not flag:
                for v in yes_no_list:
                    for sep in temp:
                        if sep.startswith(v):
                            if " or " in sep:  # alternative question
                                question = sep
                            elif " or " in question:  # alternative question
                                # question = question
                                pass
                            else:  # not alternative question
                                question = sep
                            sent = GA2S(question).strip()[:-1]
                            question = random.choice(["if ", "whether "]) + sent
                            print(question)

        elif question.startswith("when"):  # for example: when rurik past away, who took over?
            temp = question.split(', ')
            if len(temp) == 2 and ('wh' in temp[1] or 'how' in temp[1]):  # 'when' means "under the condition"
                question = temp[1]
    if question.startswith('and ') or question.startswith("but "):
        question = question[4:]
    if not question.strip().endswith("?"):
        question = question + "?"
    return question


def get_prototype_word(word):
    """
    :param word: a single word
    :return:  the prototype of the input word.
    """
    word = word.lower()
    wnl = WordNetLemmatizer()
    prototype = wnl.lemmatize(word, 'n')
    # logging.debug(f"origin word:{word}, the prototype word is {prototype}")
    return prototype


def remove_stop_words(sentence):
    res_list = []
    word_list = sentence.strip().split(' ')
    for word in word_list:
        if word not in stopwords.words("english"):
            res_list.append(word)
    return res_list


def remove_first_stopwords(phrase):
    first_word = phrase.lower().split(' ')[0]
    if first_word in stopwords.words('english'):
        phrase = phrase[1 + len(first_word):]
    return phrase


def parse_part_of_speech(sentence):
    tokens = nltk.word_tokenize(sentence)  # split the sentence to single word.
    tagged_sent = nltk.pos_tag(tokens)  # part of the speech analysis
    return tagged_sent


def word_embedding(my_str):
    '''
    :param my_str: string or list_string
    :return:
    '''

    w_embedding = model.encode(my_str)
    return w_embedding


def is_simple_sentence(sent):
    if ", " in sent or "; " in sent:  # "; " may contain candidate answers which causes false positive.
        return False
    return True


def before_preprocess(sent):
    sent = sent.replace(", however, ", " ")
    sent = sent.replace(", but, ", " ")
    conjunctions_list = ["however", "but", "otherwise", "whereas", "besides", "in addition", "even so",
                         "thus", "hence", "therefore", "furthermore", "moreover", "what's more", "anyway", "usually",
                         "also", "additionally", "specifically", "nevertheless", "firstly", "secondly", "thirdly"]
    for conj in conjunctions_list:
        if sent.lower().startswith(conj + ","):
            sent = sent[len(conj) + 1:].strip()
    return sent


def get_sentence_from_a_context(context):
    context = eliminate_coreference(context)
    res_sent_list = []
    context = context.strip()
    context = context.replace("(i.e. ", "(")  # this rule to avoid wrong sentence split by sent_tokenize().
    sent_list = sent_tokenize(context)
    for s in sent_list:
        s = s.strip()
        s = before_preprocess(s)
        if s.strip().endswith("?"):  # remove question in context.
            continue
        if not is_simple_sentence(s):  # skip complex sentence.
            continue
        if s.endswith("; "):
            s = s[:-2] + "."
        res_sent_list.append(s)
    return res_sent_list


def get_sentences_from_contexts(context_list):
    all_sentence = []
    for context in context_list:
        sentences = get_sentence_from_a_context(context)
        all_sentence.extend(sentences)
    return all_sentence


def extract_title_from_context(context_title):
    context_title = context_title.strip()
    if not context_title.startswith('('):
        logger.debug(f"This context not have a title: {context_title}")
        return "", context_title
    queue_flag = 0
    for i in range(len(context_title) - 1):
        if context_title[i] == '(':
            queue_flag += 1
        elif context_title[i] == ')':
            queue_flag -= 1
            if queue_flag == 0:
                c_ = context_title[i + 1:].strip()
                title = context_title[1:i]
                return title, c_
                break
    logger.debug(f"This context have a wrong title: {context_title}")
    return "", context_title


def context2sentences(context_list):
    sentence_list = []
    for context in context_list:
        temp_ = sent_tokenize(context.strip())
        sentence_list.extend(temp_)
    return sentence_list


if __name__ == '__main__':
    c = "The ball becomes dead when."

    res = get_sentences_from_contexts([c])
    print(res)
    res = context_preprocess(res[0])
    print(res)
