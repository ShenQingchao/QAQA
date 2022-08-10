from utils.GA2S import GA2S
from utils.preprocess import extra_question_preprocess
import random

random.seed(2022)


def opposite_answer(origin):
    if origin.strip() == 'yes':
        return 'no'
    elif origin.strip() == 'no':
        return 'yes'


def negative_question(question):
    '''
    add negative word : 'not' to the origin question.
    '''

    first_word = question.split(' ')[0]
    other_words = question[len(first_word) + 1:]
    if first_word == 'will':
        first_word_negative = "won't"
    elif first_word == "shall":
        first_word_negative = "shan't"
    elif first_word == "can":
        first_word_negative = "can't"
    else:
        first_word_negative = first_word + "n't"
    new_question = first_word_negative + ' ' + other_words
    return new_question


def add_extra2question(question, extra_sent):
    question = question.strip().lower()
    extra_sent = extra_sent.lower()
    # question = negative_question(question) if random.randint(0, 1) == 1 else question  # add negative attack randomly!
    extra_sent = extra_sent[:-1] if extra_sent.endswith('.') else extra_sent
    extra2_question_template_list = ["I have known that " + extra_sent + ', ' + question,  # 680
                                     "I heard about that " + extra_sent + ', ' + question,  # 687
                                     "It is said that " + extra_sent + ', ' + question,  # 696
                                     "Someone told me that " + extra_sent + ', ' + question,  # 706
                                     "I've heard a whisper that " + extra_sent + ', ' + question,  # 707
                                     ]
    # the above is add redundant 'sentence', the below is add redundant 'question'!!!
    '''
    "I'm not sure if " + extra_sent + ', but' + question,  # 787
    "I'm not sure whether " + extra_sent + ', but' + question,  # 829

    "I do not care if " + extra_sent + ', but ' + question,  # 661
    "Nobody cares if " + extra_sent + ', but ' + question,  # 672
    # "No one cares if " + extra_sent + ', but ' + question,  # 655

    "Put aside if " + extra_sent + ', ' + question,  # 668
    "Regardless of if " + extra_sent + ", " + question,  # 669

    "It doesn't matter if " + extra_sent + ", " + question,  # 697
    question[:-1] + " regardless of whether " + extra_sent + " or not?",  # 830
    '''
    new_question = random.choice(extra2_question_template_list)
    # new_question = extra2_question_template_list  # template evaluate
    return new_question


def get_question_as_redundancy_template_list(question, extra_question):
    question = question.strip().lower()
    extra_question = extra_question.lower().strip()
    extra_question = extra_question[:-1] if extra_question.endswith("?") else extra_question
    question_as_redundancy_template_list = [
        "I'm not sure " + extra_question + ', but ' + question,
        "I do not care " + extra_question + ', but ' + question,
        "Nobody cares " + extra_question + ', but ' + question,
        "It doesn't matter " + extra_question + ", but " + question,

        # the following templates need simple wh-question.
        "Put aside " + extra_question + ', ' + question,
        "Regardless of " + extra_question + ", " + question,

        question[:-1] + " regardless of " + extra_question + "?",
        question[:-1] + " without considering " + extra_question + "?",
        question[:-1] + " taking no account of " + extra_question + "?",
    ]
    new_question = random.choice(question_as_redundancy_template_list)
    return new_question


def add_extra2context(context, extra_sent):
    # new_context = context + ' ' + extra_sent
    new_context = extra_sent + ' ' + context
    return new_context


def combine2input(obj1, obj2, ans1, ans2):
    selected_template_id = random.randint(1, 2)  # 2 : the template number
    s1 = GA2S(obj1.Q).lower().strip()[:-1]
    s2 = GA2S(obj2.Q).lower().strip()[:-1]
    if selected_template_id == 1:
        new_question = "Is it true that " + s1 + ' and ' + s2 + "?"
    elif selected_template_id == 2:
        new_question = "Isn't it true that " + s1 + ' and ' + s2 + "?"
    else:
        raise Exception(print("the template doesn't exist!"))
    new_answer = 'yes' if (ans1 == 'yes' and ans2 == 'yes') else 'no'
    new_context = obj2.C + " " + obj1.C  # put the second context in the head.
    return new_question, new_context, new_answer


def combine2input_template_eva(obj1, obj2, ans1, ans2):
    s1 = GA2S(obj1.Q).lower().strip()[:-1]
    s2 = GA2S(obj2.Q).lower().strip()[:-1]

    new_question = "Is it true that " + s1 + ' and ' + s2 + "?"
    new_question2 = "Isn't it true that " + s1 + ' and ' + s2 + "?"
    new_question_list = [new_question, new_question2]
    new_answer = 'yes' if (ans1 == 'yes' and ans2 == 'yes') else 'no'
    new_context = obj2.C + " " + obj1.C  # put the second context in the head.
    return new_question_list, new_context, new_answer


def add_input_as_redundancy(obj1, obj2):
    s2 = GA2S(obj2.Q).lower().strip()[:-1]
    extra_question = random.choice(["if ", "whether "]) + s2
    new_question = get_question_as_redundancy_template_list(question=obj1.Q, extra_question=extra_question)
    new_context = obj2.C + " " + obj1.C
    return new_question, new_context


def add_wh_question_as_redundancy(obj1, obj2):
    extra_question = extra_question_preprocess(obj2.Q) if ', ' in obj2.Q else obj2.Q  # remove the condition of question
    new_question = get_question_as_redundancy_template_list(question=obj1.Q, extra_question=extra_question)
    new_context = obj2.C + " " + obj1.C
    return new_question, new_context


if __name__ == '__main__':
    str = 'is this the last year for once upon a time?'
    res = negative_question(str)
    print(res)
