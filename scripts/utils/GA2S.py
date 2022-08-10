import spacy
from nltk import Tree
import string
from benepar.spacy_plugin import BeneparComponent
from pattern.en import conjugate, lemma, lexeme, PRESENT, INFINITIVE, PAST, FUTURE, SG, PLURAL, PROGRESSIVE

boolean_set = {"be", "do", "will", "can", "should", "may", "have", "must", "would", "am", "could", "shell", "might"}
WH_set = {"how", "what", "who", "why", "whose", "where", "when", "which"}


def order(child_list, sent_list):
    order_list = []
    start_plc = 0
    for part in sent_list:
        if part in [["that"], ["which"], ["who"]]:
            start_plc = sent_list.index(part)
    if start_plc == 0:
        return [], 0, 0
    num = 0
    for part in sent_list[start_plc:]:
        if len(order_list) == len(child_list):
            return order_list, start_plc, num
        if len(child_list) > len(sent_list)-start_plc:
            return [], 0, 0
        order_list.extend(part)
        num += 1
    return [], 0, 0


def child_to_list_node(node, tree):
    if node.n_lefts + node.n_rights > 0:
        for child in node.children:
            tree.insert(0, child)


def tree_to_list_str(node, tree):
    if node.n_lefts + node.n_rights > 0:
        tree.insert(0, node.orth_)
        return [tree_to_list_str(child, tree) for child in node.children]
    else:
        tree.insert(0, node.orth_)

def tense(word, nlp):
    word_ = nlp(word)
    word_lemma = word_[0].lemma_
    word_did = conjugate(word_lemma, tense=PAST)
    word_does = conjugate(word_lemma, tense=PRESENT, number=SG)
    if word == word_did:
        return 1
    elif word == word_does:
        return 2
    else:
        return 3

def list_to_str(a_list):
    punc = string.punctuation
    special = ["-", "/"]
    front_special = False
    str_out = ""
    num = -1
    for i in a_list:
        num += 1
        if num == 0:
            if i not in punc:
                str_out = str_out + i
            else:
                str_out = str_out + i
        else:
            if i == "'s":
                str_out = str_out + i
            elif i not in punc:
                if front_special:
                    str_out = str_out + i
                    front_special = False
                else:
                    str_out = str_out + " " + i
            else:
                if i in special:
                    front_special = True
                str_out = str_out + i
    return str_out

def list_to_str2(a_list):
    str_out = ""
    for i in a_list:
        for j in i:
            str_out = str_out + " " + j
    return str_out[1:]

def shuchu(node, list):
    if isinstance(node, str):
        children = []
        children.append(node)
        list.append(children)
    else:
        if_print = True
        for child in node:
            if len(child) != 1 or not isinstance(child[0], str):
                if_print = False
        if if_print:
            children = []
            for i in node:
                children.append(i[0])
            list.append(children)
        else:
            for i in node:
                shuchu(i, list)


def is_noun_from_S2W(list_1, pos_list, str_list):
    if_noun = True
    num = -1
    if list_1 == ["the"] or list_1 == ["The"]:
        return True
    if list_1[0] == "the" or list_1[0] == "The":
        return True
    for i in list_1:
        num += 1
        if '-' in i and i!= "-":
            continue
        if i not in str_list:
            continue
        plc = str_list.index(i)
        pos = pos_list[plc]
        if num == len(list_1)-1:
            if i != "." and pos == "PUNCT":
                continue
        if pos not in ["NOUN", "PROPN", "PRON", "ADJ", "ADJP"] and i not in ["'s", "'", "s", "-", "I", "IV", "VI", "well"] and i[-3:] != "ing":
            if_noun = False
    if "-" in list_1:
        if_noun = True
    return if_noun


def is_noun(list_0, list_1, pos_list):
    if_noun = True
    num = -1
    for i in list_1:
        num += 1
        if pos_list[len(list_0)+1+num] not in ["NOUN", "PROPN", "PRON"]:
            if_noun = False
    return if_noun


def is_adj(list_1, pos_list, str_list):
    if_adj = True
    if list_1 == ["by"]:
        return False
    num = -1
    if list_1[0] == "the" or list_1[0] == "The":
        num = 0
        for i in list_1[1:]:
            num += 1
            if '-' in i and i != "-":
                continue
            plc = str_list.index(i)
            pos = pos_list[plc]
            if pos in ["NOUN", "PROPN", "PRON"]:
                return False
        return True
    for i in list_1:
        num += 1
        if '-' in i and i != "-":
            continue
        if i not in str_list:
            continue
        plc = str_list.index(i)
        pos = pos_list[plc]
        if pos not in ["ADJ", "ADJP", "DET", "CCONJ"] and i not in ["'s", "'", "s", "a", "an", "A", "-", "I", "IV", "VI", "well"] and i[-3:] != "ing":
            if_adj = False
    return if_adj


def statement_not_boolean(question, nlp):
    question = question.replace("  ", " ")
    if question[-1] == " ":
        question = question[:-1]
    if question[-2:-1] == ".?":
        question = question[:-2] + "?"
    elif question[-2:-1] == "??":
        question = question[:-1]
    elif question[-1] != "?":
        question = question + "?"
    doc_question = nlp(question)

    str_tokens_question = [token.text.strip() for token in doc_question if token.text.strip() != ""]  # replace string with text
    str_tokens_question[0] = str_tokens_question[0].lower()

    final = []
    final.extend(str_tokens_question)
    final.pop()
    final_out = list_to_str(final)
    final_out = final_out[0].upper() + final_out[1:] + "."
    return final_out


def boolean(question, nlp):
    question = question.replace("  ", " ")
    if question[-1] == " ":
        question = question[:-1]
    if question[-2:-1] == ".?":
        question = question[:-2] + "?"
    elif question[-2:-1] == "??":
        question = question[:-1]
    elif question[-1] != "?":
        question = question + "?"
    doc_question = nlp(question)
    pos_tokens_question = [token.pos_ for token in doc_question if token.text.strip() != ""]
    str_tokens_question = [token.text.strip() for token in doc_question if token.text.strip() != ""]
    str_tokens_question[0] = str_tokens_question[0].lower()

    first_vb = str_tokens_question[0].lower()
    new_question = list_to_str(str_tokens_question[1:])
    doc_new_question = nlp(new_question)
    tokens_new_question = [token for token in doc_new_question if token.text.strip() != ""]
    dep_tokens_new_question = [token.dep_ for token in doc_new_question if token.text.strip() != ""]
    tag_tokens_new_question = [token.tag_ for token in doc_new_question if token.text.strip() != ""]
    pos_tokens_new_question = [token.pos_ for token in doc_new_question if token.text.strip() != ""]
    str_tokens_new_question = [token.text.strip() for token in doc_new_question if token.text.strip() != ""]
    root = str_tokens_new_question[dep_tokens_new_question.index("ROOT")]
    root_plc = dep_tokens_new_question.index("ROOT")
    if "VB" not in tag_tokens_new_question[root_plc]:
        have_vb = False
        for i in tag_tokens_new_question:
            if "VB" in i:
                root_plc = tag_tokens_new_question.index(i)
                root = str_tokens_new_question[root_plc]
                break
    if str_tokens_question[0].lower() in ["do", "did", "does"]:
        this_tense = 0
        if first_vb == "do":
            this_tense = 1
        elif first_vb == "does":
            this_tense = 2
        elif first_vb == "did":
            this_tense = 3
        this_verb = ""
        if this_tense == 1 or this_tense == 0:
            this_verb = str_tokens_new_question[root_plc]
        if this_tense == 2:
            this_verb = conjugate(tokens_new_question[root_plc].lemma_, tense=PRESENT, number=SG)
        if this_tense == 3:
            # print(tokens_new_question[root_plc], tokens_new_question[root_plc])
            this_verb = conjugate(tokens_new_question[root_plc].lemma_, tense=PAST)
        final = []
        final.extend(str_tokens_new_question[:root_plc])
        final.extend([this_verb])
        final.extend(str_tokens_new_question[root_plc + 1:])
        final.pop()
        final_out = list_to_str(final)
        final_out = final_out[0].upper() + final_out[1:] + "."
        return final_out
    elif str_tokens_question[0].lower() in ["am", "is", "are", "was", "were"]:
        str_tokens_question[0] = str_tokens_question[0].lower()
        if "the same" in question:
            vb = str_tokens_question[0]
            same_plc = str_tokens_question.index("same")
            final = []
            final.extend(str_tokens_question[1:same_plc-1])
            final.extend([vb])
            final.extend(str_tokens_question[same_plc-1:])
            final_out = list_to_str(final)
            final_out = final_out[0].upper() + final_out[1:-1] + "."
            return final_out
        elif "same as " in question:
            vb = str_tokens_question[0]
            same_plc = str_tokens_question.index("same")
            final = []
            final.extend(str_tokens_question[1:same_plc])
            final.extend([vb])
            final.extend(str_tokens_question[same_plc:])
            final_out = list_to_str(final)
            final_out = final_out[0].upper() + final_out[1:-1] + "."
            return final_out
        doc = nlp(new_question)
        sent = list(doc.sents)[0]
        parse_str = sent._.parse_string
        t = Tree.fromstring(parse_str)
        this_one = []
        shuchu(t, this_one)
        have_that = False
        that = ""
        for i in this_one:
            for j in i:
                if j in ["that", "which", "who"]:
                    have_that = True
                    that = j
        if have_that:
            parent = []
            for word in tokens_new_question:
                tree = []
                child_to_list_node(word, tree)
                for child in tree:
                    if child.lemma_ in ["that", "which", "who"]:
                        parent.append(word)
            parent_tree = []
            tree_to_list_str(parent[0], parent_tree)
            order_list, order_plc, num = order(parent_tree, this_one)
            if order_list != []:
                this_one[order_plc] = order_list
                this_num = -1
                for i in range(len(this_one[order_plc + 1:])):
                    this_num += 1
                    if num == 1:
                        break
                    this_one[order_plc + 1 + this_num] = []
                    num -= 1
                num = -1
                for part in range(len(this_one)):
                    if num == len(this_one) - 1:
                        break
                    num += 1
                    if this_one[num] == []:
                        this_one.remove([])
                        num -= 1
        num = -1
        if_ctn = True
        while if_ctn:
            num += 1
            if this_one[num] == ["-"]:
                this_one[num].extend(this_one[num + 1])
                this_one[num + 1] = []
                this_one[num - 1].extend(this_one[num])
                this_one[num] = []
                this_one.remove([])
                this_one.remove([])
                num -= 1
            if num == len(this_one) - 1:
                if_ctn = False
        num = -1
        if_ctn = True
        while if_ctn:
            num += 1
            if num == len(this_one) - 1:
                if_ctn = False
            if this_one[num - 1] == ["'"] and this_one[num + 1] == ["'"]:
                this_one[num - 1].extend(this_one[num])
                this_one[num - 1].extend(this_one[num + 1])
                this_one[num] = []
                this_one[num + 1] = []
                this_one[num] = []
                this_one.remove([])
                this_one.remove([])
                num -= 1
            if num == len(this_one) - 2:
                if_ctn = False
        num = -1
        if_ctn = True
        while if_ctn:
            num += 1
            if this_one[num] == ["a"] or this_one[num] == ["an"]:
                this_one[num].extend(this_one[num + 1])
                this_one[num + 1] = []
                this_one.remove([])
                num -= 1
            if num == len(this_one) - 1:
                if_ctn = False
        str_tokens = str_tokens_question[1:]
        pos_tokens = pos_tokens_question[1:]
        all_noun = []
        all_adj = []
        for num in range(len(this_one)):
            if is_noun_from_S2W(this_one[num], pos_tokens_question[1:], str_tokens):
                all_noun.append(num)
        for num in range(len(this_one)):
            if is_adj(this_one[num], pos_tokens_question[1:], str_tokens):
                all_adj.append(num)
        comb = []
        num = -1
        for i in all_adj[:]:
            num += 1
            if num != len(all_adj) - 1:
                if all_adj[num] + 1 == all_adj[num + 1]:
                    if comb == []:
                        comb.append(all_adj[num])
                    for plc in comb:
                        if plc > all_adj[num]:
                            comb.insert(comb.index(plc), all_adj[num])
                            break
                    continue
            if all_adj[num] + 1 in all_noun:
                if comb == []:
                    comb.append(all_adj[num])
                else:
                    for plc in comb:
                        if plc > all_adj[num]:
                            comb.insert(comb.index(plc), all_adj[num])
                            break
        all_nn_adj = all_noun
        for i in all_adj:
            if i not in all_nn_adj:
                all_nn_adj.append(i)
        num = -1
        for i in this_one:
            num += 1
            if this_one[num] == []:
                continue
            if this_one[num][0] in ["with", "of", "on", "in", "that", "which", "who", "of", "and"] and num - 1 in all_nn_adj:
                first_plc = num - 1
                second_plc = num
                if first_plc not in comb:
                    if comb != []:
                        for plc in comb:
                            if plc > first_plc:
                                comb.insert(comb.index(plc), first_plc)
                                break
                        comb.append(first_plc)
                    else:
                        comb.append(first_plc)
                if second_plc not in comb:
                    if comb != []:
                        for plc in comb:
                            if plc > second_plc:
                                comb.insert(comb.index(plc), second_plc)
                                break
                        comb.append(second_plc)
                    else:
                        comb.append(second_plc)
            if this_one[num][0] in ["my", "your", "their", "her", "his", "our"] and len(this_one[num]) == 1:
                first_plc = num - 1
                second_plc = num
                if first_plc not in comb:
                    if comb != []:
                        for plc in comb:
                            if plc > first_plc:
                                comb.insert(comb.index(plc), first_plc)
                                break
                        comb.append(first_plc)
                    else:
                        comb.append(first_plc)
                if second_plc not in comb:
                    if comb != []:
                        for plc in comb:
                            if plc > second_plc:
                                comb.insert(comb.index(plc), second_plc)
                                break
                        comb.append(second_plc)
                    else:
                        comb.append(second_plc)
        new_comb = []
        for i in range(len(comb)):
            new_comb.append(comb[len(comb) - i - 1])
        if new_comb != []:
            print(new_comb)
            print(this_one)
            for i in new_comb:
                if i +1 >= len(this_one):
                    continue
                this_one[i].extend(this_one[i + 1])
                this_one[i + 1] = []
        final = []
        num = -1
        for i in range(len(this_one)):
            num += 1
            if num == len(this_one)-1:
                break
            if this_one[num] == []:
                this_one.remove([])
                num -= 1
        if len(this_one) == 1:
            final.extend(this_one[0])
            final.extend([str_tokens_question[0]])
            for i in this_one[1:]:
                final.extend(i)
        if len(this_one) == 2:
            have_in = False
            for i in this_one[0]:
                if i in ["in", "on", "during", "from", "to"]:
                    have_in = True
                    in_plc = this_one[0].index(i)
            if have_in:
                this_one.insert(1, [])
                this_one[1].extend(this_one[0][in_plc:])
                this_one[0] = this_one[0][:in_plc]
        final.extend(this_one[0])
        final.extend([str_tokens_question[0]])
        for i in this_one[1:]:
            final.extend(i)
        final_out = list_to_str(final)
        final_out = final_out[0].upper() + final_out[1:-1] + "."
        return final_out
    elif str_tokens_question[0].lower() in ["would", "could", "will", "can", "should", "might", "may", "has", "had", "have"]:
        final = []
        if str_tokens_new_question[root_plc-1] == "been" and str_tokens_new_question[root_plc-2] == "ever":
            final.extend(str_tokens_new_question[:root_plc-2])
            final.extend([str_tokens_question[0]])
            final.extend(str_tokens_new_question[root_plc-2:])
        elif "there" == str_tokens_new_question[root_plc-1] and str_tokens_question[0].lower() == 'will':  # add by sqc
            final.extend(str_tokens_new_question[:root_plc])
            final.extend([str_tokens_question[0]])
            final.extend(str_tokens_new_question[root_plc:])
        elif pos_tokens_new_question[root_plc-1] in ["ADJ", "ADV", "ADJP"] or str_tokens_new_question[root_plc-1] == "been":
            final.extend(str_tokens_new_question[:root_plc-1])
            final.extend([str_tokens_question[0]])
            final.extend(str_tokens_new_question[root_plc - 1:])
        else:
            final.extend(str_tokens_new_question[:root_plc])
            final.extend([str_tokens_question[0]])
            final.extend(str_tokens_new_question[root_plc:])
        final.pop()
        final_out = list_to_str(final)
        final_out = final_out[0].upper() + final_out[1:] + "."
        return final_out


def GA2S(question):
    doc_question = nlp_spacy(question)
    tokens_question = [doc_question[0]]
    if tokens_question[0].lemma_ in boolean_set:
        statement = boolean(question, nlp_spacy)
    elif tokens_question[0].lemma_ not in WH_set:
        statement = statement_not_boolean(question, nlp_spacy)
        if "?" in statement:
            statement.replace("?", "")
        if "?" in statement:
            statement.replace("?", "")
    else:
        statement = statement_not_boolean(question, nlp_spacy)
        if "?" in statement:
            statement.replace("?", "")
        if "?" in statement:
            statement.replace("?", "")
    statement = statement.lower()
    if ' any ' in statement:
        statement = statement.replace(' any ', ' some ')
    elif statement.startswith('any '):
        statement = statement.replace('any ', 'some ')
    elif 'anyone ' in statement:
        statement = statement.replace('anyone ', 'someone ')
    elif 'anything ' in statement:
        statement = statement.replace('anything ', 'something ')
    elif 'anybody ' in statement:
        statement = statement.replace('anybody ', 'somebody ')
    return statement


nlp_spacy = spacy.load('en_core_web_sm')
benepar_path = "../3rd_models/benepar_en3"
if spacy.__version__.startswith('2'):
    nlp_spacy.add_pipe(BeneparComponent(benepar_path))
else:
    nlp_spacy.add_pipe("benepar", config={"model": benepar_path})
try:
    print(conjugate("eat", tense=PAST))  
except Exception as e:
    print("This is a workaround for the bug: ", e)

if __name__ == '__main__':
    import time
    begin = time.time()

    questions_list = ["is there really cowbell in don't fear the reaper?",
                      ]
    for q in questions_list:
        res = GA2S(q)
        print(res)
    end = time.time()
    print(end-begin)
