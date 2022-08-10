import time
import os
import copy

from configparser import ConfigParser
from utils.my_logger import logger
from utils.read_data import load_qa
from utils.model_predict import run_predict
from generate.template import *
from calc_sim.sent_sim import calculate_sim_origin_target, calculate_sim_origin_sentence
from utils.preprocess import get_sentences_from_contexts, context_preprocess
from calc_sim.answer_sim import is_same_answer


if __name__ == '__main__':
    begin_time = time.time()
    logger.info(f"Start to run the script {__file__}")
    config = ConfigParser()
    config.read("config/config.ini")
    params = config['PARAMETERS']

    training_set_file = params['TRAINING_FILE_PATH']
    test_set_file = params['TEST_FILE_PATH']

    qa_model_path = params['QA_MODEL_PATH']
    attack_mods = params['ATTACK_MOD'].strip().split(',')
    extra_sent_num2context = int(params['EXTRA_NUM2C'].strip())
    max_attack_num = int(params['MAX_ATTACK_NUM'].strip())
    combined_context_num = int(params['CONTEXT_NUM'].strip())
    origin_answer_flag = params['ORIGIN_ANSWER']
    # ------------------------------------------------------------------------------------------------------------------
    project_name = test_set_file.split("data_")[-1].split('_')[0]
    logger.info(f"project name: {project_name}")
    if project_name != "boolq" and "TI" in attack_mods:
        attack_mods.remove("TI")
    is_boolq = False
    if project_name == 'boolq':
        is_boolq = True

    res_dir = f'../results/{project_name}/res-dev'
    if not os.path.exists(res_dir):
        os.makedirs(res_dir)
    file_comprehensive_res = os.path.join(res_dir, "test_cases_all.tsv")
    file_comprehensive_res_bug = os.path.join(res_dir, "violation_all.tsv")

    success_attack_cov = {}
    for attack_mod in attack_mods:
        success_attack_cov[attack_mod] = []

    # ----------------------------load training set and development set ------------------------------------------------
    qa_list_obj_test = load_qa(test_set_file, end_id=10)
    logger.info("origin question number: " + str(len(qa_list_obj_test.qa_list)))
    qa_list_obj_training = load_qa(training_set_file, end_id=40000, remove_redundancy_context=True)

    with open(f"../datasets/compress/{project_name}.comp.tsv", "r", encoding="utf-8") as f:
        compress_question_list = f.readlines()

    for obj in qa_list_obj_test.qa_list:
        this_attack = random.choice(attack_mods)
        logger.info(f"The qa id[test-set] is: {obj.id}")
        if origin_answer_flag == 'SUT':
            ori_ans = run_predict(obj.Q + '\n ' + '('+obj.T+') ' + obj.C)[0]
        elif origin_answer_flag == 'GT':
            ori_ans = obj.A
        else:
            raise Exception(print("wrong key of origin_answer_flag!"))
        # ----------calculate the similarity between question and each context------------------------------------------
        # -------ignore the qa[in train-set] that have same context with current context.-------------------------------
        qa_list_obj_training_no_this = copy.deepcopy(qa_list_obj_training)
        qa_list_obj_training_no_this.remove_qa_of_this_context(obj.C)
        # calculate sim between 'question' in dev-set and 'context' in train-set!   return [(id,sim_value)]
        # compress question into simple question.
        temp_line_list = compress_question_list[obj.id].split("\t")
        if not obj.Q.strip().startswith(temp_line_list[0].strip()):
            logger.warn("check is same: " + obj.Q + ", " + temp_line_list[0])
        compress_qu = temp_line_list[-1].strip() + "?"

        top_n_sim_contexts_list = calculate_sim_origin_target(compress_qu, qa_list_obj_training_no_this, combined_context_num)
        # print("top_n_sim_contexts_list:", top_n_sim_contexts_list)
        # ------------------------ split top-n similarity contexts into a sentences list. ------------------------------
        extra_sentences_list = []
        sentence_qa_id_dict = {}
        for count in range(combined_context_num):
            context_qa_id = top_n_sim_contexts_list[count][0]
            this_qa = qa_list_obj_training_no_this.qa_list[context_qa_id]
            this_sentences_list = get_sentences_from_contexts([this_qa.C])  # eliminate coreference in this process.
            extra_sentences_list.extend(this_sentences_list)
            for s in this_sentences_list:
                sentence_qa_id_dict[s] = context_qa_id

        # ------------calculate the similarity between question and each sentence from context--------------------------
        top_N_sim_sentences_list = calculate_sim_origin_sentence(compress_qu, extra_sentences_list,
                                                                 top_n=max_attack_num+extra_sent_num2context-1)
        # top_N_sim_sentences_list = list(np.array(top_N_sim_sentences_list_)[:, 0])
        logger.info("..................................................................")
        logger.info(f"origin question is: {obj.Q}")
        logger.info(f"compress question is: {compress_qu}")
        logger.info("top_N_sim_sentences_list" + str(top_N_sim_sentences_list))  # [(sentence, sim)]

        for attack_times in range(max_attack_num):
            # --------------------------------generate a new input(question, context) and answer------------------------
            selected_sent = top_N_sim_sentences_list[attack_times][0]
            selected_sent_processed = context_preprocess(selected_sent)  # remove bracket.
            selected_sent_next = top_N_sim_sentences_list[attack_times+1][0]
            sim_question_sent = top_N_sim_sentences_list[attack_times][1]

            # for fear the selected question is too short(e.g., only contain a single word)
            if len(selected_sent_processed) < 10:
                selected_sent_processed = selected_sent_next
            # --------------------combine multi sent to one that will be insert into origin contexts--------------------
            selected_sent_combine = ''
            for i in range(extra_sent_num2context):
                if attack_times + i < len(top_N_sim_sentences_list):  # for fear index out of boundary
                    selected_sent_combine += (top_N_sim_sentences_list[attack_times + i][0] + ' ')
                else:
                    selected_sent_combine = selected_sent_processed
                    logger.warn("[Warning!] [index out of scope] the hyper-parameter 'CONTEXT_NUM' is too small! ")

            if 'EC' in attack_mods:
                res_file = os.path.join(res_dir, 'EC' + ".tsv")
                new_context = add_extra2context(obj.C, selected_sent_combine)
                new_question = obj.Q
                new_predict_res = run_predict(new_question + '\n '+new_context)[0]  # ('+obj.T+')
                if not is_same_answer(new_predict_res, ori_ans, is_bool=is_boolq):
                    success_attack_cov['EC'].append(obj.id)
                    logger.info('.....................A Successful EC Attack...................................\n')
                    logger.info(f"Question: {new_question}")
                    logger.info(f"Origin context: {obj.C}")
                    logger.info(f"New context: {new_context}")
                    logger.info(f"Ground truth:{obj.A}, New answer:{new_predict_res}")
                    logger.info(f"Attack success rate:{len(success_attack_cov['EC'])/(int(obj.id)+1)}")
                    logger.info('...................................................................................\n')
                    str_output_bug = f"{obj.id}\t{new_question} \\n {new_context}\t{ori_ans}->{new_predict_res}\tEC\n"
                    with open(res_file, 'a', encoding='utf-8') as f:
                        f.write(str_output_bug)
                    if this_attack == 'EC':
                        with open(file_comprehensive_res_bug, 'a', encoding='utf-8') as f:
                            f.write(str_output_bug)
                str_output = f"{new_question} \\n {new_context}\t{ori_ans}\t{new_predict_res}\n"
                with open(res_file + '_all', 'a', encoding='utf-8') as f:
                    f.write(str_output)
                if this_attack == 'EC':
                    with open(file_comprehensive_res, 'a', encoding='utf-8') as f:
                        f.write(str_output)

            if 'EQ' in attack_mods:
                res_file = os.path.join(res_dir, 'EQ' + ".tsv")
                new_question = obj.Q
                if project_name == 'boolq' and random.randint(0, 1) == 1:   # [WARNING] add negative attack!!!
                    new_question = negative_question(new_question)
                new_question = add_extra2question(new_question, selected_sent_processed)
                new_predict_res = run_predict(new_question + '\n '+obj.C)[0]  # ('+obj.T+')
                if not is_same_answer(new_predict_res, ori_ans, is_bool=is_boolq):
                    success_attack_cov['EQ'].append(obj.id)
                    logger.info('.....................A Successful EQ Attack...................................\n')
                    logger.info(f"Origin question: {obj.Q}")
                    logger.info(f"New question: {new_question}")
                    logger.info(f"Context: {obj.C}")
                    logger.info(f"Ground truth:{ori_ans}, Predicting answer:{new_predict_res}")
                    logger.info(f"Attack success rate:{len(success_attack_cov['EQ']) / (int(obj.id)+1)}")
                    logger.info('...................................................................................\n')
                    str_output_bug = f"{obj.id}\t{new_question} \\n {obj.C}\t{ori_ans}->{new_predict_res}\tEQ\n"
                    with open(res_file, 'a', encoding='utf-8') as f:
                        f.write(str_output_bug)
                    if this_attack == 'EQ':
                        with open(file_comprehensive_res_bug, 'a', encoding='utf-8') as f:
                            f.write(str_output_bug)
                str_output = f"{new_question} \\n {obj.C}\t{ori_ans}\t{new_predict_res}\n"
                with open(res_file + '_all', 'a', encoding='utf-8') as f:
                    f.write(str_output)
                if this_attack == 'EQ':
                    with open(file_comprehensive_res, 'a', encoding='utf-8') as f:
                        f.write(str_output)

            if 'EQC' in attack_mods:
                res_file = os.path.join(res_dir, 'EQC' + ".tsv")
                new_question = obj.Q
                if project_name == 'boolq' and random.randint(0, 1) == 1:   # [WARNING] add negative attack!!!
                    new_question = negative_question(new_question)
                new_question = add_extra2question(new_question, selected_sent_processed)
                new_context = add_extra2context(obj.C, selected_sent_combine if ori_ans == 'no' else selected_sent_next)
                new_predict_res = run_predict(new_question + '\n '+new_context)[0]   # ('+obj.T+')
                if not is_same_answer(new_predict_res, ori_ans, is_bool=is_boolq):
                    success_attack_cov['EQC'].append(obj.id)
                    logger.info('.....................A Successful  EQC Attack.................................\n')
                    logger.info(f"New question: {new_question}")
                    logger.info(f"New context: {new_context}")
                    logger.info(f"Ground truth:{ori_ans}, Predicting answer:{new_predict_res}")
                    logger.info(f"Attack success rate:{len(success_attack_cov['EQC'])/(int(obj.id)+1)}")
                    logger.info('...................................................................................\n')
                    str_output_bug = f"{obj.id}\t{new_question} \\n {new_context}\t{ori_ans}->{new_predict_res}\tEQC\n"
                    with open(res_file, 'a', encoding='utf-8') as f:
                        f.write(str_output_bug)
                    if this_attack == 'EQC':
                        with open(file_comprehensive_res_bug, 'a', encoding='utf-8') as f:
                            f.write(str_output_bug)
                str_output = f"{new_question} \\n {new_context}\t{ori_ans}\t{new_predict_res}\n"
                with open(res_file + '_all', 'a', encoding='utf-8') as f:
                    f.write(str_output)
                if this_attack == 'EQC':
                    with open(file_comprehensive_res, 'a', encoding='utf-8') as f:
                        f.write(str_output)

            if 'ETI' in attack_mods:
                selected_qa_id = sentence_qa_id_dict[selected_sent]
                selected_qa = qa_list_obj_training_no_this.qa_list[selected_qa_id]

                res_file = os.path.join(res_dir, 'ETI' + ".tsv")
                if project_name == "boolq":
                    new_question, new_context = add_input_as_redundancy(obj, selected_qa)
                else:
                    new_question, new_context = add_wh_question_as_redundancy(obj, selected_qa)
                new_predict_res = run_predict(new_question + '\n '+new_context)[0]  # ('+obj.T+' && '+selected_qa.T+')
                if not is_same_answer(new_predict_res, ori_ans, is_bool=is_boolq):
                    success_attack_cov['ETI'].append(obj.id)
                    logger.info('....................A Successful ETI Attack........................\n')
                    logger.info(f"Origin question: {obj.Q}")
                    logger.info(f"Selected question: {selected_qa.Q}")
                    logger.info(f"New Question: {new_question}")
                    logger.info(f"Origin context: {obj.C}")
                    logger.info(f"Selected context: {selected_qa.C}")
                    logger.info(f"New context: {new_context}")
                    logger.info(f"Ground truth:{ori_ans}, Predicting answer:{new_predict_res}")
                    logger.info(f"Attack success rate:{len(success_attack_cov['ETI'])/(int(obj.id)+1)}")
                    logger.info('...................................................................................\n')
                    str_output_bug = f"{obj.id}\t{new_question} \\n {new_context}\t{ori_ans}->{new_predict_res}\tETI\n"
                    with open(res_file, 'a', encoding='utf-8') as f:
                        f.write(str_output_bug)
                    if this_attack == 'ETI':
                        with open(file_comprehensive_res_bug, 'a', encoding='utf-8') as f:
                            f.write(str_output_bug)
                str_output = f"{new_question} \\n {new_context}\t{ori_ans}\t{new_predict_res}\n"
                with open(res_file + '_all', 'a', encoding='utf-8') as f:
                    f.write(str_output)
                if this_attack == 'ETI':
                    with open(file_comprehensive_res, 'a', encoding='utf-8') as f:
                        f.write(str_output)

            if 'TI' in attack_mods:
                selected_qa_id = sentence_qa_id_dict[selected_sent]
                selected_qa = qa_list_obj_training_no_this.qa_list[selected_qa_id]

                res_file = os.path.join(res_dir, 'TI' + ".tsv")
                ori_ans2 = selected_qa.A if origin_answer_flag == 'GT' \
                    else run_predict(selected_qa.Q + '\n (' + selected_qa.T + ')' + selected_qa.C)[0]

                new_question, new_context, new_answer = combine2input(obj, selected_qa, ori_ans, ori_ans2)
                new_predict_res = run_predict(new_question + '\n '+new_context)[0]  # ('+obj.T+' && '+selected_qa.T+')
                if not is_same_answer(new_predict_res, new_answer, is_bool=is_boolq):
                    success_attack_cov['TI'].append(obj.id)
                    logger.info('.....................A Successful TI Attack............................\n')
                    logger.info(f"Origin question: {obj.Q}")
                    logger.info(f"Selected question: {selected_qa.Q}")
                    logger.info(f"New Question: {new_question}")
                    logger.info(f"Origin context: {obj.C}")
                    logger.info(f"Selected context: {selected_qa.C}")
                    logger.info(f"New context: {new_context}")
                    logger.info(f"Ground truth:{ori_ans}+{ori_ans2}={new_answer}, Predicting answer:{new_predict_res}")
                    logger.info(f"Attack success rate:"
                                f"{len(success_attack_cov['TI']) / (int(obj.id)+1)}")
                    logger.info('...................................................................................\n')
                    str_output_bug = f"{obj.id}\t{new_question} \\n {new_context}\t{new_answer}->{new_predict_res}\tTI\n"
                    with open(res_file, 'a', encoding='utf-8') as f:
                        f.write(str_output_bug)
                    if this_attack == 'TI':
                        with open(file_comprehensive_res_bug, 'a', encoding='utf-8') as f:
                            f.write(str_output_bug)
                str_output = f"{new_question} \\n {new_context}\t{new_answer}\t{new_predict_res}\n"
                with open(res_file + '_all', 'a', encoding='utf-8') as f:
                    f.write(str_output)
                if this_attack == 'TI':
                    with open(file_comprehensive_res, 'a', encoding='utf-8') as f:
                        f.write(str_output)
    logger.info("-----------------------------[SUMMARY]-----------------------------------------")
    for attack_mod in attack_mods:
        logger.info(f"Attack success rate of {attack_mod}:{len(success_attack_cov[attack_mod])}/{(int(obj.id)+1)}"
                    f" = {round(len(success_attack_cov[attack_mod]) / (int(obj.id)+1), 4)}")
    end_time = time.time()
    logger.info(f"Finish question_gen for question_{obj.id},during time is {round(end_time - begin_time, 2)}s")
