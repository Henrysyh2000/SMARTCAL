import os
import argparse
import ast
import json
import pdb
import re
import time
from re import L
# from turtle import pd

import datasets
import numpy as np
from tqdm import tqdm
import pandas as pd
import json

from utils import (OpenAIModel, cache_dir, chunks, get_answer, get_autocot_answer,
                   get_few_shot_prompt, get_subset, gpt3,
                   propose_decomposition, propose_instruction, substring_match)

import urllib.request
from collections import Counter

from prompt_library import (llm_similar_tasks, random_tasks,
                            similar_auto_breakdowns, similar_tasks,
                            few_shot_retrieval_prompt, few_shot_code_prompt,
                            few_shot_arithmetic_prompt, few_shot_string_prompt)
from sequential_interpreter import TopDownVisitorBeta


def exact_match(labels, predictions):
    correct = 0
    count = 0
    for label, predict in zip(labels, predictions):
        if label.lower() == predict.lower():
            correct += 1
        count += 1
    return (1.0 * correct) / count


def token_match(labels, predictions):
    correct = 0
    count = 0
    for label, predict in zip(labels, predictions):
        if label.lower() in [p.lower() for p in predict]:
            correct += 1
        count += 1
    return (1.0 * correct) / count


def nl_program(train, test, section: str = None, temperature=0.3, model_name="text-davinci-002",
               get_log=None , echo=None, self_eval=False, **kwargs):
    """
    :param train: training data
    :param test: testing data
    :param section: mintaka section in the output json file
    :param temperature: llm temperature
    :param model_name: llm for prediction
    :param get_log: get logprobs for each model generation
    :param self_eval: get self report eval on task familiarity and demo similarity
    :return: a json file in the path

    implement a checkpoint sys in case down
    """


    # global few_shot_cot_prompt
    task_name = "Open-domain qa"
    # task_description = """(Open-domain qa) Follow the format of the tasks above and answer the question below. For numerical questions, give arabic answers. For yes or no question, answer True or False."""
    task_description = """(Open-domain qa) Follow the format of the tasks above and answer the question below."""
    if isinstance(test, list):
        inputs = [d.question for d in test]
        labels = [d.answer[0] for d in test]
    else:
        inputs = [test]

    train_inputs = [d.question for d in train]
    train_labels = [d.answer for d in train]
    io_pairs = [(i, l[0]) for i, l in zip(train_inputs, train_labels)]


    # checkpoint system
    if section:
        file_name = [section, model_name, "out.json"]
        file_name = '_'.join(file_name)

        # start from the length of the json file
        if file_name in os.listdir():
            with open(file_name) as f:
                data = json.loads(f.read())
                inputs = inputs[len(data):]
                labels = labels[len(data):]
                print(f"start from question {len(data)+1}...")

            with open("cached_prompt.txt") as f:
                few_shot_cot_prompt = f.read()
                print("cached prompts successfully loaded!")

        else:
            with open(file_name, 'w') as f:
                json.dump([], f, indent=4)
                print("Create an empty list in json file!")

            # ensure the selected demos are same
            # with open("cached_prompt.txt") as f:
            #     few_shot_cot_prompt = f.read()
            #     print("cached prompts successfully loaded!")

            # uncomment to use different demo selection
            few_shot_cot_prompt = llm_similar_tasks(task_name, task_description, io_pairs, N=3)
            with open("cached_prompt.txt", 'w') as f:
                f.write(few_shot_cot_prompt)
                print("Caching the llm selected similar tasks!")

    else:
        few_shot_cot_prompt = llm_similar_tasks(task_name, task_description, io_pairs, N=3)

    interpreter = TopDownVisitorBeta(model_name=model_name, exclude_list=["[generate python code]"],
                                     get_log=get_log, echo=echo)

    def predict(description, question, get_log=False, echo=None, self_eval=None):
        gpt3 = OpenAIModel(model=model_name, max_length=500, temperature=temperature, quote='---', n=1)
        # add self eval results into generation
        if self_eval:
            gpt3.add_sys_prompt(self_eval)

        prompt = [few_shot_cot_prompt % (description, question)]
        if get_log:
            gpt3 = OpenAIModel(model=model_name, max_length=500, temperature=temperature, quote='---', n=1, logprobs=1, echo=echo)
        return prompt, gpt3(prompt)

    # perf_array = []
    # raw_answers = []
    # answers = []
    for i, x in tqdm(enumerate(inputs)):
        # compare and store simliarity & familiarity eval in the intepreter
        eval_res = None
        if self_eval and eval_res is None:
            gpt3 = OpenAIModel(model="gpt-4-turbo", max_length=500, temperature=temperature, quote='---', n=1)
            demo_chunks = few_shot_cot_prompt.split('----')[1:-1]

            # read similarity prompt
            with open("prompt_similarity.txt") as f:
                sim_eval_prompt = f.read()

            # for prompt optimization
            # if kwargs['sim_ans']:
            #     sim_eval_ans = [kwargs['sim_ans']]
            # else:
            sim_eval_ans = gpt3(sim_eval_prompt % (x, '----'.join(demo_chunks)))

            # read familiarity prompt
            with open("prompt_familiarity.txt") as f:
                fam_eval_prompt = f.read()

            # for prompt optimization
            # if kwargs['fam_ans']:
            #     fam_eval_ans = [kwargs['fam_ans']]
            # else:
            fam_eval_ans = gpt3(fam_eval_prompt % x)

            # read instruction prompt
            with open("prompt_sim_fam_instruction.txt") as f:
                ins_prompt = f.read()

            eval_res = gpt3(ins_prompt % (sim_eval_ans[0], fam_eval_ans[0]))
            interpreter.add_self_eval(eval_res[0])

            # print(f"This is the instruction prompt: \n{ins_prompt % (sim_eval_ans[0], fam_eval_ans[0])}")
            #
            print(f"This is the injected eval result: \n{eval_res[0]}")
            print("Self eval injected successfully!")

        if get_log:
            prompts, answer = predict(task_description, x, get_log=True, echo=echo, self_eval=eval_res[0])
            # print(answer)
            logprobs = [x["logprobs"] for x in answer['choices']] # initial logprobs with no tool edit
            try:
                answer = [x["text"] for x in answer['choices']]
            except:
                answer = [x["message"]["content"] for x in answer['choices']]

        else:
            prompts, answer = predict(task_description, x, self_eval=eval_res[0] if eval_res else None)

        # get llm response before tool augmentation
        raw_programs = []
        for prefix, program in zip(prompts, answer):
            program = prefix.rsplit("----\n", 1)[1].split("\n", 1)[1] + program # clean up demos and retain qa part
            raw_programs.append(program)

        # ART program edit, returns a list
        # check if want to get logprobs
        if get_log:
            new_answer, new_logprobs = interpreter.batch_visit_with_prob(prompts, answer, logprobs[0])
        else:
            new_answer = interpreter.batch_visit(prompts, answer)
        # print(f"This is the ART history:\n {new_answer}")
        # print(f"this is new logprobs: {new_logprobs}")


        pred = [get_answer(ans) for ans in new_answer]

        # add to the existing json file
        if section:
            with open(file_name) as f:
                data = json.loads(f.read())

            with open(file_name, "w") as f:
                content = {
                    "question": x,
                    "answer": labels[i],
                    "prediction": pred[0],
                    "fam_sim_ins": eval_res[0] if eval_res else None,
                    "after_art": new_answer[0],
                }
                # if get_log:
                #     content["logprobs"] = logprobs + new_logprobs
                data.append(content)
                json.dump(data, f, indent=4)
                f.close()
        # intend for prompt optimization
        else:
            content = {
                "question": x,
                # "answer": labels[i],
                "prediction": pred[0],
                "after_art": new_answer[0],
            }
            return content

    # preds = [get_answer(x) for x in answers]
    # perf_array.append(substring_match(labels, preds))
    # res = []
    # for ques, label, ans, before, after in zip(inputs, labels, preds, raw_answers, answers):
    #     res.append({
    #         "question": ques,
    #         "answer": label,
    #         "prediction": ans,
    #         "before_art": before,
    #         "after_art": after,
    #     })
    #
    # file_name = [section, model_name, "out.json"]
    # with open("_".join(file_name), "w") as f:
    #     json.dump(res, f, indent=4)



    # print(preds)
    # positive_calls = [int(len(stack_trace_list) >= 1) for stack_trace_list in interpreter.execution_details]
    # positive_rate = sum(positive_calls) / len(interpreter.execution_details)





# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--model_name", type=str,
#                         choices=["text-davinci-002", "text-davinci-003", "code-davinci-002", "code-cushman-001",
#                                  "davinci-codex-002-msft"], default="text-davinci-002")
#     parser.add_argument("--temperature", type=float, default="0.3")
#     parser.add_argument("--inference_strategy", type=str,
#                         choices=["dummy", "few_shot", "auto_cot", "cot_rollout", "few_shot_cot", "nl_program"],
#                         default="few_shot")
#     parser.add_argument("--num_train_examples", type=int, default=10)
#     parser.add_argument("--num_dev_examples", type=int, default=len(inputs))
#     parser.add_argument("--self_consistency", default=False, action='store_true')
#     parser.add_argument("--selection_strategy", type=str,
#                         choices=["fixed", "random", "similar", "similar_auto_decomp", "llm_similar"], default="fixed")
#
#     args = parser.parse_args()

    # print("Dataset statistics")
    # print(task_description)
    # print("Training examples:", len(train_inputs))
    # print("Dev examples:", len(inputs))

    # inputs = inputs[:args.num_dev_examples]
    # labels = labels[:args.num_dev_examples]

    # if args.inference_strategy == "few_shot":
    #     few_shot_prompt = get_few_shot_prompt(train_inputs, train_labels, n=args.num_train_examples)
    #     print("Length of few-shot prompt", len(tokenizer(few_shot_prompt)['input_ids']))
    #     few_shot(args.num_train_examples, args.temperature, args.model_name)
    # elif args.inference_strategy == "auto_cot":
    #     auto_cot(args.temperature, args.model_name, predict=True, use_corrected=False, self_consistency=False)
    # elif args.inference_strategy == "few_shot_cot":
    #     few_shot_cot(args.temperature, args.model_name, strategy=args.selection_strategy)
    # elif args.inference_strategy == "nl_program":
    #     nl_program(args.temperature, args.model_name, self_consistency=args.self_consistency,
    #                strategy=args.selection_strategy)

def conf_edit(file, conf_range, acc_lst):
    df = pd.read_json(file)
    print(f"Editing file {file}...")

    file_name = "edited_" + file
    start_ind = 0
    # start from the length of the json file
    if file_name in os.listdir():
        with open(file_name) as f:
            data = json.loads(f.read())
            print(f"start from question {len(data) + 1}...")
            start_ind = len(data)

    else:
        with open(file_name, 'w') as f:
            json.dump([], f, indent=4)
            print("Create an empty list in json file!")

    for ind, row in tqdm(df[start_ind:].iterrows()):
        # gpt-3.5-turbo-instruct-0914  gpt-4-turbo

        gpt = OpenAIModel(model="gpt-3.5-turbo-instruct-0914", max_length=2000, temperature=0.7, quote='---', n=1)
        prompt = """
You are given a resaoning process with confidence scores within each step in the square bracket "[]".
Your job is to refer to the accuracy confidence table below and edit the confidence scores in the reasoning.
Instructions: 
First identify the confidence range and find the corresponding accuracy in the table.
If accuracy is lower than confidence, you should decrease the score. 
If accuracy is higher than confidence, you should increase the score.
Finally, replace the original confidence score with your newly edited score.
Your answer should keep the exact same structure of reasoning text and the input question, no extra explanation is needed. 
----
Below is the accuracy-confidence table:
confidence level: %s
true accuracy: %s
----
Reasoning text to edit: %s

Your edited reasoning text:
"""
        #     print("Done!")
        row.after_art = gpt(prompt % (str(conf_range), str(acc_lst), row.after_art))[0]

        with open(file_name) as f:
            data = json.loads(f.read())

        with open(file_name, 'w') as f:
            data.append(dict(row))
            json.dump(data, f, indent=4)
            f.close()