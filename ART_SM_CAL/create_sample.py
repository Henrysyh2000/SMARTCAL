# import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import json
import collections
from typing import Union, List
import ast
import os
import dsp
# load mintaka functions
def load_mintaka(path_to_data_set: str, lang: str="en") -> pd.DataFrame:
    """
    Loads the Mintaka test set as a dataframe
    Args:
        path_to_test_set: path to the Mintaka test set file
        mode: mode of evaluation (kg or text)
        lang: language of evaluation (for text answers)
    Returns:
        mintaka_test: the Mintaka test set as a dataframe
    """
    data = pd.read_json(path_to_data_set)
    data['answer'] = data['answer'].apply(lambda x: format_answers(x, lang))
    return data

def format_answers(answer: dict, mode: str="text", lang: str="en") -> Union[list, str, None]:
    """
    Formats answers from the Mintaka test set based on evaluation mode (kg or text)
    Args:
        answer: answer from the Mintaka test set
        mode: mode of evaluation (kg or text)
        lang: language of evaluation (for text answers)
    Returns:
        The answer either as a list for KG evaluation or a string for text evaluation
    """
    if answer['answerType'] == 'entity':
        if mode == 'kg':
            if answer['answer'] is None:
                return None
            return [ent['name'] for ent in answer['answer']]  # return a list of Q-codes
        else:
            if answer['answer'] is None:
                return answer['mention']  # if no entities linked, return annotator's text answer
            else:
                return ' '.join([ent['label'][lang] if ent['label'][lang]
                                 else ent['label']['en'] for ent in answer['answer']])  # return entity labels
    else:
        return str(answer['answer'][0]) if mode == 'text' else answer['answer']

def calculate_average(x: list):
    clean_list = [i for i in x if i is not None]
    if len(clean_list) == 0:
        return None
    nums = np.array(clean_list)
    return np.average(nums)

# load and sample mintaka train and dev
def make_test(path: str, sample_num: int = 250, uniform: bool = False, full: bool = False, **kwargs) -> List[dsp.Example]:
    '''
    :param path:
    :param sample_num:
    :param kwargs:
    :return 2 lists of Example object:
    load and convert Mintaka dataset to dsp Example object
    1. support section wise sampling
    2. support a popularity threshold
    '''
    random_seed = 100
    # cols = ["question", "answer", "complexityType"]
    data = load_mintaka(path) # "data/mintaka_train.json"
    pop_data = pd.read_csv("entity_count.csv", index_col=0)
    assert (len(pop_data) == len(data))
    pop_data = pop_data.questionEntity.apply(lambda x: ast.literal_eval(x))
    pop_data = pop_data.apply(lambda x: calculate_average(x))
    data["log_pop"] = np.log10(pop_data+1).apply(int)
    data.answer = data.answer.apply(lambda x: [x] if type(x) is str else [str(i) for i in x])
    data = data.dropna()
    assert sample_num <= len(data)
    if "section" in kwargs:
        section = kwargs["section"]
        data = data[data["complexityType"] == section]

    if not full:
        # get samples
        if uniform:
            pops = data.groupby("log_pop").question.count()
            # print(pops)
            assert sample_num <= max(pops)
            indices = list(pops[pops >= sample_num].index)
            data = data[data.log_pop.isin(indices)]
            test = data.groupby("log_pop").apply(lambda x: x.sample(min(len(x), sample_num), random_state=random_seed))
        else:
            test = data.apply(lambda x: x.sample(sample_num, random_state=random_seed))

    else:
        test = data

    # convert to dsp.Example
    dsp_test = list(pd.concat([test['question'], test['answer'], test['log_pop'], test["complexityType"]], axis=1).apply(tuple, axis=1))
    dsp_test = [dsp.Example(question=question, answer=answer, pop=pop, type=type) for question, answer, pop, type in dsp_test]
    return dsp_test

def make_train(path: str, sample_num: int = 50, full: bool = False, **kwargs) -> List[dsp.Example]:
    '''
    :param path:
    :param sample_num:
    :param kwargs:
    :return 2 lists of Example object:
    load and convert Mintaka dataset to dsp Example object
    1. support section wise sampling
    2. support a popularity threshold
    '''
    random_seed = 100
    data = load_mintaka(path)
    data.answer = data.answer.apply(lambda x: [x] if type(x) is str else [str(i) for i in x])

    assert sample_num <= len(data)
    if "section" in kwargs:
        section = kwargs["section"]
        data = data[data["complexityType"] == section]

    if not full:
        # get samples
        train = data.apply(lambda x: x.sample(sample_num, random_state=random_seed))
    else:
        train = data
    # convert to dsp.Example
    dsp_train = list(pd.concat([train['question'], train['answer']], axis=1).apply(tuple, axis=1))
    dsp_train = [dsp.Example(question=question, answer=answer) for question, answer in dsp_train]
    return dsp_train

