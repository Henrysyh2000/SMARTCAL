# import datasets
# import os
import openai
import numpy as np

import adatest
import pdb
# import re
import json
# import jsonlines
# import seqio
# import os
#os.environ['CURL_CA_BUNDLE'] = "/etc/ssl/certs/ca-bundle.crt"
#from bigbench.bbseqio import tasks
#vocabulary=seqio.SentencePieceVocabulary("/gscratch/zlab/bparan/projects/cascades/models/t5-spiece.model")
# from sklearn.metrics import accuracy_score
# from typing import List
# import tqdm
import requests
subscription_key = 'BING_API_KEY'
search_url = "https://api.bing.microsoft.com/v7.0/search"
headers = {"Ocp-Apim-Subscription-Key": subscription_key}
from IPython.display import HTML
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

import subprocess
import sys

openai.api_type = 'openai'
client = openai.OpenAI(
    api_key="OPENAI_API_KEY"

)

# print(openai.api_key)
# cache_dir = '/gscratch/zlab/bparan/projects/cascades/data'
cache_dir = 'art_uw_cache'


# class HuggingFaceModel(adatest.Model):
#     def __init__(self, model="google/flan-t5-small", quote="\"", temperature=0.7, top_p=1, max_length=30, n=1, logprobs=None):
#         self.model = AutoModelForSeq2SeqLM.from_pretrained(model)
#         self.tokenizer = AutoTokenizer.from_pretrained(model)
#         self.quote = quote
#         self.temperature = temperature
#         self.top_p = top_p
#         self.max_length = max_length
#         self.n = n
#         self.logprobs = logprobs
#
#     def __call__(self, strings):
#         pdb.set_trace()
#         inputs = self.tokenizer(strings, return_tensors="pt", padding=True)
#         outputs = self.model.generate(**inputs)
#         resp = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
#         return resp
        

class OpenAIModel(adatest.Model):
    def __init__(self, model="text-davinci-002", quote="\"", temperature=0.7, top_p=1, max_length=30, n=1,
                 logprobs=None, echo=None):

        self.model_name = model
        # if "google" in model:
        #     self.model = HuggingFaceModel(model, quote, temperature, top_p, max_length, n, logprobs)
        # else:
        #     self.model = model
        # self.api_key = openai.api_key
        self.quote = quote
        self.temperature = temperature
        self.top_p = top_p
        self.max_length = max_length
        self.n = n
        self.logprobs = logprobs
        self.echo = echo
        self.sys_prompt = None # results from self evaluation to put into sys prompt

    def add_sys_prompt(self, string):
        self.sys_prompt = string

    def __call__(self, strings):
        if "instruct" not in self.model_name:
            prompt = self.sys_prompt + '\n' + ''.join(strings) if self.sys_prompt else ''.join(strings)
            resp = client.chat.completions.create(
                model=self.model_name,
                messages=[
                    # {"role": "system", "content": self.sys_prompt if self.sys_prompt else "You are a helpful assistant."},
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=self.max_length,
                temperature=self.temperature,
                top_p=self.top_p,
                # n=self.n,
                # stop=self.quote,
                logprobs=True if self.logprobs else False
            )
            # new version to convert to py dict
            resp = resp.model_dump()
            if self.logprobs:
                return resp
            return [x["message"]["content"] for x in resp['choices']]

        else:
            if self.sys_prompt:
                prompt = self.sys_prompt + '\n' + ''.join(strings)
            else:
                prompt = ''.join(strings)

            resp = client.completions.create(
                model=self.model_name,
                prompt=prompt,
                # prompt=strings,
                max_tokens=self.max_length,
                temperature=self.temperature,
                top_p=self.top_p,
                # n=self.n,
                echo=self.echo,
                stop=self.quote,
                logprobs=self.logprobs,
            )
            # new version to convert to py dict
            resp = resp.model_dump()
            if self.logprobs:
                return resp #[x["text"] for x in resp['choices']], [x["logprobs"] for x in resp['choices']]
            return [x["text"] for x in resp['choices']]

class OpenAIModelLogProb(adatest.Model):
    def __init__(self, model="text-davinci-002", quote="\"", temperature=0.7, top_p=1, max_length=30, n=1, logprobs=None):
        self.model = model
        # self.api_key = openai.api_key
        self.quote = quote
        self.temperature = temperature
        self.top_p = top_p
        self.max_length = max_length
        self.n = n
        self.logprobs = logprobs

    def __call__(self, strings):
        resp = client.completions.create(
            model=self.model,
            prompt=strings,
            max_tokens=self.max_length,
            temperature=self.temperature,
            top_p=self.top_p,
            n=self.n,
            stop=self.quote,
            logprobs = self.logprobs
        )
        # new version to convert to py dict
        resp = resp.model_dump()
        return resp

gpt3 = OpenAIModel(model="text-davinci-002",  max_length=200, quote='', n=1)

def propose_decomposition(decomp_prompt, io_pairs, n=20):
    # Default of 0.7 is being used.
    gpt3 = OpenAIModel(model="text-davinci-002",  max_length=500, quote='---', n=n)
    prompt = '''%s. Here are examples of input-output pairs for the task I'm trying to break down.
----
%s
----
Steps:
1.'''%(decomp_prompt, io_pairs)
    return gpt3(prompt)

def propose_instruction(instruct_prompt, io_pairs, n=20):
    gpt3 = OpenAIModel(model="text-davinci-002",  max_length=400, quote='---', n=n)
    prompt = '''%s. Here are examples of input-output pairs for this task.
----
%s
----
I can do this task by'''%(instruct_prompt, io_pairs)
    return gpt3(prompt)


def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]

def get_subset(inputs, labels, n=100):
    idxs = np.random.choice(len(inputs), n, replace=False)
    labs = np.array([labels[i] for i in idxs])
    subset = [inputs[i] for i in idxs]
    return labs, subset
    

def substring_match(labels, predictions):
    correct = 0
    count = 0
    for label, predict in zip(labels, predictions):
        if label.lower() in predict.lower():
            correct += 1
        count += 1
    return (1.0*correct)/count


def substring_match_v2(labels, predictions):
    correct = 0
    count = 0
    for label, predict in zip(labels, predictions):
        for l in label:
            if l.lower() in predict.lower():
                correct += 1
                break
        count += 1
    return (1.0*correct)/count


class Command:
    def __init__(self, command_name, command_conf=None, command_input=None, command_output=None):
        # don't forget to change init here!
        self.command_conf = command_conf
        self.command_name = command_name
        self.command_input = command_input
        self.command_output = command_output

    def __str__(self):
        command = ""
        # also change here!
        command += self.command_name + self.command_conf + "\n---\n"
        if (self.command_input) and (self.command_output):
            command += self.command_input + "\n---\n"
            command += self.command_output
        return command

    @classmethod
    def convert_to_nlprogram(cls, rank, command, input_only=False):
        # TODO: Add feature for multiline (not all multiline inputs and outputs are in new lines in the retrieval)

        if not command.command_input:
            # The program parse identified a command name but no command input or command output.
            # Based on strict grammer parsing rules, this command node is [EOQ]
            return "Q{0}: {1}".format(rank, command.command_name)

        input_sep = "\n" if "\n" in command.command_input else " "
        output_sep = "\n" if "\n" in command.command_output else " "
        # Also need to add conf_score here!
        if not input_only:
            if rank == 1:
                return " {1}{2}{3}{4}{5}\n#{6}:{7}{8}".format(rank, command.command_name, input_sep, command.command_conf, input_sep, command.command_input, rank, output_sep, command.command_output)
            return "Q{0}: {1}{2}{3}{4}{5}\n#{6}:{7}{8}".format(rank, command.command_name, input_sep, command.command_conf, input_sep, command.command_input, rank, output_sep, command.command_output)
        else:
            if rank == 1:
                return " {1}{2}{3}{4}{5}".format(rank, command.command_name, input_sep, command.command_conf, input_sep, command.command_input)
            return "Q{0}: {1}{2}{3}{4}{5}".format(rank, command.command_name, input_sep, command.command_conf, input_sep, command.command_input)


    @classmethod
    def convert_to_nlprogram_no_conf(cls, rank, command, input_only=False):
        # TODO: Add feature for multiline (not all multiline inputs and outputs are in new lines in the retrieval)

        if not command.command_input:
            # The program parse identified a command name but no command input or command output.
            # Based on strict grammer parsing rules, this command node is [EOQ]
            return "Q{0}: {1}".format(rank, command.command_name)

        input_sep = "\n" if "\n" in command.command_input else " "
        output_sep = "\n" if "\n" in command.command_output else " "
        # Also need to add conf_score here!
        if not input_only:
            if rank == 1:
                return " {1}{2}{4}{5}\n#{6}:{7}{8}".format(rank, command.command_name, input_sep,
                                                              command.command_conf, input_sep, command.command_input,
                                                              rank, output_sep, command.command_output)
            return "Q{0}: {1}{2}{4}{5}\n#{6}:{7}{8}".format(rank, command.command_name, input_sep,
                                                               command.command_conf, input_sep, command.command_input,
                                                               rank, output_sep, command.command_output)
        else:
            if rank == 1:
                return " {1}{2}{4}{5}".format(rank, command.command_name, input_sep, command.command_conf, input_sep,
                                                 command.command_input)
            return "Q{0}: {1}{2}{4}{5}".format(rank, command.command_name, input_sep, command.command_conf,
                                                  input_sep, command.command_input)


class StacktraceItem():
    def __init__(self,
                 command, 
                 affordance, 
                 affordance_output, 
                 affordance_details, 
                 rerun_program, 
                 rerun_program_parse):
        self.command = command
        self.affordance = affordance
        self.affordance_output = affordance_output
        self.affordance_details = affordance_details
        self.rerun_program = rerun_program
        self.rerun_program_parse = rerun_program_parse

class Program:
    def __init__(self, input_node, command_node_list, answer_node):
        self.node_list = []

        for node in command_node_list:
            command_name = node[0].text.split(" ", 1)[1]
            if len(node) > 1:
                # change index if conf_score added!
                command_conf = node[1].text
                command_input = node[2].text
                # node[3] is the begin answer in recursive node visit
                command_output = node[4].text
                self.node_list.append(Command(command_name, command_conf, command_input, command_output))
            else:
                self.node_list.append(Command(command_name))

        self.input_node = input_node
        self.answer_node = answer_node

    def __str__(self):
        program = ""
        program += "Program\n======\n"
        program += "Input\n------\n"
        program += self.input_node.__str__()
        program += "\nBreakdown\n------"
        for node in self.node_list:
            program += node.__str__()
            program += "\n------\n"
        program += "\nAnswer\n------\n"
        program += self.answer_node.__str__()
        return program


class Node:
    def __init__(self, expr_name, text):
        self.expr_name = expr_name
        self.text = text

    def __str__(self):
        return json.dumps({"expr_name": self.expr_name, "text": self.text}, indent=2)


def parse_incomplete_program(program=None):
    import parsimonious


    incomplete_grammar = parsimonious.grammar.Grammar(
r"""
program = program_start*node*partial_node
program_start = input_start~r"( |\n)"text~r"\n"
input_start = ~r"Input:"
text = ~r"(?<=Input:( |\n))(.|\n|\t)*?(?=\nQ[0-9]+:)"
node = command_node~r"\n"output_node~r"\n"
command_node = command_start~r"( |\n)"command_instruction
output_node = begin_answer~r"( |\n)"output
command_instruction = ~r"(?<=\]( |\n))(.|\n|\t)*?(?=\n\#[0-9]+)"
command_start = ~r"Q[0-9]+:[ \n]+\[[A-Za-z_ ]+\]"
begin_answer = ~r"\#[0-9]+:"
output = ~r"(?<=\#[0-9]+:( |\n))(.|\n|\t)*?(?=\nQ[0-9]+:)"
partial_node = partial_command_answer / partial_command
partial_command = command_start~r"( |\n)"~r"(?<=\]( |\n))(.|\n|\t)*?$"
partial_command_answer = command_node~r"\n"partial_answer
partial_answer = begin_answer~r"( |\n)"~r"(?<=\#[0-9]+:( |\n))(.|\n|\t)*?$"
""")
    incomplete_parsed_program = incomplete_grammar.parse(program)
    return incomplete_parsed_program


def parse_program(program=None):
    import parsimonious

    def recursive_node_visit(node, selection_criterion, node_list):
        for child in node.children:
            recursive_node_visit(child, selection_criterion, node_list)
        if node.expr_name in selection_criterion:
            node_list.append(Node(node.expr_name, node.text))
            return

# below is the grammar with no conf score:

#     grammar = parsimonious.grammar.Grammar(
# r"""
# program = program_start*node*partial_command*final_answer
# program_start = input_start~r"( |\n)"text~r"\n"
# input_start = ~r"Input:"
# text = ~r"(?<=Input:( |\n))(.|\n|\t)*?(?=\nQ[0-9]+:)"
# node = command_node~r"\n"output_node~r"\n"
# command_node = command_start~r"( |\n)"command_instruction
# output_node = begin_answer~r"( |\n)"output
# command_instruction = ~r"(?<=\]( |\n))(.|\n|\t)*?(?=\n\#[0-9]+)"
# command_start = ~r"Q[0-9]+:[ \n]+\[[A-Za-z_ ]+\]"
# begin_answer = ~r"\#[0-9]+:"
# output = ~r"(?<=\#[0-9]+:( |\n))(.|\n|\t)*?(?=\nQ[0-9]+:)"
# partial_command = command_start~r"\n"
# final_answer = ~r"Ans:( |\n)(.|\n)*$"
# """)

    # below is the grammar with conf score:

    grammar = parsimonious.grammar.Grammar(
r"""
program = program_start*node*partial_command*final_answer
program_start = input_start~r"( |\n)"text~r"\n"
input_start = ~r"Input:"
text = ~r"(?<=Input:( |\n))(.|\n|\t)*?(?=\nQ[0-9]+:)"
node = command_node~r"\n"output_node~r"\n"
command_node = command_start~r"( |\n)"confidence_score~r"( |\n)"command_instruction
output_node = begin_answer~r"( |\n)"output
command_instruction = ~r"(?<=[0-9]\]( |\n))(.|\n|\t)*?(?=\n\#[0-9]+)"
command_start = ~r"Q[0-9]+:[ \n]+\[[A-Za-z_ ]+\]"
confidence_score = ~r"(?<=\]( |\n))\[[0-9]+\]"
begin_answer = ~r"\#[0-9]+:"
output = ~r"(?<=\#[0-9]+:( |\n))(.|\n|\t)*?(?=\nQ[0-9]+:)"
partial_command = command_start~r"\n"
final_answer = ~r"Ans:( |\n)(.|\n)*$"
""")



    parsed_program = grammar.parse(program)

    # A full program has 4 children.: Input, full commans, final partial command (EOC) and answer
    # Access all children to get "input_start" and "text"
    input_node = parsed_program.children[0]
    input_node_list = []
    recursive_node_visit(input_node, ["input_start", "text"], input_node_list)

    # print(len(input_node_list))

    input_text = input_node_list[1]
    command_nodes = parsed_program.children[1]

    # print(command_nodes)

    command_node_list = []
    for node in command_nodes.children:
        # Access all children and focus on getting "command_start", "command_instruction", "begin_answer" and "output"
        child_node_list = []
        # Henry add conf_score here!
        recursive_node_visit(node, ["command_start", "confidence_score", "command_instruction", "begin_answer", "output"], child_node_list)
        command_node_list.append(child_node_list)
    # Access text of answer node that starts with Ans:

    partial_command = []
    recursive_node_visit(parsed_program.children[2], ["command_start"], partial_command)
    command_node_list.append(partial_command)

    answer_node = parsed_program.children[3]
    answer = Node(answer_node.expr_name, answer_node.text)

    nl_program = Program(input_text, command_node_list, answer)
    return nl_program

def fix_program(program):
    # Given various incomplete program parses, figure out a few common mistakes GPT-3 makes and fix them. 
    # This is a more all-encompassing version of complete_program
    pass

import re
# as per recommendation from @freylis, compile once only
CLEANR = re.compile('<.*?>')

def cleanhtml(raw_html):
  cleantext = re.sub(CLEANR, '', raw_html)
  return cleantext

def search(query, top_k=1):
    params = {"q": query, "textDecorations": True, "textFormat": "HTML"}
    response = requests.get(search_url, headers=headers, params=params)
    response.raise_for_status()
    search_results = response.json()
    if "webPages" in search_results:
        snippets = [cleanhtml(v['name'] + " " + v['snippet']) for v in search_results['webPages']["value"]][:top_k]
    else:
        return ""
    return "\n".join(snippets)
    

def execute_code(code_snippet):
    # scope of variables to be defined here?
    # output = exec(code_snippet)
    result = subprocess.run([sys.executable, "-c", code_snippet], capture_output=True, text=True)
    return result

def generate_code(instruction, code_input):
    # Call Codex 002
    response = openai.Edit.create(
    model="code-davinci-edit-001",
    input="x = " + code_input,
    instruction="Python code for " + instruction,
    temperature=0,
    top_p=1
    )
    return response
    

def get_few_shot_prompt(inputs, labels, n=150):
    idx = np.random.randint(0, len(inputs), n)
    prompt_string = ""
    for id_, (inp, label) in enumerate(zip(inputs, labels)):
        if id_ not in idx:
            continue
        prompt_string += inp + "\n"
        prompt_string += label[0] + "\n"
        prompt_string += '----\n'
    return prompt_string

def edit_code(instructions, current_code):
    # Call Codex 001 (Edit) Model
    # Call Codex 002
    response = openai.Edit.create(
    model="code-davinci-edit-001",
    input="x = " + current_code,
    instruction="Python code for " + instructions,
    temperature=0,
    top_p=1
    )
    return response

def get_answer(x, return_original=False):
    search_result = re.search("Ans: ", x)
    if search_result:
        return x[search_result.span(0)[1]:].strip()
    else:
        # Program did not complete
        matches = [match for match in re.finditer("\#[0-9]+:", x)]
        # If any match is made, return the last match output 
        if len(matches):
            return x[matches[-1].start():].strip()
        if return_original:
            return x.strip()
        return ""

def get_autocot_answer(x, answer_prompt="The final answer is ", return_original=False):
    if re.search(answer_prompt, x):
        return x[re.search(answer_prompt, x).span(0)[-1]:].strip()
    else:
        if return_original:
            return x.strip()
        return ""


program = """Input:
Python code:
try:
    n = int(input())
    m = int(input())
    integer_sum = int(n) + int(m)
    print(integer_sum)
except:
    print('error')

  choice: prints number between 5 and 6
  choice: try input and except error
  choice: inputs the string 'try'
  choice: prints sum of two input numbers only if they are integers otherwise raises error
Q1: [code generate] prints number between 5 and 6
#1:
import random

print(random.uniform(5,6))
Q2: [code generate] try input and except error
#2:
try:
    #open file
    file = open(file_name, "r")
    #read file
    data = file.read()
    #close file
    file.close()
    #split data
    data = data.split("\n")
    #remove empty string
Q3: [code generate] inputs the string 'try'
#3: print('try')
Q4: [code generate] prints sum of two input numbers only if they are integers otherwise raises error
#4:
#!/usr/bin/python

a=raw_input("enter first number: ")
b=raw_input("enter second number: ")
try:
    sum=int(a)+int(b)
    print "sum is: ",sum
except:
    print "enter integer values only"
Q5: [compare]
Which of the generated code snippets are most like the original one?"""

#5: prints sum of two input numbers only if they are integers otherwise raises error"""

"""
Q6: [EOC]
Ans:
prints sum of two input numbers only if they are integers otherwise raises error
"""

# print(parse_incomplete_program(program))
