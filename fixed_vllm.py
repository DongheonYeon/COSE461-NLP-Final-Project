#!/usr/bin/env python
# coding: utf-8
import argparse
import os
import sys
import jsonlines
import torch
import transformers
from peft import PeftModel
from transformers import LlamaTokenizer
from tqdm import tqdm
import numpy as np
from vllm import LLM, SamplingParams

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,4,5"
tensor_parallel_size=1

os.environ["TRANSFORMERS_CACHE"] = "/data/ydh/nlp/main/huggingface_model"

# path_alpaca_lora = '/data/ydh/nlp/Self_Reflection_Medical/packages/alpaca-lora'
# sys.path.append(path_alpaca_lora)
# from utils.callbacks import Iteratorize, Stream
# from utils.prompter import Prompter

path_CTRLEval = '/data/ydh/nlp/Self_Reflection_Medical/packages/CTRLEval'
sys.path.append(path_CTRLEval)
from ctrleval import CTRLEval # type: ignore

# path_GPTScore = '/data/ydh/nlp/Self_Reflection_Medical/packages/GPTScore'
# sys.path.append(path_GPTScore)
# from gpt3_score import gpt3score # type: ignore

path_self = '/data/ydh/nlp/Self_Reflection_Medical'
sys.path.append(path_self)
from evaluate.sent_similarity import Sent_Similar # type: ignore
from evaluate.loop_eval_utils import evaluate_response, evaluate_knowledge, evaluate_uscore # type: ignore
from loop_utils import main_loop # type: ignore

# from transformers.utils import logging
# logging.set_verbosity_info()

test_file_path = '/data/ydh/nlp/dataset/tirivia_test.jsonl'

def generate_step_vllm(args, llm, prompt):
    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        max_tokens=args.max_new_tokens,
        n=args.num_beams
    )
    outputs = llm.generate([prompt], sampling_params)
    return outputs[0].outputs[0].text

def knowledge_loop_vllm(args, llm, question, knowledge_loop_list=[]):
    print("knowledge_loop")
    THRESHOLD_FACTUAL = args.threshold_fact
    MAX_KNOWLEDGE_LOOP = args.max_knowledge_loop
    candidates = []
    history = []

    # Prompt 구성 ex) instruction = f'''Provide background knowledge to answer the given question: "{question}".'''
    system_msg = "You are a helpful assistant that provides background knowledge to help answer questions."
    user_msg = f'Provide background knowledge to answer the given question: "{question}"'
    prompt = f"<|system|>\n{system_msg}\n<|user|>\n{user_msg}\n<|assistant|>\n"

    if knowledge_loop_list:
        knowledge = knowledge_loop_list[0]
    else:
        knowledge = generate_step_vllm(args, llm, prompt)
    knowledge_prompt = prompt + knowledge
    with open("/data/ydh/nlp/output/knowledge_prompt_log.txt", "a", encoding="utf-8") as f:
        f.write("=== Knowledge Prompt Start ===\n")
        f.write(knowledge_prompt + "\n")
        f.write("=== Knowledge Prompt End ===\n\n")   

    loop_i = 0
    if MAX_KNOWLEDGE_LOOP > 1:
        if args.gptscore_model == 'gpt3':
            factuality_score = evaluate_knowledge('gpt3', args.demo_num, question, knowledge)
        else:
            factuality_score = evaluate_knowledge(llm, args.demo_num, question, knowledge)
        candidates.append([factuality_score, knowledge])
        history.append([loop_i, knowledge, factuality_score])

    # refine knowledge
    loop_i += 1
    while loop_i < MAX_KNOWLEDGE_LOOP and factuality_score < THRESHOLD_FACTUAL:
        if args.no_aspect:
            refine_msg = f"Please refine the knowledge."
        elif args.no_number:
            refine_msg = f"The knowledge is not strongly supported by empirical evidence. Please refine the knowledge to improve its factuality."
        else:
            refine_msg = f"The factuality score for the knowledge is {factuality_score} less than {THRESHOLD_FACTUAL}, which means the knowledge is not strongly supported by empirical evidence. Please refine the knowledge to improve its factuality."
        
        refine_prompt = f"<|system|>\n{system_msg}\n<|user|>\n{refine_msg}\n<|assistant|>\n"
        knowledge = generate_step_vllm(args, llm, refine_prompt)
        
        knowledge_prompt = refine_prompt + knowledge
        with open("/data/ydh/nlp/output/knowledge_prompt_log.txt", "a", encoding="utf-8") as f:
            f.write("=== Knowledge Prompt Start ===\n")
            f.write(knowledge_prompt + "\n")
            f.write("=== Knowledge Prompt End ===\n\n")
            
        if args.gptscore_model == 'gpt3':
            factuality_score = evaluate_knowledge('gpt3', args.demo_num, question, knowledge)
        else:
            factuality_score = evaluate_knowledge(llm, args.demo_num, question, knowledge)

        candidates.append([factuality_score, knowledge])
        history.append([loop_i, knowledge, factuality_score])
        loop_i += 1

    if MAX_KNOWLEDGE_LOOP > 1 and factuality_score < THRESHOLD_FACTUAL:
        candidates.sort()
        return candidates[-1][-1], history
    else:
        return knowledge, history

def response_loop_vllm(args, llm, question, is_correct, candidate, final_knowledge):
    print("response_loop")
    THRESHOLD_CONS = args.threshold_consistency
    MAX_RESPONSE_LOOP = args.max_response_loop
    candidates = []
    entailment_score_question_list = []
    history = []

    # vLLM용 프롬프트 구성
    # instruction = f'''Refer to the knowledge: "{final_knowledge}" and answer the question: "{question}" with one paragraph.'''
    # instruction = f'''Refer to the knowledge: "{final_knowledge}" and provide a justification for why "{answer}" is the correct answer to the question: "{question}". If the answer is incorrect, explain why it is wrong. Provide your explanation in one paragraph.'''
    # instruction = f'''Given the question: "{question}", analyze why "{answer}" is {"correct" if answer == line["answer"] else "incorrect"}. 
    #                 Refer to this knowledge: "{final_knowledge}". Provide detailed justification in 3-5 sentences.'''
    
    # Test1
    # system_msg = (
    #     "You are a knowledgeable teacher helping students understand multiple-choice questions. "
    #     "For each question and answer choice, explain why the choice is correct or incorrect. "
    #     "Use the provided knowledge to support your explanation. "
    #     "Write your explanation in one paragraph."
    # )
    # user_msg = f'''Question: "{question}"
    # Answer choice: "{candidate}"  
    # Is this the correct answer? → {is_correct}
    # Refer to the knowledge: "{final_knowledge}"
    # Explanation:'''
    
    # Test2
    # system_msg = "For each question and answer choice, explain why the choice is correct or incorrect."
    system_msg = '''You are an expert tutor who explains MCQ choices using background knowledge.'''
    user_msg = f'''Based on the following knowledge: "{final_knowledge}", explain why the answer "{candidate}" is {is_correct} for the question: "{question}" in a single paragraph.
    
    Example 1:
    Question: What is the capital of France?
    Choice: Paris
    Is Correct?: True
    Knowledge: France's capital is Paris.
    Explanation: Paris is the capital of France, so this answer is correct.

    Example 2:
    Question: What is the capital of France?
    Choice: Berlin
    Is Correct?: False
    Knowledge: Berlin is the capital of Germany, not France.
    Explanation: Berlin is the capital of Germany, not France. Therefore, it is not the correct answer.
    '''

    prompt = f"<|system|>\n{system_msg}\n<|user|>\n{user_msg}\n<|assistant|>\n"
    with open("/data/ydh/nlp/output/response_prompt_log.txt", "a", encoding="utf-8") as f:
        f.write("=== Resnpnse Prompt Start ===\n")
        f.write(prompt + "\n")
        f.write("=== Response Prompt End ===\n\n")
        
    response = generate_step_vllm(args, llm, prompt)

    loop_i = 0
    # if MAX_RESPONSE_LOOP > 1:
    entailment_score_question, cons_score_knowledge = evaluate_response(
        entailment_scorer, ctrleval_scorer, question, response, final_knowledge
    )
    candidates.append([(entailment_score_question + cons_score_knowledge) / 2, response])
    entailment_score_question_list.append(entailment_score_question)
    history.append([loop_i, response, entailment_score_question, cons_score_knowledge])
        
    loop_i += 1
    while loop_i < MAX_RESPONSE_LOOP and cons_score_knowledge < THRESHOLD_CONS:
        if args.no_aspect:
            refine_msg = f"Please refine the response."
        elif args.no_number:
            refine_msg = f"The alignment and consistency between response and knowledge are low. Please refine the response to improve its consistency."
        else:
            refine_msg = f"The consistency score for the knowledge is {cons_score_knowledge} less than {THRESHOLD_CONS}, which means the alignment and consistency between response and knowledge are low. Please refine the response to improve its consistency."
        
        refine_prompt = f"<|system|>\n{system_msg}\n<|user|>\n{refine_msg}\n<|assistant|>\n"
        with open("/data/ydh/nlp/output/response_prompt_log.txt", "a", encoding="utf-8") as f:
            f.write("=== Response Prompt Start ===\n")
            f.write(refine_prompt + "\n")
            f.write("=== Response Prompt End ===\n\n")
        response = generate_step_vllm(args, llm, refine_prompt)

        entailment_score_question, cons_score_knowledge = evaluate_response(
            entailment_scorer, ctrleval_scorer, question, response, final_knowledge
        )
        candidates.append([(entailment_score_question + cons_score_knowledge) / 2, response])
        entailment_score_question_list.append(entailment_score_question)
        history.append([loop_i, response, entailment_score_question, cons_score_knowledge])

        loop_i += 1

    if MAX_RESPONSE_LOOP > 1 and cons_score_knowledge < THRESHOLD_CONS:
        merge = list(zip(candidates, entailment_score_question_list))
        merge = sorted(merge)
        candidates, entailment_score_question_list = zip(*merge)
        return candidates[-1][-1], history, entailment_score_question_list[-1]
    else:
        return response, history, entailment_score_question

def initialize_models(args):
    # Initialize scorers
    entailment_scorer = Sent_Similar()
    
    # 필요한 파일 경로 설정
    iwf_dir = '/data/ydh/nlp/Self_Reflection_Medical/packages/CTRLEval/iwf_full.txt'
    prompt_dir = '/data/ydh/nlp/Self_Reflection_Medical/packages/CTRLEval/prompt/prompt_topic.txt'
    verbal_dir = '/data/ydh/nlp/Self_Reflection_Medical/packages/CTRLEval/prompt/verbal_topic.txt'
    
    ctrleval_scorer = CTRLEval(
        iwf_dir=iwf_dir,
        prompt_dir=prompt_dir,
        verbal_dir=verbal_dir,
        device='cuda'
    )  # consistency scorer
    
    # Initialize VLLM model
    # model_path = 'decapoda-research/llama-7b-hf'  # You can change this to your model path
    # llm = LLM(model=model_path)
    llm = LLM(model="deepseek-ai/deepseek-llm-7b-chat", tensor_parallel_size=tensor_parallel_size)
    # sampling_params = SamplingParams(
    #     temperature=0.7,
    #     top_p=0.9,
    #     max_tokens=256
    # )
    return llm, entailment_scorer, ctrleval_scorer

def process_data(args, llm, entailment_scorer, ctrleval_scorer):
    out_base_dir = '/data/ydh/nlp/output'
    out_dir = f"VLLM_MaxL{args.max_loop}_MaxKL{args.max_knowledge_loop}MaxRL{args.max_response_loop}_ThE{args.threshold_entailment}ThF{args.threshold_fact}ThC{args.threshold_consistency}_{args.gptscore_model}_Demo{args.demo_num}"
    out_dir = os.path.join(out_base_dir, out_dir)
    os.makedirs(out_dir, exist_ok=True)

    input_file = args.input_file
    out_file = f'{out_dir}/T{args.temperature}.jsonl'


    # Define the wrapper functions to match what main_loop expects
    def knowledge_loop(args, model, tokenizer, question, knowledge_loop_list=[]):
        # tokenizer는 무시하고 model은 llm으로 사용
        return knowledge_loop_vllm(args, llm, question, knowledge_loop_list)
    
    def response_loop(args, model, tokenizer, question, is_correct, candidate, final_knowledge):
        # tokenizer는 무시하고 model은 llm으로 사용
        return response_loop_vllm(args, llm, question, is_correct, candidate, final_knowledge)

    if args.continue_generate and os.path.exists(out_file):
        print("continue generate")
        with jsonlines.open(out_file) as reader:
            old_lines = list(reader)
        with jsonlines.open(input_file) as reader:
            reader = list(reader)
            for i, line in tqdm(enumerate(reader), total=len(reader)):
                if i < len(old_lines):
                    continue
                if i > args.max_sample:
                    break
                
                candidate_justifications = {}
                for candidate in line['candidate_answers']:
                    temp_line = {
                        'id': line['id'],
                        'url': line.get('url', ''),
                        'subject': line.get('subject', ''),
                        'topic': line.get('topic', ''),
                        'context': line.get('context', ''),
                        'question': line['question'],
                        'answer': [candidate]  # Wrap in list to match original format
                    }
                    # Generate answers for each candidate
                    final_knowledge, final_response, all_history_knowledge, all_history_response = main_loop(args, temp_line, None, None, knowledge_loop, response_loop)            
                    
                    candidate_justifications[candidate] = {             
                            'justification': final_response,
                            'history_knowledge': all_history_knowledge,
                            'history_response': all_history_response,
                            'generated_knowledge': final_knowledge
                        }
                line["candidate_answers"] = candidate_justifications
                
                # Write results
                writer = jsonlines.open(out_file, mode='a')
                writer.write(line)
                writer.close()

    else:
        with jsonlines.open(input_file) as reader:
            reader = list(reader)
            for i, line in tqdm(enumerate(reader), total=len(reader)):
                if i > args.max_sample:
                    break
                
                candidate_justifications = {}
                for candidate in line['candidate_answers']:
                    temp_line = {
                        'id': line['id'],
                        'url': line.get('url', ''),
                        'subject': line.get('subject', ''),
                        'topic': line.get('topic', ''),
                        'context': line.get('context', ''),
                        'question': line['question'],
                        'is_correct': (candidate == line['answer']),    # 정답 선지면 True, 틀린 선지면 False
                        'candidate': [candidate]                        # 선지
                    }
                    # Generate answers for each candidate
                    final_knowledge, final_response, all_history_knowledge, all_history_response = main_loop(args, temp_line, None, None, knowledge_loop, response_loop)            
                    
                    candidate_justifications[candidate] = {             
                            'justification': final_response,
                            'history_knowledge': all_history_knowledge,
                            'history_response': all_history_response,
                            'generated_knowledge': final_knowledge
                        }
                line["candidate_answers"] = candidate_justifications
                
                # Write results
                writer = jsonlines.open(out_file, mode='a')
                writer.write(line)
                writer.close()

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-file", type=str, default=test_file_path)
    parser.add_argument("--continue-generate", action="store_true")
    parser.add_argument("--no-number", action="store_true")
    parser.add_argument("--no-aspect", action="store_true")
    
    parser.add_argument("--max-loop", type=int, default=3)
    parser.add_argument("--max-knowledge-loop", type=int, default=3)
    parser.add_argument("--max-response-loop", type=int, default=3)
    parser.add_argument("--gptscore-model", type=str, default="Alpaca_Lora")
    parser.add_argument("--demo-num", type=int, default=0)
    
    parser.add_argument("--threshold-entailment", type=float, default=0.8)
    parser.add_argument("--threshold-fact", type=float, default=-1)
    parser.add_argument("--threshold-consistency", type=float, default=-5)
    
    parser.add_argument("--max-sample", type=int, default=3000)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=1)
    parser.add_argument("--top_k", type=int, default=1)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    
    args = parser.parse_args()
    
    # Initialize models and run the process
    llm, entailment_scorer, ctrleval_scorer = initialize_models(args)
    process_data(args, llm, entailment_scorer, ctrleval_scorer)
