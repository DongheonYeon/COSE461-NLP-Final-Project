# from GPTScore.gpt3_score import gpt3score
# from opt_score import directly_get_score

import sys
import os

path_GPTScore = '/data/ydh/nlp/Self_Reflection_Medical/packages/GPTScore'
sys.path.append(path_GPTScore)
from gpt3_score import gpt3score
from opt_score import OPTScorer

def evaluate_response(entailment_scorer, ctrleval_scorer, question, answer, knowledge):
    scores, _ = entailment_scorer.get_scores(question, [answer])
    entailment_score = scores[0]
    
    if knowledge:
        prefix = [knowledge]
        data = [knowledge+'\n'+answer]
        try:
            cons_result = ctrleval_scorer.score(aspect='cons', data=data, prefix=prefix, batch_size=1)
            cons_score = cons_result[0]
        except:
            cons_score = float('-inf')
    else:
        cons_score = float('-inf')
    return entailment_score, cons_score
        
    # print('cosistency', cons_result)
    
#     Pre, Recall, F1 = bert_score.score([response], [golden_response], lang="en", return_hash=False)
#     Pre = Pre.item()
#     Recall = Recall.item()
#     F1 = F1.item()
#     # print('bert_score Pre, Recall, F1', Pre, Recall, F1)

    

def evaluate_knowledge(gptscore_model, demo_num, question, knowledge, gptscore_tokenizer=None):
    PREFIX = {0: f'''Based on Question, please generate the factual knowledge. To do this, please consider these factors: Verifiability, Objectivity, and Reliability of Source. Note that this evaluation should be based on the best available medical knowledge.
    Question: {question}
    Knowledge: {knowledge}''',}
    prefix = PREFIX[0]
    
    # OPTScorer 초기화 (checkpoint를 gptscore_model로 사용)
    scorer = OPTScorer()
    
    # 입력 데이터 준비
    srcs = [prefix]
    tgts = [knowledge]
    
    # 점수 계산 (batch_size=1로 설정)
    score_list = scorer.score(
        srcs=srcs,
        tgts=tgts,
        prompt_text="",
        batch_size=1
    )
    gptscore = score_list[0]

    return gptscore

# evaluate_understandability
def evaluate_uscore(question, refined):
    uscore = 1
    return uscore