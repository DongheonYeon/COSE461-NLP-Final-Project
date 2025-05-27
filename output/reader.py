import json
import jsonlines
import os

fixed_lines = []
input_path ="/data/ydh/nlp/output/VLLM_exp2_MaxL3_ThE0.8ThF-1ThC-5_Alpaca_Lora_Demo0/T1.0.jsonl"
# input_path = "/data/ydh/nlp/output/VLLM_exp1_MaxL3_ThE0.8ThF-1ThC-5_Alpaca_Lora_Demo0/T1.0.jsonl"
output_path = os.path.join(os.path.dirname(input_path), 'fixed_T1.0.txt')

with open(input_path, 'r', encoding='utf-8') as infile, \
     open(output_path, 'w', encoding='utf-8') as outfile:
    
    for line_num, line in enumerate(infile, 1):
        try:
            # 비표준 float 제거
            clean_line = line.replace('-Infinity', 'null').replace('Infinity', 'null').replace('NaN', 'null')
            item = json.loads(clean_line)
            
            # 보기 좋게 정리
            outfile.write(f"ID: {item['id']}\n")
            outfile.write(f"Question: {item['question']}\n\n")
            
            for candidate, content in item['candidate_answers'].items():
                outfile.write(f"Candidate: {candidate}\n")
                outfile.write(f"Justification:\n{content['justification']}\n\n")
            
            outfile.write("=" * 80 + "\n\n")
        
        except json.JSONDecodeError as e:
            print(f"[!] JSONDecodeError at line {line_num}: {e}")

print(f"\n✅ 저장 완료:'{output_path}'에 저장되었습니다.")

