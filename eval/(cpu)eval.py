import json
import numpy as np
from typing import Dict, List, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rouge_score import rouge_scorer
import torch
from transformers import AutoTokenizer, AutoModel
import warnings
warnings.filterwarnings('ignore')

CUDA_VISIBLE_DEVICES = "7"

class JustificationEvaluator:
    def __init__(self):
        """
        Initialize the evaluator with necessary models and scorers
        """
        # Initialize ROUGE scorer
        self.rouge_scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
        
        # Initialize BERT model for BERTScore
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        self.model = AutoModel.from_pretrained('bert-base-uncased')
        self.model.eval()
        
        # TF-IDF vectorizer for cosine similarity
        self.tfidf_vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words='english',
            ngram_range=(1, 2),
            max_features=10000
        )
    
    def load_jsonl(self, file_path: str) -> List[Dict]:
        """
        Load JSONL file
        """
        data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                data.append(json.loads(line.strip()))
        return data
    
    def get_bert_embeddings(self, texts: List[str]) -> torch.Tensor:
        """
        Get BERT embeddings for texts
        """
        embeddings = []
        
        for text in texts:
            # Tokenize and encode
            inputs = self.tokenizer(
                text, 
                return_tensors='pt', 
                padding=True, 
                truncation=True, 
                max_length=512
            )
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                # Use [CLS] token embedding
                cls_embedding = outputs.last_hidden_state[:, 0, :]
                embeddings.append(cls_embedding.squeeze())
        
        return torch.stack(embeddings)
    
    def calculate_bertscore(self, reference: str, candidate: str) -> float:
        """
        Calculate BERTScore between reference and candidate texts
        """
        ref_embedding = self.get_bert_embeddings([reference])
        cand_embedding = self.get_bert_embeddings([candidate])
        
        # Calculate cosine similarity
        similarity = torch.nn.functional.cosine_similarity(
            ref_embedding, cand_embedding, dim=1
        )
        
        return similarity.item()
    
    def calculate_cosine_similarity(self, reference: str, candidate: str) -> float:
        """
        Calculate TF-IDF based cosine similarity
        """
        try:
            # Fit TF-IDF on both texts
            tfidf_matrix = self.tfidf_vectorizer.fit_transform([reference, candidate])
            
            # Calculate cosine similarity
            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            return similarity
        except:
            return 0.0
    
    def calculate_rouge_l(self, reference: str, candidate: str) -> float:
        """
        Calculate ROUGE-L score
        """
        scores = self.rouge_scorer.score(reference, candidate)
        return scores['rougeL'].fmeasure
    
    def evaluate_single_item(self, gt_item: Dict, output_item: Dict) -> Dict:
        """
        Evaluate a single item comparing GT and output justifications
        """
        results = {
            'id': gt_item['id'],
            'question': gt_item['question'],
            'candidate_scores': {}
        }
        
        # Get candidates from both GT and output
        gt_candidates = gt_item.get('candidates', {})
        output_candidates = output_item.get('candidate_answers', {})
        
        # Evaluate each candidate
        for candidate_key in gt_candidates.keys():
            if candidate_key in output_candidates:
                gt_justification = gt_candidates[candidate_key].get('justification', '')
                output_justification = output_candidates[candidate_key].get('justification', '')
                
                # Calculate all metrics
                bertscore = self.calculate_bertscore(gt_justification, output_justification)
                cosine_sim = self.calculate_cosine_similarity(gt_justification, output_justification)
                rouge_l = self.calculate_rouge_l(gt_justification, output_justification)
                
                results['candidate_scores'][candidate_key] = {
                    'gt_justification': gt_justification,
                    'output_justification': output_justification,
                    'bertscore': bertscore,
                    'cosine_similarity': cosine_sim,
                    'rouge_l': rouge_l,
                    'gt_predicted': gt_candidates[candidate_key].get('predicted', False),
                    'gt_groundtruth': gt_candidates[candidate_key].get('groundtruth', False)
                }
        
        return results
    
    def evaluate_dataset(self, gt_file: str, output_file: str) -> Dict:
        """
        Evaluate entire dataset
        """
        # Load data
        gt_data = self.load_jsonl(gt_file)
        output_data = self.load_jsonl(output_file)
        
        # Create mapping for faster lookup
        output_dict = {item['id']: item for item in output_data}
        
        all_results = []
        overall_scores = {
            'bertscore': [],
            'cosine_similarity': [],
            'rouge_l': []
        }
        
        print(f"Evaluating {len(gt_data)} items...")
        
        for i, gt_item in enumerate(gt_data):
            item_id = gt_item['id']
            
            if item_id in output_dict:
                result = self.evaluate_single_item(gt_item, output_dict[item_id])
                all_results.append(result)
                
                # Collect scores for overall statistics
                for candidate_scores in result['candidate_scores'].values():
                    overall_scores['bertscore'].append(candidate_scores['bertscore'])
                    overall_scores['cosine_similarity'].append(candidate_scores['cosine_similarity'])
                    overall_scores['rouge_l'].append(candidate_scores['rouge_l'])
            
            if (i + 1) % 10 == 0:
                print(f"Processed {i + 1}/{len(gt_data)} items")
        
        # Calculate overall statistics
        summary = {
            'total_items': len(all_results),
            'total_candidates': len(overall_scores['bertscore']),
            'average_scores': {
                'bertscore': np.mean(overall_scores['bertscore']),
                'cosine_similarity': np.mean(overall_scores['cosine_similarity']),
                'rouge_l': np.mean(overall_scores['rouge_l'])
            },
            'std_scores': {
                'bertscore': np.std(overall_scores['bertscore']),
                'cosine_similarity': np.std(overall_scores['cosine_similarity']),
                'rouge_l': np.std(overall_scores['rouge_l'])
            }
        }
        
        return {
            'summary': summary,
            'detailed_results': all_results
        }
    
    def save_results(self, results: Dict, output_file: str):
        """
        Save evaluation results to JSON file
        """
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"Results saved to {output_file}")
    
    def print_summary(self, results: Dict):
        """
        Print evaluation summary
        """
        summary = results['summary']
        
        print("\n" + "="*50)
        print("EVALUATION SUMMARY")
        print("="*50)
        print(f"Total Items: {summary['total_items']}")
        print(f"Total Candidates: {summary['total_candidates']}")
        print()
        
        print("AVERAGE SCORES:")
        print(f"  BERTScore:        {summary['average_scores']['bertscore']:.4f} (±{summary['std_scores']['bertscore']:.4f})")
        print(f"  Cosine Similarity: {summary['average_scores']['cosine_similarity']:.4f} (±{summary['std_scores']['cosine_similarity']:.4f})")
        print(f"  ROUGE-L:          {summary['average_scores']['rouge_l']:.4f} (±{summary['std_scores']['rouge_l']:.4f})")
        print("="*50)


def main():
    """
    Main function to run evaluation
    """
    # Initialize evaluator
    evaluator = JustificationEvaluator()
    
    # File paths (modify these to your actual file paths)
    gt_file = "/data/ydh/nlp/dataset/acc_75.jsonl"
    output_file = "/data/ydh/nlp/output/(75%)VLLM_exp1_MaxL3_ThE0.8ThF-1ThC-5_Alpaca_Lora_Demo0/T1.0.jsonl"
    results_file = "/data/ydh/nlp/eval/evaluation_results.json"
    
    try:
        # Run evaluation
        print("Starting evaluation...")
        results = evaluator.evaluate_dataset(gt_file, output_file)
        
        # Print summary
        evaluator.print_summary(results)
        
        # Save detailed results
        evaluator.save_results(results, results_file)
        
        # Print some example detailed results
        print("\nExample detailed results for first item:")
        if results['detailed_results']:
            first_result = results['detailed_results'][0]
            print(f"ID: {first_result['id']}")
            print(f"Question: {first_result['question']}")
            
            for candidate, scores in first_result['candidate_scores'].items():
                print(f"\nCandidate {candidate}:")
                print(f"  BERTScore: {scores['bertscore']:.4f}")
                print(f"  Cosine Sim: {scores['cosine_similarity']:.4f}")
                print(f"  ROUGE-L: {scores['rouge_l']:.4f}")
                print(f"  GT Text: {scores['gt_justification'][:100]}...")
                print(f"  Output Text: {scores['output_justification'][:100]}...")
        
    except Exception as e:
        print(f"Error during evaluation: {str(e)}")
        print("Please check your file paths and data format.")


if __name__ == "__main__":
    main()


# Additional utility functions for analysis
def analyze_by_correctness(results: Dict):
    """
    Analyze scores by correctness (predicted vs groundtruth)
    """
    correct_scores = {'bertscore': [], 'cosine_similarity': [], 'rouge_l': []}
    incorrect_scores = {'bertscore': [], 'cosine_similarity': [], 'rouge_l': []}
    
    for result in results['detailed_results']:
        for candidate_scores in result['candidate_scores'].values():
            scores_dict = {
                'bertscore': candidate_scores['bertscore'],
                'cosine_similarity': candidate_scores['cosine_similarity'],
                'rouge_l': candidate_scores['rouge_l']
            }
            
            if candidate_scores['gt_predicted'] == candidate_scores['gt_groundtruth']:
                for metric, score in scores_dict.items():
                    correct_scores[metric].append(score)
            else:
                for metric, score in scores_dict.items():
                    incorrect_scores[metric].append(score)
    
    print("\nAnalysis by Correctness:")
    print("Correct Predictions:")
    for metric in correct_scores:
        if correct_scores[metric]:
            print(f"  {metric}: {np.mean(correct_scores[metric]):.4f}")
    
    print("Incorrect Predictions:")
    for metric in incorrect_scores:
        if incorrect_scores[metric]:
            print(f"  {metric}: {np.mean(incorrect_scores[metric]):.4f}")


# Usage example:
"""
# To use this code:

1. Install required packages:
   pip install torch transformers scikit-learn rouge-score numpy

2. Prepare your data files:
   - gt.jsonl: Ground truth file
   - output.jsonl: Model output file

3. Run the evaluation:
   python evaluation_script.py

4. Check results:
   - Console output for summary
   - evaluation_results.json for detailed results
"""