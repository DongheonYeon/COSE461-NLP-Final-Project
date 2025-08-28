import json
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Set
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
import re
from collections import defaultdict
import pickle
import warnings
warnings.filterwarnings('ignore')

class TOEICReadabilityEvaluator:
    def __init__(self, use_cuda=True):
        """
        Initialize TOEIC vocabulary-based readability evaluator
        """
        self.device = torch.device('cuda' if use_cuda and torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Initialize BERT model for word embeddings
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        self.model = AutoModel.from_pretrained('bert-base-uncased')
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # TOEIC score ranges and their difficulty weights
        self.toeic_ranges = {
            '400-500': {'weight': 2.0, 'words': set()},    # Easy words get positive score
            '600-700': {'weight': 1.0, 'words': set()},    # Medium words get small positive
            '800-900': {'weight': -1.0, 'words': set()},   # Hard words get negative score
            '1000-1100': {'weight': -2.0, 'words': set()}, # Very hard words get large negative
        }
        
        # Cache for word embeddings
        self.word_embeddings_cache = {}
        
        # Similarity threshold for word matching
        self.similarity_threshold = 0.7
    
    def create_sample_toeic_vocabulary(self):
        """
        Create sample TOEIC vocabulary sets (replace with actual TOEIC word lists)
        """
        sample_vocab = {
            '400-500': [
                'hello', 'good', 'morning', 'thank', 'you', 'please', 'sorry', 'help',
                'time', 'day', 'work', 'home', 'food', 'water', 'friend', 'family',
                'happy', 'sad', 'big', 'small', 'new', 'old', 'good', 'bad',
                'go', 'come', 'see', 'know', 'think', 'want', 'like', 'need',
                'one', 'two', 'first', 'last', 'here', 'there', 'now', 'today'
            ],
            '600-700': [
                'business', 'meeting', 'project', 'manager', 'office', 'computer',
                'important', 'decision', 'problem', 'solution', 'customer', 'service',
                'experience', 'opportunity', 'development', 'technology', 'information',
                'communication', 'presentation', 'conference', 'schedule', 'appointment',
                'discuss', 'recommend', 'suggest', 'analyze', 'improve', 'organize',
                'effective', 'efficient', 'successful', 'professional', 'responsible'
            ],
            '800-900': [
                'comprehensive', 'implementation', 'methodology', 'infrastructure', 'strategic',
                'optimization', 'competitive', 'innovative', 'substantial', 'significant',
                'collaboration', 'negotiation', 'acquisition', 'diversification', 'consolidation',
                'sophisticated', 'preliminary', 'tremendous', 'exceptional', 'outstanding',
                'facilitate', 'coordinate', 'integrate', 'establish', 'demonstrate',
                'circumstances', 'requirements', 'specifications', 'regulations', 'procedures'
            ],
            '1000-1100': [
                'quintessential', 'unprecedented', 'meticulous', 'comprehensive', 'intricate',
                'sophisticated', 'phenomenal', 'exponential', 'paradigmatic', 'synergistic',
                'entrepreneurial', 'revolutionary', 'transformational', 'multifaceted', 'exponential',
                'substantiate', 'corroborate', 'extrapolate', 'synthesize', 'conceptualize',
                'proliferation', 'diversification', 'standardization', 'systematization', 'optimization',
                'juxtaposition', 'categorization', 'systematization', 'conceptualization', 'implementation'
            ]
        }
        
        # Convert to sets and store
        for level, words in sample_vocab.items():
            self.toeic_ranges[level]['words'] = set(word.lower() for word in words)
        
        print("Sample TOEIC vocabulary loaded:")
        for level, data in self.toeic_ranges.items():
            print(f"  {level}: {len(data['words'])} words")
    
    def load_custom_toeic_vocabulary(self, vocab_file: str):
        """
        Load custom TOEIC vocabulary from JSON file
        Expected format:
        {
            "400-500": ["word1", "word2", ...],
            "600-700": ["word1", "word2", ...],
            "800-900": ["word1", "word2", ...],
            "1000-1100": ["word1", "word2", ...]
        }
        """
        try:
            with open(vocab_file, 'r', encoding='utf-8') as f:
                vocab_data = json.load(f)
            
            for level, words in vocab_data.items():
                if level in self.toeic_ranges:
                    self.toeic_ranges[level]['words'] = set(word.lower() for word in words)
            
            print(f"Custom TOEIC vocabulary loaded from {vocab_file}:")
            for level, data in self.toeic_ranges.items():
                print(f"  {level}: {len(data['words'])} words")
                
        except FileNotFoundError:
            print(f"Vocabulary file {vocab_file} not found. Using sample vocabulary.")
            self.create_sample_toeic_vocabulary()
        except Exception as e:
            print(f"Error loading vocabulary: {e}. Using sample vocabulary.")
            self.create_sample_toeic_vocabulary()
    
    def preprocess_text(self, text: str) -> List[str]:
        """
        Preprocess text and extract words
        """
        # Convert to lowercase and remove punctuation
        text = re.sub(r'[^\w\s]', ' ', text.lower())
        # Split into words and filter out short words
        words = [word for word in text.split() if len(word) > 2]
        return words
    
    def get_word_embedding(self, word: str) -> torch.Tensor:
        """
        Get BERT embedding for a single word with caching
        """
        if word in self.word_embeddings_cache:
            return self.word_embeddings_cache[word]
        
        # Tokenize word
        inputs = self.tokenizer(
            word, 
            return_tensors='pt', 
            padding=True, 
            truncation=True, 
            max_length=512
        )
        
        # Move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            # Use [CLS] token embedding
            embedding = outputs.last_hidden_state[:, 0, :].cpu().squeeze()
        
        # Cache the embedding
        self.word_embeddings_cache[word] = embedding
        return embedding
    
    def get_batch_embeddings(self, words: List[str], batch_size: int = 32) -> Dict[str, torch.Tensor]:
        """
        Get embeddings for multiple words in batches
        """
        embeddings = {}
        
        # Filter out already cached words
        uncached_words = [word for word in words if word not in self.word_embeddings_cache]
        
        # Process in batches
        for i in range(0, len(uncached_words), batch_size):
            batch_words = uncached_words[i:i + batch_size]
            
            # Tokenize batch
            inputs = self.tokenizer(
                batch_words,
                return_tensors='pt',
                padding=True,
                truncation=True,
                max_length=512
            )
            
            # Move to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                batch_embeddings = outputs.last_hidden_state[:, 0, :].cpu()
            
            # Store in cache
            for word, embedding in zip(batch_words, batch_embeddings):
                self.word_embeddings_cache[word] = embedding
        
        # Return all requested embeddings
        for word in words:
            embeddings[word] = self.word_embeddings_cache[word]
        
        return embeddings
    
    def find_word_difficulty_level(self, word: str) -> Tuple[str, float]:
        """
        Find the difficulty level of a word based on TOEIC vocabulary similarity
        Returns: (level, similarity_score)
        """
        word_embedding = self.get_word_embedding(word)
        
        best_level = None
        best_similarity = 0.0
        
        # Check direct match first
        for level, data in self.toeic_ranges.items():
            if word in data['words']:
                return level, 1.0
        
        # If no direct match, find closest match using embeddings
        for level, data in self.toeic_ranges.items():
            if not data['words']:
                continue
                
            # Get embeddings for all words in this level
            level_words = list(data['words'])
            level_embeddings = self.get_batch_embeddings(level_words)
            
            # Calculate similarities
            for vocab_word, vocab_embedding in level_embeddings.items():
                similarity = torch.nn.functional.cosine_similarity(
                    word_embedding.unsqueeze(0), 
                    vocab_embedding.unsqueeze(0)
                ).item()
                
                if similarity > best_similarity and similarity >= self.similarity_threshold:
                    best_similarity = similarity
                    best_level = level
        
        return best_level, best_similarity
    
    def calculate_readability_score(self, text: str) -> Dict:
        """
        Calculate TOEIC-based readability score for a text
        """
        words = self.preprocess_text(text)
        
        if not words:
            return {
                'total_score': 0.0,
                'word_count': 0,
                'level_breakdown': {},
                'unmatched_words': [],
                'matched_words': {}
            }
        
        level_counts = defaultdict(int)
        matched_words = defaultdict(list)
        unmatched_words = []
        total_score = 0.0
        
        print(f"Analyzing {len(words)} words...")
        
        # Analyze each word
        for i, word in enumerate(words):
            level, similarity = self.find_word_difficulty_level(word)
            
            if level and similarity >= self.similarity_threshold:
                # Word matched to a difficulty level
                weight = self.toeic_ranges[level]['weight']
                # Apply similarity weighting
                weighted_score = weight * similarity
                total_score += weighted_score
                
                level_counts[level] += 1
                matched_words[level].append({
                    'word': word,
                    'similarity': similarity,
                    'score': weighted_score
                })
            else:
                # Word not matched - apply neutral penalty
                unmatched_words.append(word)
                total_score += -0.5  # Small penalty for unknown words
            
            if (i + 1) % 50 == 0:
                print(f"  Processed {i + 1}/{len(words)} words")
        
        # Normalize score by word count
        normalized_score = total_score / len(words) if words else 0.0
        
        # Create level breakdown
        level_breakdown = {}
        for level, data in self.toeic_ranges.items():
            count = level_counts[level]
            percentage = (count / len(words)) * 100 if words else 0.0
            avg_score = np.mean([w['score'] for w in matched_words[level]]) if matched_words[level] else 0.0
            
            level_breakdown[level] = {
                'count': count,
                'percentage': percentage,
                'weight': data['weight'],
                'avg_score': avg_score,
                'words': matched_words[level]
            }
        
        return {
            'total_score': total_score,
            'normalized_score': normalized_score,
            'word_count': len(words),
            'matched_count': sum(level_counts.values()),
            'unmatched_count': len(unmatched_words),
            'level_breakdown': level_breakdown,
            'unmatched_words': unmatched_words[:20],  # Show first 20 unmatched words
            'readability_interpretation': self.interpret_score(normalized_score)
        }
    
    def interpret_score(self, normalized_score: float) -> str:
        """
        Interpret the readability score
        """
        if normalized_score >= 1.0:
            return "Very Easy (Elementary level)"
        elif normalized_score >= 0.5:
            return "Easy (Pre-intermediate level)"
        elif normalized_score >= 0.0:
            return "Moderate (Intermediate level)"
        elif normalized_score >= -0.5:
            return "Difficult (Upper-intermediate level)"
        else:
            return "Very Difficult (Advanced level)"
    
    def evaluate_justifications(self, data_file: str, vocab_file: str = None) -> Dict:
        """
        Evaluate readability of justifications in the dataset
        """
        # Load vocabulary
        if vocab_file:
            self.load_custom_toeic_vocabulary(vocab_file)
        else:
            self.create_sample_toeic_vocabulary()
        
        # Load data
        with open(data_file, 'r', encoding='utf-8') as f:
            data = [json.loads(line) for line in f]
        
        results = []
        overall_scores = []
        
        print(f"Evaluating readability for {len(data)} items...")
        
        for i, item in enumerate(data):
            item_result = {
                'id': item['id'],
                'question': item.get('question', ''),
                'candidate_readability': {}
            }
            
            # Analyze justifications for each candidate
            candidates = item.get('candidate_answers', item.get('candidates', {}))
            
            for candidate_key, candidate_data in candidates.items():
                justification = candidate_data.get('justification', '')
                
                if justification:
                    readability = self.calculate_readability_score(justification)
                    item_result['candidate_readability'][candidate_key] = readability
                    overall_scores.append(readability['normalized_score'])
            
            results.append(item_result)
            
            if (i + 1) % 10 == 0:
                print(f"Processed {i + 1}/{len(data)} items")
        
        # Calculate overall statistics
        summary = {
            'total_items': len(results),
            'total_justifications': len(overall_scores),
            'average_readability_score': np.mean(overall_scores) if overall_scores else 0.0,
            'std_readability_score': np.std(overall_scores) if overall_scores else 0.0,
            'score_distribution': {
                'very_easy': len([s for s in overall_scores if s >= 1.0]),
                'easy': len([s for s in overall_scores if 0.5 <= s < 1.0]),
                'moderate': len([s for s in overall_scores if 0.0 <= s < 0.5]),
                'difficult': len([s for s in overall_scores if -0.5 <= s < 0.0]),
                'very_difficult': len([s for s in overall_scores if s < -0.5])
            }
        }
        
        return {
            'summary': summary,
            'detailed_results': results
        }
    
    def save_vocabulary_cache(self, cache_file: str):
        """Save word embeddings cache to file"""
        with open(cache_file, 'wb') as f:
            pickle.dump(self.word_embeddings_cache, f)
        print(f"Vocabulary cache saved to {cache_file}")
    
    def load_vocabulary_cache(self, cache_file: str):
        """Load word embeddings cache from file"""
        try:
            with open(cache_file, 'rb') as f:
                self.word_embeddings_cache = pickle.load(f)
            print(f"Vocabulary cache loaded from {cache_file}")
        except FileNotFoundError:
            print(f"Cache file {cache_file} not found. Starting with empty cache.")


def main():
    """
    Main function to demonstrate TOEIC readability evaluation
    """
    # Initialize evaluator
    evaluator = TOEICReadabilityEvaluator(use_cuda=True)
    
    # Example usage
    sample_texts = [
        "Hello, this is a very simple and easy sentence for beginners.",
        "The business meeting will discuss important project developments and strategic decisions.",
        "The comprehensive implementation of sophisticated methodologies requires meticulous coordination.",
        "Quintessential paradigmatic transformations necessitate unprecedented multifaceted conceptualization."
    ]
    
    print("="*60)
    print("TOEIC READABILITY EVALUATION DEMO")
    print("="*60)
    
    # Load vocabulary (you can replace with custom file)
    evaluator.create_sample_toeic_vocabulary()
    
    # Evaluate sample texts
    for i, text in enumerate(sample_texts):
        print(f"\nText {i+1}: {text}")
        print("-" * 40)
        
        result = evaluator.calculate_readability_score(text)
        
        print(f"Readability Score: {result['normalized_score']:.3f}")
        print(f"Interpretation: {result['readability_interpretation']}")
        print(f"Words: {result['word_count']} (Matched: {result['matched_count']}, Unmatched: {result['unmatched_count']})")
        
        print("\nLevel Breakdown:")
        for level, breakdown in result['level_breakdown'].items():
            if breakdown['count'] > 0:
                print(f"  {level}: {breakdown['count']} words ({breakdown['percentage']:.1f}%) - Avg Score: {breakdown['avg_score']:.3f}")
    
    # Example dataset evaluation
    print("\n" + "="*60)
    print("DATASET EVALUATION EXAMPLE")
    print("="*60)
    
    # You would call this with your actual data file
    results = evaluator.evaluate_justifications('output.jsonl', 'toeic_vocabulary.json')
    evaluator.save_results(results, 'readability_results.json')


if __name__ == "__main__":
    main()


# Additional utility functions
def create_sample_vocabulary_file():
    """
    Create a sample TOEIC vocabulary JSON file
    """
    sample_vocab = {
        "400-500": [
            "hello", "good", "morning", "thank", "you", "please", "sorry", "help",
            "time", "day", "work", "home", "food", "water", "friend", "family",
            "happy", "sad", "big", "small", "new", "old", "go", "come", "see", "know"
        ],
        "600-700": [
            "business", "meeting", "project", "manager", "office", "computer",
            "important", "decision", "problem", "solution", "customer", "service",
            "experience", "opportunity", "development", "technology", "discuss", "recommend"
        ],
        "800-900": [
            "comprehensive", "implementation", "methodology", "infrastructure", "strategic",
            "optimization", "competitive", "innovative", "substantial", "significant",
            "collaboration", "negotiation", "facilitate", "coordinate", "integrate"
        ],
        "1000-1100": [
            "quintessential", "unprecedented", "meticulous", "sophisticated", "intricate",
            "phenomenal", "exponential", "paradigmatic", "synergistic", "entrepreneurial",
            "substantiate", "corroborate", "extrapolate", "synthesize", "conceptualize"
        ]
    }
    
    with open('toeic_vocabulary.json', 'w', encoding='utf-8') as f:
        json.dump(sample_vocab, f, indent=2, ensure_ascii=False)
    
    print("Sample TOEIC vocabulary file created: toeic_vocabulary.json")


# Usage Instructions:
"""
1. Install required packages:
   pip install torch transformers scikit-learn numpy pandas

2. Create TOEIC vocabulary file (toeic_vocabulary.json):
   {
     "400-500": ["easy", "words", ...],
     "600-700": ["medium", "words", ...],
     "800-900": ["hard", "words", ...],
     "1000-1100": ["very", "hard", "words", ...]
   }

3. Run evaluation:
   evaluator = TOEICReadabilityEvaluator(use_cuda=True)
   results = evaluator.evaluate_justifications('output.jsonl', 'toeic_vocabulary.json')

4. Scoring System:
   - Easy words (400-500): +2.0 points
   - Medium words (600-700): +1.0 points
   - Hard words (800-900): -1.0 points
   - Very hard words (1000-1100): -2.0 points
   - Unknown words: -0.5 points

5. Higher positive scores = easier to understand
   Lower negative scores = harder to understand
"""