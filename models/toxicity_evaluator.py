import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import evaluate
from tqdm import tqdm
import numpy as np
from config import CONFIG
from utils.helpers import Timer

class ToxicityEvaluator:
    def __init__(self, performance_tracker=None):
        self.model_name = CONFIG["toxicity_model"]
        self.device = CONFIG["device"]
        self.performance_tracker = performance_tracker
        self.setup_evaluator()

    def setup_evaluator(self):
        """Initialize toxicity detection components"""
        with Timer(self.performance_tracker, 'model_load_time'):
            # Initialize tokenizer and model
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                device_map="auto"
            )
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name,
                device_map="auto"
            )

            # Setup sentiment pipeline
            self.sentiment_pipe = pipeline(
                "sentiment-analysis",
                model=self.model_name,
                device=self.device
            )

            # Configure reward settings
            self.reward_logits_kwargs = {
                "top_k": None,
                "function_to_apply": "none",
                "batch_size": CONFIG["reward_kwargs"]["batch_size"]
            }
            
            self.reward_probabilities_kwargs = {
                "top_k": None,
                "function_to_apply": "softmax",
                "batch_size": CONFIG["reward_kwargs"]["batch_size"]
            }

            # Load toxicity evaluator
            self.toxicity_evaluator = evaluate.load(
                "toxicity",
                self.model_name,
                module_type="measurement",
                toxic_label="hate"
            )

    def get_toxicity_scores(self, texts, use_probabilities=True):
        """Get toxicity scores for batch of texts"""
        with Timer(self.performance_tracker, 'response_time'):
            kwargs = (self.reward_probabilities_kwargs 
                     if use_probabilities 
                     else self.reward_logits_kwargs)
            
            results = self.sentiment_pipe(texts, **kwargs)
            
            if use_probabilities:
                scores = [result[0]['score'] for result in results]
            else:
                not_hate_index = 0
                scores = [result[not_hate_index]["score"] for result in results]
                
            # Track toxicity scores
            if self.performance_tracker:
                for score in scores:
                    self.performance_tracker.add_metric('toxicity_scores', score)
                    
            return scores

    def evaluate_model_toxicity(self, model, tokenizer, texts, num_samples=10):
        """Evaluate model's toxicity levels"""
        toxicities = []
        
        for i, text in enumerate(texts):
            if i >= num_samples:
                break
                
            inputs = tokenizer(
                text,
                return_tensors="pt",
                padding=True
            ).to(self.device)
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=100,
                    do_sample=True,
                    temperature=0.7
                )
                
            generated_text = tokenizer.decode(
                outputs[0],
                skip_special_tokens=True
            )
            
            toxicity_score = self.toxicity_evaluator.compute(
                predictions=[text + " " + generated_text]
            )
            toxicities.extend(toxicity_score["toxicity"])

        return np.mean(toxicities), np.std(toxicities)

    def get_toxicity_reward(self, text):
        """Get toxicity reward for a single text"""
        input_ids = self.tokenizer(text, return_tensors="pt").input_ids.to(self.device)
        logits = self.model(input_ids=input_ids).logits
        not_hate_index = 0
        reward = logits[:, not_hate_index].item()
        
        if self.performance_tracker:
            self.performance_tracker.add_metric('toxicity_scores', reward)
            
        return reward

    def batch_evaluate_toxicity(self, texts):
        """Evaluate toxicity for a batch of texts"""
        with Timer(self.performance_tracker, 'response_time'):
            results = self.sentiment_pipe(
                texts,
                **self.reward_logits_kwargs
            )
            
            not_hate_index = 0
            rewards = [
                result[not_hate_index]["score"]
                for result in results
            ]
            
            if self.performance_tracker:
                for reward in rewards:
                    self.performance_tracker.add_metric('toxicity_scores', reward)
            
            return rewards

    def evaluate_toxicity_improvement(self, before_texts, after_texts):
        """Compare toxicity levels before and after training"""
        before_scores = self.batch_evaluate_toxicity(before_texts)
        after_scores = self.batch_evaluate_toxicity(after_texts)
        
        before_mean = np.mean(before_scores)
        before_std = np.std(before_scores)
        after_mean = np.mean(after_scores)
        after_std = np.std(after_scores)
        
        improvement = {
            'before': {'mean': before_mean, 'std': before_std},
            'after': {'mean': after_mean, 'std': after_std},
            'reduction': {
                'mean': (before_mean - after_mean) / before_mean * 100,
                'std': (before_std - after_std) / before_std * 100
            }
        }
        
        return improvement