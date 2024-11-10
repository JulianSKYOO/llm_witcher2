import torch
from transformers import AutoTokenizer, GenerationConfig, AutoModelForCausalLM
from trl import PPOConfig, PPOTrainer
from trl import AutoModelForSeq2SeqLMWithValueHead
from tqdm import tqdm
import numpy as np
from config import CONFIG
from utils.helpers import LengthSampler

def create_reference_model(model):
    """Create a reference model from the base model"""
    return type(model)(model.config)

class PPOModelHandler:
    def __init__(self, performance_tracker=None):
        self.base_model_name = CONFIG["base_model"]
        self.device = CONFIG["device"]
        self.performance_tracker = performance_tracker
        self.setup_models()
        self.setup_ppo_config()
        
        # Initialize length sampler
        self.output_length_sampler = LengthSampler(
            CONFIG["output_min_length"],
            CONFIG["output_max_length"]
        )

    def setup_models(self):
        """Initialize models and tokenizer"""
        try:
            base_model = AutoModelForCausalLM.from_pretrained(
                self.base_model_name,
                torch_dtype=torch.float32,
                device_map="auto"
            )
            # Setup main model
            self.model = AutoModelForSeq2SeqLMWithValueHead.from_pretrained(
                base_model,
                torch_dtype=torch.bfloat16,
                is_trainable=True
            )
            
            # Setup reference model
            self.ref_model = create_reference_model(self.model)
            
            # Setup tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.base_model_name,
                padding_side="left"
            )
            if not self.tokenizer.pad_token:
                self.tokenizer.pad_token = self.tokenizer.eos_token

        except Exception as e:
            raise Exception(f"Error setting up models: {str(e)}")

    def setup_ppo_config(self):
        """Setup PPO configuration"""
        self.ppo_config = PPOConfig(
            model_name=self.base_model_name,
            learning_rate=1.41e-5,
            ppo_epochs=1,
            mini_batch_size=4,
            batch_size=16
        )

        self.generation_kwargs = {
            "min_length": 5,
            "top_k": 0.0,
            "top_p": 1.0,
            "do_sample": True
        }

        self.reward_kwargs = {
            "top_k": None,
            "function_to_apply": "none",
            "batch_size": 16
        }

    def train(self, dataset, toxicity_evaluator, max_steps=10):
        """Train model using PPO"""
        try:
            # Initialize PPO trainer
            ppo_trainer = PPOTrainer(
                config=self.ppo_config,
                model=self.model,
                ref_model=self.ref_model,
                tokenizer=self.tokenizer,
                dataset=dataset["train"],
                data_collator=self.collate_data
            )

            # Training loop
            for step, batch in tqdm(enumerate(ppo_trainer.dataloader)):
                if step >= max_steps:
                    break

                # Get prompt tensors
                prompt_tensors = batch["input_ids"]

                # Generate responses
                summary_tensors = []
                for prompt_tensor in prompt_tensors:
                    max_new_tokens = self.output_length_sampler()
                    self.generation_kwargs["max_new_tokens"] = max_new_tokens
                    
                    summary = ppo_trainer.generate(
                        prompt_tensor,
                        **self.generation_kwargs
                    )
                    summary_tensors.append(summary.squeeze()[-max_new_tokens:])

                # Decode responses
                batch["response"] = [
                    self.tokenizer.decode(r.squeeze())
                    for r in summary_tensors
                ]

                # Compute rewards using toxicity evaluator
                query_response_pairs = [
                    q + r
                    for q, r in zip(batch["query"], batch["response"])
                ]
                
                rewards = toxicity_evaluator.sentiment_pipe(
                    query_response_pairs,
                    **self.reward_kwargs
                )

                # Convert rewards to tensors
                not_hate_index = 0
                reward_tensors = [
                    torch.tensor(reward[not_hate_index]["score"])
                    for reward in rewards
                ]

                # PPO step
                stats = ppo_trainer.step(prompt_tensors, summary_tensors, reward_tensors)
                ppo_trainer.log_stats(stats, batch, reward_tensors)

            return ppo_trainer

        except Exception as e:
            raise Exception(f"Error in PPO training: {str(e)}")

    @staticmethod
    def collate_data(data):
        """Collate data for batch processing"""
        return dict((key, [d[key] for d in data]) for key in data[0])

    def evaluate_toxicity(self, toxicity_evaluator, dataset, num_samples=10):
        """Evaluate model's toxicity levels"""
        max_new_tokens = 100
        toxicities = []

        for i, sample in enumerate(dataset["test"]):
            if i >= num_samples:
                break

            input_text = sample["query"]
            input_ids = self.tokenizer(
                input_text,
                return_tensors="pt",
                padding=True
            ).input_ids.to(self.device)

            generation_config = GenerationConfig(
                max_new_tokens=max_new_tokens,
                top_k=0.0,
                top_p=1.0,
                do_sample=True
            )

            outputs = self.model.generate(
                input_ids=input_ids,
                generation_config=generation_config
            )

            generated_text = self.tokenizer.decode(
                outputs[0],
                skip_special_tokens=True
            )

            toxicity_score = toxicity_evaluator.compute(
                predictions=[(input_text + " " + generated_text)]
            )
            toxicities.extend(toxicity_score["toxicity"])

        return np.mean(toxicities), np.std(toxicities)

    def generate_response(self, input_text, temperature=0.7, max_tokens=100):
        """Generate a response using the PPO model"""
        try:
            inputs = self.tokenizer(
                input_text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=CONFIG["max_length"]
            ).to(self.device)

            generation_config = GenerationConfig(
                max_new_tokens=max_tokens,
                temperature=temperature,
                **self.generation_kwargs
            )

            outputs = self.model.generate(
                **inputs,
                generation_config=generation_config
            )

            response = self.tokenizer.decode(
                outputs[0],
                skip_special_tokens=True
            )

            return response, None

        except Exception as e:
            raise Exception(f"Error generating response: {str(e)}")