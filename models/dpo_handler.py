import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    GenerationConfig
)
from peft import LoraConfig, TaskType, get_peft_model
from trl import DPOTrainer
from tqdm import tqdm
import pandas as pd
from datasets import Dataset
from config import CONFIG
from utils.helpers import Timer, LengthSampler

class DPOModelHandler:
    def __init__(self, performance_tracker=None):
        self.base_model_name = CONFIG["base_model"]
        self.device = CONFIG["device"]
        self.performance_tracker = performance_tracker
        self.setup_model()
        
        self.output_length_sampler = LengthSampler(
            CONFIG["output_min_length"],
            CONFIG["output_max_length"]
        )

    def setup_model(self):
        """Initialize model and tokenizer"""
        with Timer(self.performance_tracker, 'model_load_time'):
            try:
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.base_model_name,
                    torch_dtype=torch.float32 if self.device == "cpu" else torch.float16,
                    device_map={"": self.device}
                )
                
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.base_model_name,
                    padding_side="left",
                    model_max_length=CONFIG["max_length"]
                )
                
                if not self.tokenizer.pad_token:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                    
                self.model.config.use_cache = True
                
            except Exception as e:
                raise Exception(f"Error loading model: {str(e)}")

    def prepare_for_dpo(self):
        """Configure model with LoRA for DPO training"""
        peft_config = LoraConfig(
            r=4,
            lora_alpha=8,
            target_modules=["c_attn"],
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.CAUSAL_LM
        )
        self.model = get_peft_model(self.model, peft_config)
        return self.model

    def prepare_dpo_dataset(self, dialogue_pairs, max_samples=1000):
        """Prepare dataset for DPO training"""
        formatted_data = []
        
        for pair in tqdm(dialogue_pairs[:max_samples], desc="Preparing DPO dataset"):
            if 'user_input' in pair and 'response' in pair:
                prompt = f"User: {pair['user_input']}\nAssistant:"
                chosen = pair['response']
                
                # Generate alternative response
                alternatives = self.generate_responses(prompt, num_responses=2)
                rejected = alternatives[1] if len(alternatives) > 1 else alternatives[0]
                
                formatted_data.append({
                    'prompt': prompt,
                    'chosen': chosen,
                    'rejected': rejected
                })
        
        return Dataset.from_pandas(pd.DataFrame(formatted_data))

    def generate_responses(self, prompt, num_responses=2):
        """Generate multiple responses for a prompt"""
        with Timer(self.performance_tracker, 'response_time'):
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=CONFIG["max_length"]
            )
            
            if self.device != "cpu":
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            responses = []
            for _ in range(num_responses):
                max_new_tokens = self.output_length_sampler()
                
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        **CONFIG["generation_kwargs"]
                    )
                    response = self.tokenizer.decode(
                        outputs[0],
                        skip_special_tokens=True
                    )
                    responses.append(response)
            
            return responses

    def train_with_dpo(self, train_dataset, toxicity_evaluator, output_dir=CONFIG["model_path"]):
        """Train model using DPO"""
        with Timer(self.performance_tracker, 'training_time'):
            try:
                # Initialize training arguments
                training_args = Trainer.TrainingArguments(
                    output_dir=output_dir,
                    num_train_epochs=1,
                    per_device_train_batch_size=CONFIG["batch_size"],
                    learning_rate=CONFIG["learning_rate"],
                    logging_steps=10,
                    save_strategy="steps",
                    save_steps=50,
                    evaluation_strategy="no",
                    gradient_accumulation_steps=1
                )

                # Initialize DPO trainer
                dpo_trainer = DPOTrainer(
                    model=self.model,
                    args=training_args,
                    beta=0.1,
                    train_dataset=train_dataset,
                    tokenizer=self.tokenizer,
                    max_prompt_length=128,
                    max_length=512
                )

                # Training
                dpo_trainer.train()
                
                # Save model
                self.model.save_pretrained(output_dir)
                self.tokenizer.save_pretrained(output_dir)
                
                return dpo_trainer

            except Exception as e:
                raise Exception(f"Error in DPO training: {str(e)}")

    def generate_response(self, input_text, temperature=0.7, max_tokens=100):
        """Generate a single response"""
        with Timer(self.performance_tracker, 'response_time'):
            try:
                inputs = self.tokenizer(
                    input_text,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=CONFIG["max_length"]
                )
                
                if self.device != "cpu":
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                generation_config = GenerationConfig(
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    **CONFIG["generation_kwargs"]
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