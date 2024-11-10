import torch

CONFIG = {
    "base_model": "distilgpt2",
    "toxicity_model": "facebook/roberta-hate-speech-dynabench-r4-target",
    "embeddings_model": "all-MiniLM-L6-v2",
    "max_length": 512,
    "chunk_size": 600,
    "chunk_overlap": 120,
    "device": 0 if torch.cuda.is_available() else "cpu",
    "model_path": "./ppo_model",
    
    # Training settings
    "output_min_length": 100,
    "output_max_length": 400,
    "batch_size": 4,
    "learning_rate": 1.41e-5,
    "max_steps": 10,
    
    # Generation settings
    "generation_kwargs": {
        "min_length": 5,
        "top_k": 0.0,
        "top_p": 1.0,
        "do_sample": True
    },
    
    # Reward settings
    "reward_kwargs": {
        "top_k": None,
        "function_to_apply": "none",
        "batch_size": 16
    },

    # PPO specific settings
    "ppo_config": {
        "learning_rate": 1.41e-5,
        "mini_batch_size": 4,
        "batch_size": 16,
        "gradient_accumulation_steps": 1
    }
}
