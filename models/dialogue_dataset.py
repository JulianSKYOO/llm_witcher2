from torch.utils.data import Dataset
import torch

class DialogueDataset(Dataset):
    def __init__(self, dialogues, tokenizer, max_length=512):
        self.dialogues = dialogues
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.dialogues)

    def __getitem__(self, idx):
        dialogue = self.dialogues[idx]
        
        # Format dialogue pairs
        prompt = f"User: {dialogue['user_input']}\nAssistant:"
        response = dialogue['response']
        
        # Tokenize inputs
        prompt_tokens = self.tokenizer(
            prompt,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        response_tokens = self.tokenizer(
            response,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'prompt': prompt,
            'response': response,
            'input_ids': prompt_tokens['input_ids'].squeeze(),
            'attention_mask': prompt_tokens['attention_mask'].squeeze(),
            'labels': response_tokens['input_ids'].squeeze()
        }

def create_dialogue_dataloader(dialogues, tokenizer, batch_size=4):
    """Create DataLoader for dialogue dataset"""
    dataset = DialogueDataset(dialogues, tokenizer)
    
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_dialogues
    )

def collate_dialogues(batch):
    """Collate dialogues for batch processing"""
    return {
        'prompt': [item['prompt'] for item in batch],
        'response': [item['response'] for item in batch],
        'input_ids': torch.stack([item['input_ids'] for item in batch]),
        'attention_mask': torch.stack([item['attention_mask'] for item in batch]),
        'labels': torch.stack([item['labels'] for item in batch])
    }