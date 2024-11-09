# main.py

import os
import streamlit as st
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.document_loaders import WebBaseLoader
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    TrainingArguments,
    GenerationConfig,
    pipeline
)
import evaluate
from datasets import Dataset
import torch
from trl import DPOTrainer
import pandas as pd
import numpy as np
from tqdm import tqdm
import bs4
import shutil

# Load environment variables
load_dotenv()

# Configuration
CONFIG = {
    "base_model": "distilgpt2",
    "toxicity_model": "facebook/roberta-hate-speech-dynabench-r4-target",
    "embeddings_model": "all-MiniLM-L6-v2",
    "max_length": 512,
    "chunk_size": 600,
    "chunk_overlap": 120,
    "device": 0 if torch.cuda.is_available() else "cpu",
    "model_path": "./dpo_model"
}

class ToxicityEvaluator:
    def __init__(self):
        self.model_name = CONFIG["toxicity_model"]
        self.device = CONFIG["device"]
        self.setup_evaluator()

    def setup_evaluator(self):
        """Initialize toxicity detection components"""
        self.sentiment_pipe = pipeline(
            "sentiment-analysis",
            model=self.model_name,
            device=self.device
        )
        
        self.toxicity_evaluator = evaluate.load(
            "toxicity",
            self.model_name,
            module_type="measurement",
            toxic_label="hate"
        )
        
        self.reward_logits_kwargs = {
            "top_k": None,
            "function_to_apply": "none",
            "batch_size": 16
        }
        
        self.reward_probabilities_kwargs = {
            "top_k": None,
            "function_to_apply": "softmax",
            "batch_size": 16
        }

    def evaluate_toxicity(self, model, tokenizer, dataset, num_samples):
        """Evaluate model generation toxicity"""
        toxicities = []
        for i, sample in tqdm(enumerate(dataset)):
            if i >= num_samples:
                break
                
            input_text = sample["prompt"]
            inputs = tokenizer(
                input_text,
                return_tensors="pt",
                padding=True
            )
            
            if self.device != "cpu":
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            outputs = model.generate(
                **inputs,
                max_new_tokens=100,
                do_sample=True,
                temperature=0.7
            )
            
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            toxicity_score = self.toxicity_evaluator.compute(
                predictions=[f"{input_text} {generated_text}"]
            )
            toxicities.extend(toxicity_score["toxicity"])

        return np.mean(toxicities), np.std(toxicities)

    def get_toxicity_scores(self, texts, use_probabilities=True):
        """Get toxicity scores for a batch of texts"""
        kwargs = (self.reward_probabilities_kwargs 
                 if use_probabilities 
                 else self.reward_logits_kwargs)
        
        results = self.sentiment_pipe(texts, **kwargs)
        return [result[0]['score'] for result in results]

class DialogueSystem:
    def __init__(self):
        self.setup_components()

    def setup_components(self):
        """Initialize dialogue system components"""
        # Load documents
        self.docs = self.load_documents()
        
        # Setup text splitting and embeddings
        splits = self.prepare_text_splits()
        self.setup_vectorstore(splits)
        
        # Setup chat components
        self.store = {}
        self.setup_prompts()

    def load_documents(self):
        """Load dialogue documents"""
        loader = WebBaseLoader(
            web_paths=(
                "https://laurelnose.github.io/tutorial/",
                "https://laurelnose.github.io/prologue/",
                "https://laurelnose.github.io/chapter-1/",
                "https://laurelnose.github.io/chapter-2-iorveth/",
                "https://laurelnose.github.io/chapter-2-roche/",
                "https://laurelnose.github.io/chapter-3-roche/",
                "https://laurelnose.github.io/epilogue/"
            ),
            bs_kwargs=dict(
                parse_only=bs4.SoupStrainer(
                    class_=("dialogue", "stage", "indent", "choice")
                )
            )
        )
        return loader.load()

    def prepare_text_splits(self):
        """Prepare text splits for embedding"""
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CONFIG["chunk_size"],
            chunk_overlap=CONFIG["chunk_overlap"]
        )
        return text_splitter.split_documents(self.docs)

    def setup_vectorstore(self, splits):
        """Setup vector store for document retrieval"""
        embeddings = HuggingFaceEmbeddings(
            model_name=CONFIG["embeddings_model"]
        )
        vectorstore = Chroma.from_documents(
            documents=splits,
            embedding=embeddings,
            persist_directory="./chromaDB3"
        )
        self.retriever = vectorstore.as_retriever()

    def setup_prompts(self):
        """Setup chat prompts"""
        self.contextualise_prompt = ChatPromptTemplate.from_messages([
            ("system", "Given a chat history and the latest user question, formulate a standalone question."),
            MessagesPlaceholder("chat_history"),
            ("user", "{input}")
        ])
        
        self.qa_prompt = ChatPromptTemplate.from_messages([
            ("system", """
            You are an NPC, Triss.Use the dialogue and the chat history to carry on
            the dialogue. The user is Geralt.
            If the user asks information not provided from the retreived context, chat history, and dialogue, Say you do not know.
            Use maximum three sentences.
            This is the dialogue. \n\n{context}
             """),
            MessagesPlaceholder("chat_history"),
            ("user", "{input}")
        ])

    def get_session_history(self, session_id: str) -> BaseChatMessageHistory:
        """Get or create chat history for session"""
        if session_id not in self.store:
            self.store[session_id] = ChatMessageHistory()
        return self.store[session_id]
    def prepare_dialogue_data(self, max_samples=1000):
            """Extract dialogue pairs from documents"""
            dialogues = []
            for doc in self.docs[:max_samples]:
                content = doc.page_content
                if "Geralt:" in content and "Triss:" in content:
                    parts = content.split("Triss:")
                    if len(parts) > 1:
                        geralt_part = parts[0].split("Geralt:")[-1].strip()
                        triss_part = parts[1].strip()
                        
                        if geralt_part and triss_part:
                            dialogues.append({
                                "user_input": geralt_part,
                                "response": triss_part
                            })
            return dialogues

    def generate_response(self, input_text, llm_model, temperature, max_tokens, dpo_model=None):
        """Generate response using either DPO or standard pipeline"""
        if dpo_model and dpo_model.model is not None:
            # Generate using DPO model
            prompt = f"User: {input_text}\nAssistant:"
            responses = dpo_model.generate_responses(prompt, num_responses=3)
            toxicity_scores = dpo_model.toxicity_evaluator.get_toxicity_scores(
                responses,
                use_probabilities=True
            )
            
            best_response_idx = np.argmax(toxicity_scores)
            response = responses[best_response_idx]
            toxicity_score = toxicity_scores[best_response_idx]
            
            if "\nAssistant:" in response:
                response = response.split("\nAssistant:")[-1].strip()
            return response, toxicity_score
        
        # Standard LangChain generation
        llm = ChatOpenAI(
            model=llm_model,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        
        history_retriever = create_history_aware_retriever(
            llm,
            self.retriever,
            self.contextualise_prompt
        )
        
        qa_chain = create_stuff_documents_chain(
            llm,
            self.qa_prompt
        )
        
        rag_chain = create_retrieval_chain(
            history_retriever,
            qa_chain
        )
        
        chat_with_history = RunnableWithMessageHistory(
            rag_chain,
            self.get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer",
        )
        
        response = chat_with_history.invoke(
            {"input": input_text},
            config={"configurable": {"session_id": "Triss"}}
        )
        return response["answer"], None

class DPOModelHandler:
    def __init__(self):
        self.base_model_name = CONFIG["base_model"]
        self.device = CONFIG["device"]
        self.toxicity_evaluator = ToxicityEvaluator()
        self.setup_model()

    def setup_model(self):
        """Initialize language model and tokenizer"""
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
            st.error(f"Error loading model: {str(e)}")
            raise

    def prepare_for_dpo(self):
        """Configure model for DPO training"""
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

    def generate_responses(self, prompt, num_responses=2):
        """Generate multiple responses for a prompt"""
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
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=100,
                    do_sample=True,
                    temperature=0.7,
                    num_return_sequences=1
                )
                response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                responses.append(response)
        
        return responses

    def prepare_dpo_dataset(self, dialogue_pairs, max_samples=1000):
        """Prepare dataset for DPO training"""
        formatted_data = []
        
        for pair in dialogue_pairs[:max_samples]:
            if 'user_input' in pair and 'response' in pair:
                prompt = f"User: {pair['user_input']}\nAssistant:"
                responses = self.generate_responses(prompt, num_responses=3)
                
                toxicity_scores = self.toxicity_evaluator.get_toxicity_scores(
                    responses,
                    use_probabilities=True
                )
                
                response_scores = list(zip(responses, toxicity_scores))
                response_scores.sort(key=lambda x: x[1], reverse=True)
                
                if len(response_scores) >= 2:
                    formatted_data.append({
                        'prompt': prompt,
                        'chosen': response_scores[0][0],
                        'rejected': response_scores[-1][0],
                        'chosen_reward': response_scores[0][1],
                        'rejected_reward': response_scores[-1][1]
                    })
        
        return Dataset.from_pandas(pd.DataFrame(formatted_data))

    def train_with_dpo(self, train_dataset, output_dir=CONFIG["model_path"]):
        """Train model using DPO"""
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=1,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=4,
            learning_rate=5e-5,
            remove_unused_columns=False,
            log_level="error",
            fp16=(self.device != "cpu"),
            save_strategy="epoch",
            evaluation_strategy="steps",
            eval_steps=100,
            save_total_limit=1,
            load_best_model_at_end=True
        )

        dpo_trainer = DPOTrainer(
            model=self.model,
            args=training_args,
            beta=0.1,
            train_dataset=train_dataset,
            tokenizer=self.tokenizer,
            max_length=256,
            max_prompt_length=128,
            max_target_length=128
        )

        # Evaluate initial toxicity
        initial_mean, initial_std = self.toxicity_evaluator.evaluate_toxicity(
            self.model,
            self.tokenizer,
            train_dataset,
            num_samples=100
        )
        print(f"Initial toxicity - Mean: {initial_mean:.4f}, Std: {initial_std:.4f}")

        # Train model
        dpo_trainer.train()

        # Evaluate final toxicity
        final_mean, final_std = self.toxicity_evaluator.evaluate_toxicity(
            self.model,
            self.tokenizer,
            train_dataset,
            num_samples=100
        )
        print(f"Final toxicity - Mean: {final_mean:.4f}, Std: {final_std:.4f}")

        return dpo_trainer

class StreamlitInterface:
    def __init__(self):
        """Initialize Streamlit interface"""
        st.set_page_config(
            page_title="Witcher Dialogue AI",
            layout="wide"
        )
        self.initialize_session_state()

    @staticmethod
    def initialize_session_state():
        """Initialize session state variables"""
        if 'initialized' not in st.session_state:
            st.session_state.initialized = True
            st.session_state.dialogue_system = DialogueSystem()
            st.session_state.chat_history = []
            st.session_state.dpo_model = None

    def create_sidebar(self):
        """Create sidebar with model settings"""
        with st.sidebar:
            st.title("Settings")
            
            # Model settings
            model_settings = {
                "llm": st.selectbox(
                    "Select Language Model",
                    ["gpt-3.5-turbo", "gpt-4"],
                    help="Choose the base language model"
                ),
                "temperature": st.slider(
                    "Temperature",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.7,
                    help="Controls response creativity"
                ),
                "max_tokens": st.slider(
                    "Max Tokens",
                    min_value=50,
                    max_value=300,
                    value=160,
                    help="Maximum length of responses"
                )
            }
            
            # DPO settings
            st.subheader("DPO Settings")
            dpo_settings = {
                "enable_dpo": st.checkbox(
                    "Enable DPO Fine-tuning",
                    help="Use DPO for response generation"
                )
            }
            
            if dpo_settings["enable_dpo"]:
                dpo_settings["num_samples"] = st.number_input(
                    "Training Samples",
                    min_value=100,
                    max_value=5000,
                    value=1000,
                    help="Number of dialogue samples for training"
                )
                
                if st.button("Start DPO Training"):
                    self.handle_dpo_training(dpo_settings["num_samples"])
            
            return model_settings, dpo_settings

    def handle_dpo_training(self, num_samples):
        """Handle DPO model training process"""
        with st.spinner("Training model with DPO..."):
            try:
                dpo_handler = DPOModelHandler()
                model = dpo_handler.prepare_for_dpo()
                
                dialogue_pairs = st.session_state.dialogue_system.prepare_dialogue_data(
                    max_samples=num_samples
                )
                
                train_dataset = dpo_handler.prepare_dpo_dataset(
                    dialogue_pairs,
                    max_samples=num_samples
                )
                
                trained_model = dpo_handler.train_with_dpo(train_dataset)
                trained_model.save_pretrained(CONFIG["model_path"])
                
                st.session_state.dpo_model = dpo_handler
                st.success("DPO training completed successfully!")
                
            except Exception as e:
                st.error(f"Error during DPO training: {str(e)}")

    def display_chat_history(self):
        """Display chat history"""
        st.write("Chat History:")
        for message in st.session_state.chat_history:
            with st.container():
                if message["role"] == "user":
                    st.write("Geralt: " + message["content"])
                else:
                    st.write("Triss: " + message["content"])
                    if "toxicity_score" in message:
                        st.info(f"Response toxicity score: {message['toxicity_score']:.4f}")

    def handle_user_input(self, user_input, model_settings, dpo_settings):
        """Process user input and generate response"""
        if user_input:
            # Add user message to history
            st.session_state.chat_history.append({
                "role": "user",
                "content": user_input
            })
            
            # Generate response
            dpo_model = st.session_state.dpo_model if dpo_settings["enable_dpo"] else None
            response, toxicity_score = st.session_state.dialogue_system.generate_response(
                user_input,
                model_settings["llm"],
                model_settings["temperature"],
                model_settings["max_tokens"],
                dpo_model
            )
            
            # Add response to history
            response_message = {
                "role": "assistant",
                "content": response
            }
            if toxicity_score is not None:
                response_message["toxicity_score"] = toxicity_score
                
            st.session_state.chat_history.append(response_message)

    def create_utility_buttons(self):
        """Create utility buttons for chat management"""
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Clear Chat History"):
                st.session_state.chat_history = []
                st.experimental_rerun()
        
        with col2:
            if st.button("Reset DPO Model"):
                st.session_state.dpo_model = None
                if os.path.exists(CONFIG["model_path"]):
                    shutil.rmtree(CONFIG["model_path"])
                st.experimental_rerun()

    def run(self):
        """Run the Streamlit interface"""
        st.title("Witcher 2 Dialogue Chatbot with Triss")
        
        # Create sidebar and get settings
        model_settings, dpo_settings = self.create_sidebar()
        
        # Display chat interface
        self.display_chat_history()
        
        # User input
        user_input = st.text_input(
            "Geralt:",
            key="user_input",
            help="Type your message as Geralt"
        )
        
        # Handle user input
        self.handle_user_input(user_input, model_settings, dpo_settings)
        
        # Create utility buttons
        self.create_utility_buttons()

def main():
    """Main application entry point"""
    try:
        # Set up environment
        os.environ['USER_AGENT'] = 'Witcher-DPO-Chatbot/1.0'
        
        # Initialize and run interface
        interface = StreamlitInterface()
        interface.run()
        
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.error("Please refresh the page and try again.")

if __name__ == "__main__":
    main()