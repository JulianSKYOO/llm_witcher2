import os
import streamlit as st
import time
from dotenv import load_dotenv
import shutil
from models.dialogue_system import DialogueSystem
from models.dpo_handler import DPOModelHandler
from models.ppo_handler import PPOModelHandler
from models.toxicity_evaluator import ToxicityEvaluator
from config import CONFIG
from utils.helpers import format_time, PerformanceTracker, Timer

# Load environment variables
load_dotenv()

class StreamlitInterface:
    def __init__(self):
        """Initialize Streamlit interface"""
        st.set_page_config(
            page_title="Witcher Dialogue AI",
            layout="wide"
        )
        
        # Initialize performance tracking
        if 'performance_tracker' not in st.session_state:
            st.session_state.performance_tracker = PerformanceTracker()
        
        self.initialize_session_state()
        self.setup_sidebar()
        self.setup_main_content()

    def initialize_session_state(self):
        """Initialize session state variables"""
        if 'initialized' not in st.session_state:
            st.session_state.initialized = True
            st.session_state.dialogue_system = DialogueSystem(
                st.session_state.performance_tracker
            )
            st.session_state.toxicity_evaluator = ToxicityEvaluator(
                st.session_state.performance_tracker
            )
            st.session_state.chat_history = []
            st.session_state.current_model = None
            st.session_state.training_results = None

    def setup_sidebar(self):
        """Setup sidebar components"""
        with st.sidebar:
            st.title("Model Settings")
            
            # Model selection
            self.model_settings = {
                "llm": st.selectbox(
                    "Base Language Model",
                    ["gpt-3.5-turbo", "gpt-4"],
                    help="Choose the base language model"
                ),
                "temperature": st.slider(
                    "Temperature",
                    0.0, 1.0, 0.7,
                    help="Controls response creativity"
                ),
                "max_tokens": st.slider(
                    "Max Tokens",
                    50, 300, 160,
                    help="Maximum length of responses"
                )
            }
            
            # Training settings
            st.subheader("Training Settings")
            self.training_settings = {
                "method": st.selectbox(
                    "Training Method",
                    ["DPO", "PPO"],
                    help="Choose training method"
                ),
                "enabled": st.checkbox(
                    "Enable Training",
                    help="Enable model fine-tuning"
                )
            }
            
            if self.training_settings["enabled"]:
                self.training_settings.update({
                    "num_samples": st.number_input(
                        "Training Samples",
                        100, 5000, 1000,
                        help="Number of dialogue samples"
                    ),
                    "max_steps": st.number_input(
                        "Max Steps",
                        1, 100, 10,
                        help="Maximum training steps"
                    ) if self.training_settings["method"] == "PPO" else None
                })
                
                # Training button
                if st.button(f"Start {self.training_settings['method']} Training"):
                    self.handle_training()
            
            # Display performance metrics
            self.display_performance_metrics()

    def setup_main_content(self):
        """Setup main content area"""
        st.title("Witcher 2 Dialogue Chatbot")
        
        # Display chat interface
        self.display_chat_history()
        
        # User input
        user_input = st.text_input(
            "Geralt:",
            key="user_input",
            help="Type your message as Geralt"
        )
        
        if user_input:
            self.handle_user_input(user_input)
        
        # Utility buttons
        self.create_utility_buttons()

    def handle_training(self):
        """Handle model training based on selected method"""
        with st.spinner(f"Training model with {self.training_settings['method']}..."):
            try:
                # Prepare training data
                dialogue_pairs = st.session_state.dialogue_system.prepare_dialogue_data(
                    max_samples=self.training_settings["num_samples"]
                )
                
                if self.training_settings["method"] == "DPO":
                    # DPO training
                    model_handler = DPOModelHandler(st.session_state.performance_tracker)
                    model = model_handler.prepare_for_dpo()
                    dataset = model_handler.prepare_dpo_dataset(dialogue_pairs)
                    
                    results = model_handler.train_with_dpo(
                        dataset,
                        st.session_state.toxicity_evaluator
                    )
                    st.session_state.current_model = model_handler
                    
                else:
                    # PPO training
                    model_handler = PPOModelHandler(st.session_state.performance_tracker)
                    dataset = model_handler.prepare_dataset(dialogue_pairs)
                    
                    results = model_handler.train_with_ppo(
                        dataset,
                        st.session_state.toxicity_evaluator,
                        max_steps=self.training_settings["max_steps"]
                    )
                    st.session_state.current_model = model_handler
                
                st.session_state.training_results = results
                st.success(f"{self.training_settings['method']} training completed!")
                
            except Exception as e:
                st.error(f"Error during training: {str(e)}")

    def handle_user_input(self, user_input):
        """Process user input and generate response"""
        with Timer(st.session_state.performance_tracker, 'response_time'):
            try:
                # Add user message
                st.session_state.chat_history.append({
                    "role": "user",
                    "content": user_input,
                    "timestamp": time.time()
                })
                
                # Generate response
                response, toxicity_score = st.session_state.dialogue_system.generate_response(
                    user_input,
                    self.model_settings["llm"],
                    self.model_settings["temperature"],
                    self.model_settings["max_tokens"],
                    st.session_state.current_model
                )
                
                # Add response
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": response,
                    "timestamp": time.time(),
                    "toxicity_score": toxicity_score
                })
                
            except Exception as e:
                st.error(f"Error generating response: {str(e)}")

    def display_chat_history(self):
        """Display chat history"""
        st.write("Chat History:")
        for message in st.session_state.chat_history:
            with st.container():
                if message["role"] == "user":
                    st.write("Geralt: " + message["content"])
                else:
                    col1, col2 = st.columns([4, 1])
                    with col1:
                        st.write("Triss: " + message["content"])
                    with col2:
                        if "toxicity_score" is not None:
                            st.caption(f"Toxicity: {message.get('toxicity_score', 'N/A')}")

    def display_performance_metrics(self):
        """Display performance metrics"""
        st.subheader("Performance Metrics")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric(
                "Average Response Time",
                format_time(st.session_state.performance_tracker.get_average('response_time'))
            )
            st.metric(
                "Average Toxicity",
                f"{st.session_state.performance_tracker.get_average('toxicity_scores'):.4f}"
            )
            
        with col2:
            st.metric(
                "Latest Response Time",
                format_time(st.session_state.performance_tracker.get_latest('response_time'))
            )
            st.metric(
                "Latest Toxicity",
                f"{st.session_state.performance_tracker.get_latest('toxicity_scores'):.4f}"
            )

    def create_utility_buttons(self):
        """Create utility buttons"""
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("Clear Chat History"):
                st.session_state.chat_history = []
                st.session_state.dialogue_system.clear_history()
                st.experimental_rerun()
        
        with col2:
            if st.button("Reset Model"):
                st.session_state.current_model = None
                if os.path.exists(CONFIG["model_path"]):
                    shutil.rmtree(CONFIG["model_path"])
                st.experimental_rerun()
        
        with col3:
            if st.button("Export Chat History"):
                self.export_chat_history()

    def export_chat_history(self):
        """Export chat history to file"""
        if st.session_state.chat_history:
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            filename = f"chat_history_{timestamp}.txt"
            
            with open(filename, 'w', encoding='utf-8') as f:
                for message in st.session_state.chat_history:
                    role = "Geralt" if message["role"] == "user" else "Triss"
                    f.write(f"{role}: {message['content']}\n")
                    if "toxicity_score" in message:
                        f.write(f"Toxicity Score: {message['toxicity_score']}\n")
                    f.write("-" * 50 + "\n")
            
            st.success(f"Chat history exported to {filename}")

    def display_training_results(self):
        """Display training results if available"""
        if st.session_state.training_results:
            st.subheader("Training Results")
            
            results = st.session_state.training_results
            
            # Display metrics
            col1, col2 = st.columns(2)
            with col1:
                st.metric(
                    "Initial Toxicity",
                    f"{results['metrics']['initial']['mean']:.4f}"
                )
                st.metric(
                    "Training Time",
                    format_time(st.session_state.performance_tracker.get_latest('training_time'))
                )
            
        with col2:
            st.metric(
                "Final Toxicity",
                f"{results['metrics']['final']['mean']:.4f}"
            )
            improvement = ((results['metrics']['initial']['mean'] - 
                            results['metrics']['final']['mean']) / 
                            results['metrics']['initial']['mean'] * 100)
            st.metric(
                "Improvement",
                f"{improvement:.1f}%"
            )

def main():
    """Main application entry point"""
    try:
        # Set environment variables
        os.environ['USER_AGENT'] = 'Witcher-DPO-Chatbot/1.0'
        os.environ['TOKENIZERS_PARALLELISM'] = 'false'  # Avoid warnings
        
        # Initialize interface
        interface = StreamlitInterface()
        
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.error("Please refresh the page and try again.")

if __name__ == "__main__":
    main()