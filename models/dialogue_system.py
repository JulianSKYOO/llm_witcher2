from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
import bs4
from config import CONFIG
from utils.helpers import Timer

class DialogueSystem:
    def __init__(self, performance_tracker=None):
        self.performance_tracker = performance_tracker
        self.chat_history = []
        self.setup_components()

    def setup_components(self):
        with Timer(self.performance_tracker, 'model_load_time'):
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
            persist_directory="./chromaDB"
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
            ("system", """You are Triss Merigold based on the Context below. Maintain her personality, knowledge, and mannerisms.
            The User is Geralt. If the user asks information not provided from the retrieved context, chat history, and dialogue, Say you do not know.
            Use maximum three sentences.
            Use the provided context and chat history to engage in meaningful dialogue.
            
            Context:
            {context}"""),
            MessagesPlaceholder("chat_history"),
            ("user", "{input}")
        ])

    def get_session_history(self, session_id: str):
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
                    geralt_parts = [p.split("Geralt:")[-1].strip() 
                                  for p in parts[0].split("Geralt:") 
                                  if p.strip()]
                    triss_responses = [p.strip() for p in parts[1:]]
                    
                    for geralt_part, triss_response in zip(geralt_parts, triss_responses):
                        if geralt_part and triss_response:
                            dialogues.append({
                                "user_input": geralt_part,
                                "response": triss_response,
                                "query": f"User: {geralt_part}\nAssistant: {triss_response}"
                            })
        
        return dialogues

    def generate_response(self, input_text, llm_model, temperature, max_tokens, model=None):
        """Generate response using specified model or default pipeline"""
        with Timer(self.performance_tracker, 'response_time'):
            try:
                # Use custom model if provided
                if model:
                    if hasattr(model, 'generate_response'):
                        return model.generate_response(
                            input_text,
                            temperature=temperature,
                            max_tokens=max_tokens
                        )
                    else:
                        raise ValueError("Model must implement generate_response method")
                
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

            except Exception as e:
                raise Exception(f"Error generating response: {str(e)}")

    def clear_history(self, session_id: str = None):
        """Clear chat history for specified session or all sessions"""
        if session_id:
            if session_id in self.store:
                self.store[session_id] = ChatMessageHistory()
        else:
            self.store = {}