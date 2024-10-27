import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import WebBaseLoader
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_core.prompts import MessagesPlaceholder
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
import bs4
import os
from dotenv import load_dotenv

load_dotenv()

# Langsmith Tracing
os.environ['LANGCHAIN_API_KEY'] = os.getenv("LANGCHAIN_API_KEY")
os.environ['LANGCHAIN_TRACING_V2'] = "true"
os.environ['LANGCHAIN_PROJECT'] = "GenAI_OPENAI"

# OpenAI
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# HuggingFace
os.environ["HF_API_KEY"] = os.getenv("HF_API_KEY")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Witcher 2 dialogue doc
loader = WebBaseLoader(
    web_paths = ("https://laurelnose.github.io/tutorial/",
                 "https://laurelnose.github.io/prologue/",
                 "https://laurelnose.github.io/chapter-1/",
                 "https://laurelnose.github.io/chapter-2-iorveth/",
                 "https://laurelnose.github.io/chapter-2-roche/",
                 "https://laurelnose.github.io/chapter-3-roche/",
                 "https://laurelnose.github.io/epilogue/",
                 ),
    bs_kwargs = dict(
        parse_only=bs4.SoupStrainer(
            class_=("dialogue", "stage", "indent", "choice")
        )
    ),
)

docs = loader.load()

# Split the text
text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=120)
splits = text_splitter.split_documents(docs)
vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings, persist_directory="./chromaDB")
retriever = vectorstore.as_retriever()

store = {}

def get_session_history(session_id: str)->BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

configs = {"configurable": {"session_id": "Triss"}}

# Prompt Template
contextualise_system_prompt = (
    "Give a chat history and the latest user question"
    "which might reference context in the chat history."
    "formulate a standalone question which can be understandable."
    "Do not answer the question, reformulate it if needed."
    "otherwise, return it as it is. Use maximum three"
    "sentences to keep the answers concise."
)
contextualise_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualise_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("user", "{input}"),
    ]
)

system_prompt = (
"You are an NPC, Triss."
"Use the retrieved context and the chat history to carry on"
"the dialogue. The user is Geralt."
" Use maximum three sentences to"
"keep the answers concise."
"\n\n"
"{context}"
)
qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("user", "{input}"),
    ]
)

def generate_response(input_text, llm, temperature, max_tokens):
    llm = ChatOpenAI(model=llm,
                     temperature=temperature,
                     max_tokens=max_tokens,
                     )
    history_retriever = create_history_aware_retriever(llm, retriever, contextualise_prompt)
    qa_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_retriever, qa_chain)
    chat_with_history = RunnableWithMessageHistory(rag_chain,  
                                                        get_session_history,
                                                        input_messages_key="input",
                                                        history_messages_key="chat_history",
                                                        output_messages_key="answer", 
                                                        )
    response = chat_with_history.invoke({"input": input_text},
                            config=configs)
    return response["answer"]

# Streamlit config
st.title("Witcher 2 dialogue Chatbot with OPENAI")
st.sidebar.title("Settings")
#api_key = st.sidebar.text_input("Enter your OPEN AI API KEY:", type="password")

# Streamlit OpenAI models
llm = st.sidebar.selectbox("Select an Open AI Model.", ["gpt-4o", "gpt-4-turbo", "gpt-4"])
npc = st.sidebar.selectbox("Select an NPC.", ["Triss"])

# Streamlit parameter
temperature = st.sidebar.slider("Temperature", min_value=0.0, max_value=1.0, value=0.7)
max_tokens = st.sidebar.slider("Max_tokens", min_value=50, max_value=300, value=160)

# Streamlit Interface
st.write("What do you want to say?")
user_input = st.text_input("Geralt:")

if user_input:
    response = generate_response(user_input, llm, temperature, max_tokens)
    st.write(response)
else:
    st.write("Please provide what you want to say.")