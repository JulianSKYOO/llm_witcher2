# LLM with Witcher 2 Dialogue

You can generate a chat with an NPC in Witcher 2 with this LLM-powered Chatbot.

* Currently there is only Triss

### How to run?

1. Download the `requirements.txt` and `main.py` files. (or `requirements_num.txt` and `main_num.py`)
2. There are several cloning needed `git clone http://github.com/huggingface/trl.git`
3. Open your VScode.
4. This code is based on Python version 3.10, to create an environtment `conda create -p venv python==3.10 -y`.
5. you need to activate your own environment, `conda activate venv/`.
6. at your .env file, type your LANGCHAIN_API_KEY, LANGCHAIN_PROJECT, HF_API_KEY, and OPEN_AI_KEY.
* If you woud like to use any free version of the model, you have to modify the file and add the corresponding api key.
5. `pip install -r requirements.txt` (or `pip install -r requirements_num.txt`) to download all the libraries *Note that there are many libraries do not correspond to the current task for requirements.txt.
6. Change your directory to the corresponding `main.py` file (or `main_num.py`).
7. run `streamlit run main.py` (or `stream lit run main_num.py`).


### Preview
##### main.py
![스크린샷 2024-10-27 203246](https://github.com/user-attachments/assets/d870db95-6676-4361-91b7-3ac0aa907537)

#### main3.py
![스크린샷 2024-11-09 173041](https://github.com/user-attachments/assets/f185c8ef-0c0f-480e-8940-bfa8503c8fc0)

**References**

https://github.com/huggingface/trl

https://github.com/artidoro/qlora

facebook/roberta-hate-speech-dynabench-r4-target
