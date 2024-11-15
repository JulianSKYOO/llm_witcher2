# LLM with Witcher 2 Dialogue

You can generate a chat with an NPC in Witcher 2 with this LLM-powered Chatbot.

* Currently there is only Triss

### Simple Model

1. Download the `requirements_num.txt` and `main_num.py` files.
2. There are several cloning needed `git clone http://github.com/huggingface/trl.git`
3. Open your VScode.
4. This code is based on Python version 3.10, to create an environtment `conda create -p venv python==3.10 -y`.
5. you need to activate your own environment, `conda activate venv/`.
6. at your .env file, type your LANGCHAIN_API_KEY, LANGCHAIN_PROJECT, HF_API_KEY, and OPEN_AI_KEY.
* If you woud like to use any free version of the model, you have to modify the file and add the corresponding api key.
5. `pip install -r requirements.txt` (or `pip install -r requirements_num.txt`) to download all the libraries *Note that there are many libraries do not correspond to the current task for requirements.txt.
6. Change your directory to the corresponding `main_num.py` file.
7. run `streamlit run main_num.py`.

### Applying PPO, DPO option
1. Download folders `models` and `utils`, and `config.py` `app.py`.
2. The process is same as above, run `streamlit run app.py`.

### Preview
##### main.py
![스크린샷 2024-10-27 203246](https://github.com/user-attachments/assets/d870db95-6676-4361-91b7-3ac0aa907537)

#### main3.py
![스크린샷 2024-11-09 180834](https://github.com/user-attachments/assets/15db1cf5-e471-459b-a1de-5968ac555da0)

#### app.py
![스크린샷 2024-11-10 114116](https://github.com/user-attachments/assets/8fb582d0-f08a-4411-b636-a796caaca926)

#### References

https://github.com/huggingface/trl

https://github.com/artidoro/qlora

facebook/roberta-hate-speech-dynabench-r4-target

#### Caveat

TRL needs to be adjusted
