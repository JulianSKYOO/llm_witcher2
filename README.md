# LLM with Witcher 2 Dialogue

You can generate a chat with an NPC in Witcher 2 with this LLM-powered Chatbot.

* Currently there is only Triss

### How to run?

1. Download the `requirements.txt` and `main.py` files.
2. Open your VScode.
3. This code is based on Python version 3.10, to create an environtment `conda create -p venv python==3.10 -y`.
4. you need to activate your own environment, `conda activate venv/`.
5. at your .env file, type your LANGCHAIN_API_KEY, LANGCHAIN_PROJECT, HF_API_KEY, and OPEN_AI_KEY.
* If you woud like to use any free version of the model, you have to modify the file and add the corresponding api key.
5. `pip install -r requirements.txt` to download all the libraries *Note that there are many libraries do not correspond to the current task.
6. Change your directory to the corresponding `main.py` file.
7. run `streamlit run main.py`.
8. Enjoy using it!


### Preview
![스크린샷 2024-10-27 195847](https://github.com/user-attachments/assets/08042061-3db1-4e2c-bf21-29b8070d79c7)
![스크린샷 2024-10-27 203038](https://github.com/user-attachments/assets/729e3027-83af-4a30-99b0-0cd6d1da72cc)
![스크린샷 2024-10-27 203246](https://github.com/user-attachments/assets/d870db95-6676-4361-91b7-3ac0aa907537)
