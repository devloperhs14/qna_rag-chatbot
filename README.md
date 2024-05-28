A PDF ChatBot powered by Google Gema & Groq Inference Engine

STEPS TO RUN

1. Clone Repo & create env
```
git clone https://github.com/devloperhs14/qna_rag-chatbot.git
```
create env
```
conda create -p env_name python ==3.12
```


3. Create a `.env` file as:
```
GROK_API_KEY = "your_groq_api_key"
GOOGLE_API_KEY = "your_google_api_key"
```

3. Install requirements
```
pip install requirements.txt
```

4. Run app
```
streamlit run app.py
```

Ask questions and get responses
Enjoy!

> To change pdfs , nativage to pdf folder and put your own pdf and generate vector embedding by pressing `Create Document Embedding` 
