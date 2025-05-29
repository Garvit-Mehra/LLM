# Custom Chatbot
This is personalized chatbot that utilizes the RAG system that can answer questions on your own private dataset
- Note: Backend uses OpenAI models, the required model can be selected within the ```Chat_Interface.py```file

## Setting up
Create a ```.env``` file where you store your OpenAI api key as ```OPENAI_API_KEY```

## Running
In your commmand line interface run
```python
git clone https://github.com/Garvit-Mehra/LLM
cd LLM
pip install -r Requirements.txt
cd Chatbot
streamlit run Chatbot.py
```
## Version History
v0.0 : Basic conversational capabilities

v0.1 : Added ability to interpret text-based PDFs (not text in images)

v0.2 : Added ability to understand images and OCR capabilities (Images have to be present inside a .pdf file)

## Future Plans
I'm planning to add multi file type uplaod system
