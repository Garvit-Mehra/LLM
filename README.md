# Custom Chatbot
- Note: Backend uses OpenAI models, the required model can be selected within the file

## Setting up
Create a ```.env``` file where you store your OpenAI api key as ```OPENAI_API_KEY```

## Running
In your commmand line interface run
```python
git clone https://github.com/Garvit-Mehra/LLM
cd LLM
pip install -r Requirements.txt
streamlit run Chatbot.py
```
## Version History
v0.0 : Basic conversational capabilities
v0.1 : Added ability to interpret text-based PDFs (not text in images)

## Future Plans
I'm planning to add image capaclities, where the chatbot will understand images and can answer question about them
