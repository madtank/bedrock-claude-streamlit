
import streamlit as st
import json
import os
import sys
from langchain.chains import ConversationChain
from langchain.llms.bedrock import Bedrock
from langchain import PromptTemplate
from langchain.memory import ConversationBufferMemory

# Instantiate the model
cl_llm = Bedrock(model_id="anthropic.claude-v2", model_kwargs={"max_tokens_to_sample": 500})
memory = ConversationBufferMemory()
conversation = ConversationChain(llm=cl_llm, verbose=False, memory=memory)

# Update the prompt template
claude_prompt = PromptTemplate.from_template("""The following is a friendly conversation between a human and an AI.
The AI is helpful and provides details from its context. If the AI does not know
the answer to a question, it truthfully says it does not know. 
Human has the option to type file during the conversation followed by path on the local machine, 
this will send the contents of the file, so that human and AI can discuss. Don't provide feedback on file until asked.                                                                                     

Current conversation:
{history}

Human: {input}

Assistant:
""")
conversation.prompt = claude_prompt

# Streamlit UI for user interaction
st.title("Claude v2 Chatbot")
user_input = st.text_input("Enter your message:", "")
if user_input:
    if user_input.lower().startswith('file'):
        file_path = user_input.split(' ', 1)[1].strip()
        try:
            with open(file_path, 'r') as file:
                file_contents = file.read()
            file_contents = "User has sent file " + file_path + ". Contents for file: " + file_contents
            response = conversation.predict(input=file_contents)
        except FileNotFoundError:
            response = f"No file found at the provided path: {file_path}"
    else:
        response = conversation.predict(input=user_input)
    st.write(f"Claude v2: {response}")
