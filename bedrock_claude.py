
import boto3
import json
import os
import sys
import streamlit as st

from langchain.chains import ConversationChain
from langchain.llms.bedrock import Bedrock
from langchain import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains.conversational_retrieval.prompts import CONDENSE_QUESTION_PROMPT


st.title("AWS Bedrock with Claude V2")
region = "us-west-2"

# streamlit sidebar
# Initialize session_state for 'persona' if not already initialized
if 'persona' not in st.session_state:
    st.session_state.persona = "Friendly AI"

# Sidebar for persona selection and chat reset
with st.sidebar:
    # Choose a Persona
    persona_option = st.selectbox('Choose a Persona:', ['Friendly AI', 'Dev', 'Guru', 'Comedian'], key='persona_selectbox')
    if persona_option:
        st.session_state.persona = persona_option

    # Show selected persona and option to reset
    if st.session_state.persona:
        st.write(f"Selected Persona: {st.session_state.persona}")
        if st.button("Clear Chat"):
            st.session_state.persona = "Friendly AI"
            st.session_state.messages = []  # Clear the chat messages
            st.session_state.conversation_memory = ConversationBufferMemory(ai_prefix="Assistant")  # Reset the conversation memory

# Mapping of persona to prompt
persona_to_prompt = {
    'Friendly AI': 'The following is a friendly conversation between a human and an AI.',
    'Dev': 'I want you to act as a software developer.',
    'Guru': 'I want you to act as a yogi.',
    'Comedian': 'I want you to act as a comedian.'
}

# Display prompt based on the persona
prompt = persona_to_prompt[st.session_state.persona]

# Instantiate the model
cl_llm = Bedrock(
    model_id="anthropic.claude-v2", 
    region_name=region,
    model_kwargs={"max_tokens_to_sample": 500}
    )

# Claude needs to know the prefix of the AI's name in the conversation Assistant
memory = ConversationBufferMemory(ai_prefix="Assistant")
conversation = ConversationChain(
    llm=cl_llm, verbose=False, memory=ConversationBufferMemory(ai_prefix="Assistant")
    )

# Update the prompt template
# Update the prompt template
claude_prompt = PromptTemplate.from_template(f"""Human: {prompt}
                                             
Assistant: Yes, I will be a helpful AI and play the part.
                                                       
Current conversation:
{{history}}

Human: {{input}}

Assistant:
""")

conversation.prompt = claude_prompt
print(conversation.prompt)

def get_answer_from_query(query):
    result = conversation.predict(input=query)
    return result

# If the messages and conversation memory are not in session state, initialize them
if "messages" not in st.session_state:
    st.session_state.messages = []
if "conversation_memory" not in st.session_state:
    st.session_state.conversation_memory = memory

# Update the conversation's memory with the session state's memory
conversation.memory = st.session_state.conversation_memory

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("What is up?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        response = get_answer_from_query(prompt)
        st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})
    # Update the session state memory after the conversation
    st.session_state.conversation_memory = conversation.memory
