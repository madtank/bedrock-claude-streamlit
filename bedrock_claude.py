# Author: madtank10
# Date: 20223-10-13

import boto3
import json
import os
import sys
import streamlit as st

from langchain.chains import ConversationChain
from langchain.llms.bedrock import Bedrock
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains.conversational_retrieval.prompts import CONDENSE_QUESTION_PROMPT


# Streamlit title
st.title("AWS Bedrock with Claude")
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
        print(f"[Debug] Persona Changed to: {persona_option}")
    # Now set the global variable persona based on session state
    persona = st.session_state.persona

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

# Model ID for the Claude model
# anthropic.claude-instant-v1
# 7x more expensive than instant-v1
# anthropic.claude-v1
# anthropic.claude-v2
# AWS region for the model
region = "us-west-2"
# Instantiate the model
cl_llm = Bedrock(
    model_id="anthropic.claude-instant-v1", 
    region_name=region,
    model_kwargs={"max_tokens_to_sample": 500}
    )

# Memory for the conversation
memory = ConversationBufferMemory(ai_prefix="Assistant") # Needs ai_prefix for Claude
conversation = ConversationChain(
    llm=cl_llm, verbose=False, memory=memory
    )

# If the messages and conversation memory are not in session state, initialize them
if "messages" not in st.session_state:
    st.session_state.messages = []
if "conversation_memory" not in st.session_state:
    st.session_state.conversation_memory = memory

# Update the conversation's memory with the session state's memory
conversation.memory = st.session_state.conversation_memory
print(f"[Debug] Conversation Memory: {st.session_state.conversation_memory}")
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Update the prompt template
claude_prompt = PromptTemplate.from_template(f"""Human: {prompt}

Assistant: Acknowledged. I will respond as a {persona}.

Current conversation:
{{history}}

Human: {{input}}

Assistant:
""")

# Update the conversation's prompt template
conversation.prompt = claude_prompt

# Define a function to get the AI's response from a query
def get_answer_from_query(query):
    result = conversation.predict(input=query)
    print(f"[Debug] Latest Query: {query}")
    print(f"[Debug] Latest Response: {result}")
    return result

# Check if user input is received from the chat
if prompt := st.chat_input("Talking with an AI, ask anything."):
    # Append user message to the session's message list
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Render user's message in the chat interface
    with st.chat_message("user"):
        st.markdown(prompt)

    # Render AI's response in the chat interface
    with st.chat_message("assistant"):
        # Get response from AI based on user's query
        response = get_answer_from_query(prompt)
        st.markdown(response)

    # Append AI's response to the session's message list
    st.session_state.messages.append({"role": "assistant", "content": response})

    # Update the session's conversation memory after the interaction
    st.session_state.conversation_memory = conversation.memory

    # Initialize 'previous_persona' in session state if it doesn't exist
    if 'previous_persona' not in st.session_state:
        st.session_state.previous_persona = None

    # Update 'previous_persona' in session state if a new persona option is selected
    if st.session_state.previous_persona != persona_option:
        st.session_state.previous_persona = persona_option
