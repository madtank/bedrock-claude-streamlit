
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
claude_prompt = PromptTemplate.from_template("""The following is a friendly conversation between a human and an AI.
Keep the answers short and concise. If the AI does not know
the answer to a question, it truthfully says it does not know.                                                                               

Current conversation:
{history}


Human: {input}


Assistant:
""")
conversation.prompt = claude_prompt

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
