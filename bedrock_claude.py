import boto3
import json
import os
import sys

from langchain.chains import ConversationChain
from langchain.llms.bedrock import Bedrock
from langchain import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains.conversational_retrieval.prompts import CONDENSE_QUESTION_PROMPT

region = "us-west-2"

# Instantiate the model
cl_llm = Bedrock(
    model_id="anthropic.claude-v2", 
    region_name=region,
    model_kwargs={"max_tokens_to_sample": 500}
    )
memory = ConversationBufferMemory()
conversation = ConversationChain(
    llm=cl_llm, verbose=False, memory=ConversationBufferMemory()
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
    print('Output from get_answer_from_query:', result)
    return result
