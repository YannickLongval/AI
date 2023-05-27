"""
Langchain agent to determine which tools are needed to create accurate outputs to the user's prompts.
"""

from langchain.chat_models import ChatOpenAI
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
import os
from langchain.agents import initialize_agent
from HandDetectorTool import HandDetectorTool

# initialize LLM (we use ChatOpenAI because we'll later define a `chat` agent)
llm = ChatOpenAI(
        openai_api_key=os.environ["OPENAI_API_KEY"],
        temperature=0,
        model_name='gpt-3.5-turbo'
)

# initialize conversational memory
conversational_memory = ConversationBufferWindowMemory(
        memory_key='chat_history',
        k=5,
        return_messages=True
)

# creating list of tools
tools = [HandDetectorTool()]

# initialize agent with tools
agent = initialize_agent(
    agent='chat-conversational-react-description',
    tools=tools,
    llm=llm,
    verbose=True,
    max_iterations=3,
    early_stopping_method='generate',
    memory=conversational_memory
)


while True:
    # get input from user and pass it into the agent to determine which tools to uses
    agent(input("How may I assist you: "))