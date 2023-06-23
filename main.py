"""
Langchain agent to determine which tools are needed to create accurate outputs to the user's prompts.
"""

from langchain.chat_models import ChatOpenAI
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
import os
from langchain.agents import initialize_agent
from HandDetectorTool import HandDetectorTool
from ImageTool import ImageTool
import pyttsx3
import speech_recognition as sr

# initialize LLM (we use ChatOpenAI because we'll later define a `chat` agent)
llm = ChatOpenAI(
        openai_api_key=os.environ["OPENAI_API_KEY"],
        temperature=0,
        model_name='gpt-3.5-turbo'
)

# initializing text to speech 
engine = pyttsx3.init()
voices = engine.getProperty('voices')       #getting details of current voice
engine.setProperty('voice', voices[1].id)   #changing index, changes voices. 1 for female

# speech recognition
r = sr.Recognizer()
mic = sr.Microphone()

# initialize conversational memory
conversational_memory = ConversationBufferWindowMemory(
        memory_key='chat_history',
        k=5,
        return_messages=True
)

# creating list of tools
tools = [HandDetectorTool(), ImageTool()]

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

reponse = agent("Hi")['output']
engine.say(reponse)
engine.runAndWait()

while True:
    # with mic as source:
    #     r.adjust_for_ambient_noise(source)
    #     audio = r.listen(source)

    # get input from user and pass it into the agent to determine which tools to uses
    # reponse = agent(r.recognize_google(audio))['output']

    # To type the question rather than speak it
    reponse = agent(input("Question: "))['output']

    engine.say(reponse)
    engine.runAndWait()