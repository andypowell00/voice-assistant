import speech_recognition as sr
import pyttsx3
import openai
import os
from config.constants import *
from langchain import OpenAI, ConversationChain, LLMChain, PromptTemplate
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv


load_dotenv()

#llm(s)


# Setup OpenAI API credentials & llm
openai.api_key = os.getenv(OPEN_API_ENV_NAME)
llm = OpenAI(temperature=0)

#store chat memory
chat_memory = ConversationBufferMemory(input_key='question', memory_key='chat_history')

prompt = PromptTemplate(input_variables=["history", "input"], template=ASSISTANT_TEMPLATE)


llm_chain = LLMChain(
    llm,
    prompt,
    verbose=True,
    memory=chat_memory
)

# Create a recognizer object
r = sr.Recognizer()

# Create a text-to-speech engine
engine = pyttsx3.init()

def listen():
    with sr.Microphone() as source:
        print("Listening...")
        audio = r.listen(source)
   
        try:
            text = r.recognize_google(audio)
            print("You said:", text.lower())
            return text.lower()
        except sr.UnknownValueError:
            print("Sorry, I couldn't understand you.")
            main()
        except sr.RequestError as e:
            print("Sorry, an error occurred while processing your request:", str(e))
            main()

def speak(text):
    engine.say(text)
    engine.runAndWait()

def process_command(command):
    if WAKE_COMMAND in command:
        speak("Hello Andy! How can I help?")
        while True:
            command = listen()
            if SLEEP_COMMAND in command:
                speak("Ok bye!")
                break
            else: #run agent
                response = llm_chain.predict(command)
                print(response)
                speak(response)
                
    else:
        speak("You can wake me up by saying 'arturo'")

def main():
    while True:
        command = listen()
        process_command(command)

if __name__ == "__main__":
    main()