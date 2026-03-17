import streamlit as st
import speech_recognition as sr
import pyttsx3
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.prompts import PromptTemplate
from langchain_ollama import OllamaLLM

# Load AI Model
llm = OllamaLLM(
    model="phi3",
    temperature=0.7,
    num_predict=100
)

# Initialize Memory (LangChain v1.0+)
if "chat_history" not in st.session_state:
    st.session_state.chat_history = ChatMessageHistory()  # Stores user-AI conversation history

if "processing" not in st.session_state:
    st.session_state.processing = False

# if st.button("🎤 Start Listening", disabled=st.session_state.processing):
#     st.session_state.processing = True

# Speech Recognition
if "recognizer" not in st.session_state:
    st.session_state.recognizer = sr.Recognizer()

recognizer = st.session_state.recognizer


# Function to Speak AI Responses
def speak(text):
    engine = pyttsx3.init()
    engine.setProperty("rate", 150)
    engine.say(str(text))
    engine.runAndWait()
    engine.stop()

if "mic" not in st.session_state:
    st.session_state.mic = sr.Microphone()

mic = st.session_state.mic

# Function to Listen to Voice Input
def listen():
    with mic as source:
        placeholder = st.empty()
        placeholder.write("🎤 Listening...")
        # recognizer.adjust_for_ambient_noise(source)

        try:
            audio = recognizer.listen(source, timeout=5, phrase_time_limit=5)
        except sr.WaitTimeoutError:
            st.write("⌛ No speech detected")
            return ""

    placeholder.empty()
    try:
        query = recognizer.recognize_google(audio)
        st.write(f"👂 You Said: {query}")
        return query.lower()
    except sr.UnknownValueError:
        st.write("🤖 Sorry, I couldn't understand. Try again!")
        return ""
    except sr.RequestError:
        st.write("⚠️ Speech Recognition Service Unavailable")
        return ""

# Define AI Chat Prompt
prompt = PromptTemplate(
    input_variables=["chat_history", "question"],
    template="""You are a helpful AI voice assistant. Give short, direct answers in 1-2 sentences.
        {chat_history}User: {question}
        Assistant:"""
)

# Function to Process AI Responses
def run_chain(question):
    # Retrieve past chat history (last 4 messages only to keep prompt small = faster)
    chat_history = st.session_state.chat_history.messages[-4:]
    chat_history_text = ""
    for msg in chat_history:
        role = "User" if msg.type == "human" else "Assistant"
        chat_history_text += f"{role}: {msg.content}\n"

    # Run AI response generation
    response = llm.invoke(prompt.format(chat_history=chat_history_text, question=question))

    # Clean response: take only the AI's first reply, ignore any hallucinated continuation
    for stop_word in ["User:", "Human:", "\nAssistant:"]:
        response = response.split(stop_word)[0]
    response = response.strip()

    # Store new user input and AI response in memory
    st.session_state.chat_history.add_user_message(question)
    st.session_state.chat_history.add_ai_message(response)

    return response

# Streamlit Web UI
st.title("🤖 AI Voice Assistant (Web UI)")
st.write("🎙️ Click the button below to speak to your AI assistant!")

# engine = st.session_state.engine

# Button to Record Voice Input
if st.button("🎤 Start Listening", key="listen_btn", disabled=st.session_state.processing):
    st.session_state.processing = True
    try:
        with st.spinner("AI Running..."):
            user_query = listen()

            if user_query:
                with st.spinner("Processing..."):
                    ai_response = run_chain(user_query)

                    st.write(f"**You:** {user_query}")
                    st.write(f"**AI:** {ai_response}")

                speak(ai_response)
    except Exception as e:
        st.error(f"Something went wrong: {e}")
    finally:
        st.session_state.processing = False

# Display Full Chat History
st.subheader("📜 Chat History")
for msg in st.session_state.chat_history.messages:
    st.write(f"**{msg.type.capitalize()}**: {msg.content}")
