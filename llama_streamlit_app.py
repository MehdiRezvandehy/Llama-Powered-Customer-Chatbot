"""Streamlit app for question-answering using llama"""

# Import packages required across different parts of app
import warnings
warnings.filterwarnings('ignore')

# import streamlit
import streamlit as st
from streamlit_chat import message

import pandas as pd
pd.set_option('display.max_colwidth', None)
import matplotlib.pyplot as plt


# set up page configuration
st.set_page_config(page_title="Llama-Powered Customer Chatbot",
    page_icon="🤖",
    layout="wide",
    #layout="centered"
)

st.title('Llama-Powered Customer Chatbot 🤖')

# Typing effect that stops at the author's name length and repeats from the beginning
st.markdown(
    """
    <style>
        .author-title {
            font-size: 1.3em;
            font-weight: bold;
            color: #007acc; /* Color for "Author:" */
            white-space: nowrap;
            vertical-align: middle; /* Ensures alignment with animated text */
        }

        .author-name {
            font-size: 1.2em;
            font-weight: bold;
            color: red; /* Color for the author's name */
            overflow: hidden;
            white-space: nowrap;
            border-right: 3px solid;
            display: inline-block;
            vertical-align: middle; /* Aligns with the static "Author:" text */
            animation: typing 5s steps(20, end) infinite, blink-caret 0.75s step-end infinite;
            max-width: 10ch; /* Limit width to fit text length */
        }

        /* Typing effect */
        @keyframes typing {
            0% { max-width: 0; }
            50% { max-width: 30ch; } /* Adjust to match the name's length */
            100% { max-width: 0; } /* Reset back to zero */
        }

        /* Blinking cursor animation for the author's name */
        @keyframes blink-caret {
            from, to { border-color: transparent; }
            50% { border-color: red; }
        }
    </style>

    <p><span class="author-title">Author:</span> \
    <span class="author-name">Mehdi Rezvandehy</span></p>

    """,
    unsafe_allow_html=True
)

# Table of contents
st.sidebar.title("Table of Contents")
st.sidebar.markdown("[**Dataset**](#dataset)")
st.sidebar.markdown("[**Select Llama Model Configuration**](#select-llama-model-configuration)")
st.sidebar.markdown("[**Question-Answering Chatbot**](#question-answering-chatbot)")


st.markdown("<br>", unsafe_allow_html=True)
st.markdown("""
This app, built with [Streamlit](https://streamlit.io/), serves as a question-answering chatbot 
designed for customer support. The **TinyLlama-1.1B-Chat-v1.0** model, which has been fine-tuned using labeled data.
"""
)
st.header(' Dataset')
st.markdown("""
The [Bitext Customer Support LLM Chatbot Training](https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset) 
dataset is a comprehensive resource for developing conversational AI models tailored to customer service needs. It features fields such as `instruction` 
(representing user queries), `response` (providing model replies), `category` (for semantic grouping), and `intent` 
(to capture specific user intents). The dataset supports various customer service scenarios, including account management, 
refunds, invoices, and order processing. It consists of structured question-answer pairs created through a hybrid 
approach, combining natural language processing with generation techniques, and curated by computational linguists. This dataset 
was used to fine-tune **TinyLlama-1.1B-Chat-v1.0**.

"""
)

st.image('bitext_customer.jpg')


# Custom CSS for centering and resizing the button
st.markdown("""
    <style>
    .stButton > button {
        padding: 1rem 5rem;
        font-size: 16px !important; /* Forces font size */
        font-weight: bold; /* Makes font bolder */
    }
    </style>
    """, unsafe_allow_html=True)

if 'load_fine_tuned_model' not in st.session_state:
    with st.spinner("Loading fine-tuned model..."):
        st.session_state.load_fine_tuned_model = True
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        # Specify the path where you saved the model and tokenizer
        model_path = "saved_model"
        
        # Load the tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        st.session_state.tokenizer = tokenizer

        # Load the model
        model_fine_tuned = AutoModelForCausalLM.from_pretrained(model_path)
        st.session_state.model_fine_tuned = model_fine_tuned

st.header("Select Llama Model Configuration")
if st.session_state.load_fine_tuned_model:
    from transformers import pipeline
    from langchain.llms import HuggingFacePipeline
    # run a simple LLMChain
    from langchain.chains import LLMChain
    from langchain.prompts import PromptTemplate

    # Use st.number_input for user input with a default value as float
    col1, col2 = st.columns([1, 1])  # Adjust the list to control column widths
    
    with col1:
        # Display a constant, non-editable text input field
        temperature = st.number_input("temperature", value=0.0)
        max_length = st.number_input("max_length", value=200)
    
    with col2:
        top_p = st.number_input("top_p", value=0.8, min_value=0.0, max_value=1.0)
        top_k = st.number_input("top_k", value=1, min_value=1, max_value=50)
    
    # Create a text-generation pipeline
    pipe = pipeline("text-generation", 
                    model=st.session_state.model_fine_tuned , 
                    tokenizer=st.session_state.tokenizer,
                    # Adjust generation parameters here
                    temperature=temperature,  # Adjust the temperature
                    top_k=top_k,              # Set top_k for controlling sampling diversity
                    top_p=top_p,              # Use nucleus sampling with top_p
                    max_length=max_length,    # Max number of tokens to generate
                   )
    
    
    # Question-Answering Chatbot
    st.header('Question-Answering Chatbot')
    
    
    # Apply custom CSS for increasing text input box width
    st.markdown(
        """
        <style>
        div[data-baseweb="input"] {
            width: 600px !important; /* Adjust the width as needed */
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    
    user_input = st.text_area("Ask me questions for customer support 👇")
    response = st.selectbox("""Does your question pertain to any of the following topics: 
        ORDER, SHIPPING, CANCELLATION, INVOICE, PAYMENT, REFUND, FEEDBACK, 
        CONTACT, ACCOUNT, DELIVERY, or SUBSCRIPTION?""",options=["No", "Yes"], index=0)
    if response == "Yes":
        pass
    else:
        st.error("Action canceled. Please submit only questions \
            related to the topics listed above.")
    
    if user_input and response =="Yes" and st.button('Generate Response'):
        from langchain.chains import ConversationChain
        from langchain.memory import ConversationBufferWindowMemory

        # Wrap it for LangChain
        llm_chat = HuggingFacePipeline(pipeline=pipe)
        
        # Define a prompt template
        prompt = PromptTemplate(
            input_variables=["question"],
            template="Question: {question}\nAnswer:"
        )

        # Create the chain
        chain = prompt | llm_chat
        
        # initialize memory
        #memory = ConversationBufferWindowMemory(k=3)
        #conversation = ConversationChain(
        #    llm=chain,
        #    memory=memory,
        #    verbose=True # see what is going on in background
        #)


        with st.spinner("Please wait, processing..."):
            if 'produced_doc' not in st.session_state:
                st.session_state['produced_doc'] = []
    
            if 'old_doc' not in st.session_state:
                st.session_state['old_doc'] = []
    
            #llm_response = conversation.invoke({"input": user_input})
            response = chain.invoke({"question": user_input})
            answer = response.split('Answer: ')[1]
            #latest_response = llm_response.get("response")
            #Answer = latest_response.split('Answer: ')[1]
            st.session_state.old_doc.append(user_input)
            st.session_state.produced_doc.append(answer)
    
            # Button to clear history
            if st.button("Clear History", key="Run"):
                #conversation.memory.clear()
                st.session_state['old_doc'] = []  # Clear user questions
                st.session_state['produced_doc'] = []  # Clear model responses
    
            if st.session_state['produced_doc']:
                for i in range(len(st.session_state['produced_doc'])):
                    message(st.session_state['old_doc'][i],
                            is_user=True, key=str(i)+ '_old_user')
                    message(st.session_state['produced_doc'][i],
                            key=str(i)+ '_prod_user')
    
