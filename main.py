import json
import streamlit as st
from pinecone import Pinecone, ServerlessSpec
from langchain_openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from streamlit_lottie import st_lottie
from sentence_transformers import CrossEncoder
import numpy as np


st.set_page_config(page_title="Legal Article Generator", page_icon="💬")

openai_api_key = st.secrets.openai_api_key
gemini_api_key = st.secrets.GOOGLE_API_KEY
pinecone_api_key = st.secrets.PINECONE_API_KEY

embedding_model = OpenAIEmbeddings(api_key=openai_api_key, model="text-embedding-3-small")
openai_chat = ChatOpenAI(model_name='gpt-4o-mini', temperature=0.5, api_key=openai_api_key)
gemini_chat = ChatGoogleGenerativeAI(model="gemini-1.5-pro", api_key=gemini_api_key, temperature=0.5)

pc = Pinecone(api_key=st.secrets.PINECONE_API_KEY)
index = pc.Index("legalarticlegenerator")

cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')


def augment_multiple_query(query):
    messages = [
        {
            "role": "system",
            "content": "You are a helpful expert law research assistant. Your users are asking questions about various legal cases. "
            "Suggest up to five additional related questions to help them find the information they need, for the provided question. "
            "Suggest only short questions without compound sentences. Suggest a variety of questions that cover different aspects of the topic."
            "Make sure they are complete questions, and that they are related to the original question."
            "Output one question per line. Do not number the questions."
        },
        {"role": "user", "content": query}
    ]

    response = openai_chat.invoke(messages)
    content = response.content
    content = [c.strip() for c in content.split("\n")]
    return content


def rag_search(query, namespace, top_k):
    xq = embedding_model.embed_query(query)

    # get relevant contexts (including the questions)
    res = index.query(vector=xq, top_k=top_k, include_metadata=True, namespace=namespace)

    return res


def advanced_rag_search(query, return_documents=10):
    
    augmented_queries = [query] + augment_multiple_query(query)
    
    print("augmented queries: ", augmented_queries)
    unique_documents = set()
    for q in augmented_queries:
        res = rag_search(q, namespace="FamilyLaw", top_k=20)
        for r in res['matches']:
            result_tuple = (r.metadata['chunk_text'], r.metadata['file_name'], r.metadata['page_number'])
            unique_documents.add(result_tuple)

    unique_documents = list(unique_documents)

    pairs = [[query, doc[0]] for doc in unique_documents]

    scores = cross_encoder.predict(pairs)

    sorted_unique_documents = [unique_documents[o] for o in np.argsort(scores)[::-1]]

    return sorted_unique_documents[:return_documents]


def openai_article_generator(question, context, article_placeholder):
    print("using Openai model on question:", question)
    messages=[
            {
                "role": "system",
                "content": "You are an article writer for a legal blog.",
            },
            {
                "role": "user",
                "content": f"""
    Use the following context to write a short first paragraph that answer for the user's query. If you cannot answer using the context then, please respond with 'I don't know'.
                
    User's Query:
    {question}

    Context:
    {context}
    """,
            }
        ]

    intro_para = ""
    for chunk in openai_chat.stream(messages):
        intro_para += chunk.content
        article_placeholder.write(intro_para, unsafe_allow_html=True)
    
    if intro_para.find("I don't know") == 0:
        return intro_para  
      
    messages.extend([
        {
            "role": "assistant",
            "content": intro_para,
        },
        {
            "role": "user",
            "content": "Now generate the remaining article, of at least 1000 words, starting from second paragraph based on the context and the user's query.",
        }
    ]
    )

    full_article = intro_para + "\n\n"

    for chunk in openai_chat.stream(messages):
        full_article += chunk.content
        article_placeholder.write(full_article, unsafe_allow_html=True)

    return full_article


def gemini_article_generator(question, context, article_placeholder):
    print("using Gemini model on question:", question)
    messages = [
        ("system", "You are an article writer for a legal blog."),
        ("user", f"""
    Use the following context to write a short first paragraph that answer for the user's query. If you cannot answer using the context then, please respond with 'I don't know'.
                
    User's Query:
    {question}

    Context:
    {context}
    """)
    ]

    intro_para = ""

    for chunk in gemini_chat.stream(messages):
        intro_para += chunk.content
        article_placeholder.write(intro_para, unsafe_allow_html=True)


    if intro_para.find("I don't know") == 0:
        return intro_para

    messages.extend([
        ("assistant", intro_para),
        ("user", "Now generate the remaining article, of at least 1000 words, starting from second paragraph based on the context and the user's query.")
    ])

    full_article = intro_para + "\n\n"

    for chunk in gemini_chat.stream(messages):
        full_article += chunk.content
        article_placeholder.write(full_article, unsafe_allow_html=True)
        
    return full_article


def process_text(input_text, model_name="GPT 4o mini", article_placeholder=None):
    # context = rag_search(input_text, "FamilyLaw", 10)

    context = advanced_rag_search(input_text, return_documents=10)

    context_str = "\n\n".join([c[0] for c in context])

    # context_list = [c["metadata"]["chunk_text"] for c in context['matches']]
    # context_str = "\n\n".join(context_list)
    
    if article_placeholder:
        if model_name == "GPT 4o mini":
            article = openai_article_generator(input_text, context_str, article_placeholder)
        elif model_name == "Gemini 1.5 Pro":
            article = gemini_article_generator(input_text, context_str, article_placeholder)

        if article.find("I don't know") == 0:
            context = []

    return context


def main():

    if 'model_name' not in st.session_state:
        st.session_state.model_name = 'GPT 4o mini'
    if 'law_type' not in st.session_state:
        st.session_state.law_type = 'Family Law'
    if 'user_input' not in st.session_state:
        st.session_state.user_input = ''

    with open("anima.json") as source:
        animation = json.load(source)

    col1, col2, col3 = st.columns(3)

    with col2:
        st_lottie(animation, height=100)

    st.markdown("<h2 style='text-align: center;'>Legal Article Generator</h2>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    model_name = col1.selectbox("Select LLM:", ['GPT 4o mini', 'Gemini 1.5 Pro'], key='model_name', index=['GPT 4o mini', 'Gemini 1.5 Pro'].index(st.session_state.model_name))
    law_type = col3.selectbox("Select type of law:", ['Family Law', 'Property Law', 'Civil Law', 'Corporate Law'], key='law_type', index=['Family Law', 'Property Law', 'Civil Law', 'Corporate Law'].index(st.session_state.law_type))

    # Create an input field
    user_input = st.text_area("Enter your question here:", key='user_input', value=st.session_state.user_input)

    col1, col2, col3 = st.columns(3)

    generate_article = col1.button("Generate Article", use_container_width=True)
    research = col3.button("Research", use_container_width=True)

    if generate_article or research:

        # Check if the user has entered something
        if user_input:

            if len(user_input) >  1000:
                st.warning("Please enter a question with less than 1000 characters.")
                return

            if generate_article:
                article_placeholder = st.empty()

                # Call the function with the user's input
                contexts = process_text(st.session_state.user_input, st.session_state.model_name, article_placeholder)

                # Display the result
                # st.write(result)
                st.write("---")
            
            else:
                contexts = process_text(st.session_state.user_input, st.session_state.model_name)

            # Display
            for i, context in enumerate(contexts, 1):
                st.subheader(f"Reference {i}")
                st.write(f"File Name: {context[1]}")
                st.write(f"Page Number: {int(context[2])}")
                st.write("Text:")
                st.text(context[0])
                st.write("---")
        else:
            st.warning("Please enter a question before generating the article.")


if __name__ == "__main__":
    main()