import json
import streamlit as st
from pinecone import Pinecone, ServerlessSpec
from langchain_openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from streamlit_lottie import st_lottie


st.set_page_config(page_title="Legal Article Generator", page_icon="💬")

openai_api_key = st.secrets.openai_api_key
gemini_api_key = st.secrets.GOOGLE_API_KEY
pinecone_api_key = st.secrets.PINECONE_API_KEY


embedding_model = OpenAIEmbeddings(api_key=openai_api_key, model="text-embedding-3-small")
openai_chat = ChatOpenAI(model_name='gpt-4o-mini', temperature=0.5, api_key=openai_api_key)
gemini_chat = ChatGoogleGenerativeAI(model="gemini-1.5-pro", api_key=gemini_api_key, temperature=0.5)

pc = Pinecone(api_key=st.secrets.PINECONE_API_KEY)
index = pc.Index("legalarticlegenerator")


def rag_search(query, namespace, top_k):
    xq = embedding_model.embed_query(query)

    # get relevant contexts (including the questions)
    res = index.query(vector=xq, top_k=top_k, include_metadata=True, namespace=namespace)

    return res


def openai_article_generator(question, context):
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

    res = openai_chat.invoke(messages)

    intro_para = res.content
    
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

    res = openai_chat.invoke(messages)

    return intro_para + "\n\n" + res.content


def gemini_article_generator(question, context):
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

    res = gemini_chat.invoke(messages)

    intro_para = res.content

    if intro_para.find("I don't know") == 0:
        return intro_para

    messages.extend([
        ("assistant", intro_para),
        ("user", "Now generate the remaining article, of at least 1000 words, starting from second paragraph based on the context and the user's query.")
    ])

    res = gemini_chat.invoke(messages)

    return intro_para + "\n\n" + res.content


def process_text(input_text, model_name="GPT 4o mini"):
    context = rag_search(input_text, "FamilyLaw", 10)

    context_list = [c["metadata"]["chunk_text"] for c in context['matches']]
    context_str = "\n\n".join(context_list)
    
    if model_name == "GPT 4o mini":
        article = openai_article_generator(input_text, context_str)
    elif model_name == "Gemini 1.5 Pro":
        article = gemini_article_generator(input_text, context_str)

    if article.find("I don't know") == 0:
        article = "Sorry, I couldn't find a relevant data to generate an article for your query."
        context['matches'] = []
    return article, context['matches']


def main():

    if 'model_name' not in st.session_state:
        st.session_state.model_name = 'GPT 4o mini'
    if 'law_type' not in st.session_state:
        st.session_state.law_type = 'Family Law'
    if 'user_input' not in st.session_state:
        st.session_state.user_input = ''

    print(f"Model Name: {st.session_state.model_name} | Law Type: {st.session_state.law_type} | User Input: {st.session_state.user_input}")

    with open("anima.json") as source:
        animation = json.load(source)


    col1, col2, col3 = st.columns(3)

    with col2:
        st_lottie(animation, height=100)

    st.markdown("<h2 style='text-align: center;'>Legal Article Generator</h2>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    model_name = col1.selectbox("Select LLM:", ['GPT 4o mini', 'Gemini 1.5 Pro'], key='model_name')
    law_type = col3.selectbox("Select type of law:", ['Family Law', 'Property Law', 'Civil Law', 'Corporate Law'], key='law_type')

    # Create an input field
    user_input = st.text_area("Enter your question:", key='user_input')



    if st.button("Generate Article"):

        # Check if the user has entered something
        if user_input:

            if len(user_input) >  1000:
                st.warning("Please enter a question with less than 1000 characters.")
                return

            # Call the function with the user's input
            result, contexts = process_text(st.session_state.user_input, st.session_state.model_name)

            # Display the result
            st.write(result)
            st.write("---")

            # Display
            for i, context in enumerate(contexts, 1):
                st.subheader(f"Reference {i}")
                st.write(f"File Name: {context['metadata'].get('file_name', 'N/A')}")
                st.write(f"Page Number: {int(context['metadata'].get('page_number', 'N/A'))}")
                st.write("Text:")
                st.text(context['metadata'].get('chunk_text', 'N/A'))
                st.write("---")
        else:
            st.warning("Please enter a question before generating the article.")


if __name__ == "__main__":
    main()