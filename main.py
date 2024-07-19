import streamlit as st
from pinecone import Pinecone, ServerlessSpec
from langchain_openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI

st.set_page_config(page_title="Legal Article Generator", page_icon="💬")

openai_api_key = st.secrets.openai_api_key
pinecone_api_key = st.secrets.PINECONE_API_KEY


embedding_model = OpenAIEmbeddings(api_key=openai_api_key, model="text-embedding-3-small")
openai_chat = ChatOpenAI(model_name='gpt-3.5-turbo', temperature=0.5, api_key=openai_api_key)

pc = Pinecone(api_key=st.secrets.PINECONE_API_KEY)
index = pc.Index("legalarticlegenerator")


def rag_search(query, namespace, top_k):
    xq = embedding_model.embed_query(query)

    # get relevant contexts (including the questions)
    res = index.query(vector=xq, top_k=top_k, include_metadata=True, namespace=namespace)

    return res


def openai_article_generator(question, context):
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


def process_text(input_text):
    print("processing text:", input_text)
    context = rag_search(input_text, "FamilyLaw", 10)

    context_list = [c["metadata"]["chunk_text"] for c in context['matches']]
    context_str = "\n\n".join(context_list)

    article = openai_article_generator(input_text, context_str)

    return article, context['matches']


def main():
    st.markdown("<h1 style='text-align: center'>Legal Article Generator</h1>", unsafe_allow_html=True)

    col1, col2 = st.columns([1, 2])
    selected_option = col1.selectbox("Select type of law:", ['Family Law', 'Property Law', 'Civil Law', 'Corporate Law'])

    # Create an input field
    user_input = st.text_area("Enter your question:")

    # Check if the user has entered something
    if user_input:
        # Call the function with the user's input
        result, contexts = process_text(user_input)

        # Display the result
        st.write(result)

        # if st.button("References"):
        for i, context in enumerate(contexts, 1):
            st.subheader(f"Reference {i}")
            st.write(f"File Name: {context['metadata'].get('file_name', 'N/A')}")
            st.write(f"Page Number: {int(context['metadata'].get('page_number', 'N/A'))}")
            st.write("Text:")
            st.text(context['metadata'].get('chunk_text', 'N/A'))
            st.write("---")


if __name__ == "__main__":
    main()