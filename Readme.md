# Legal Article Generator

This Streamlit application generates legal articles based on user queries using advanced language models and a vector database.

## Features

- Utilizes OpenAI's GPT-4 and Google's Gemini 1.5 Pro for article generation
- Retrieves relevant context from a Pinecone vector database
- Supports multiple areas of law (Family, Property, Civil, Corporate)
- Interactive web interface built with Streamlit

## Setup

1. Install required dependencies
2. Set up API keys in Streamlit secrets:
   - OpenAI API key
   - Google API key (for Gemini)
   - Pinecone API key

## Usage

1. Select the desired language model and law type
2. Enter your legal question in the text area
3. Click "Generate Article" to create a comprehensive legal article
4. View the generated article and supporting references

## Note

Ensure you have the necessary API access and a properly configured Pinecone index before running the application.
