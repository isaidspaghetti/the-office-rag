import os
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv

load_dotenv()



def main():
  print("Main function called")
  # Pre requisite: run the normalize_docs.py script to normalize the documents and store in the normalized_docs directory
  # Load normalized docs
  # Chuck Files
  #  Embed and store in vector db

if __name__ == "__main__":
  main()  