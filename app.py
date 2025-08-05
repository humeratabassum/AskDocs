import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# Updated imports (LangChain v0.2+)
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_openai import ChatOpenAI

from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.question_answering import load_qa_chain

# Load PDF file
loader = PyPDFLoader("docs/sample.pdf")
 # Change filename here
documents = loader.load()

# Split documents into chunks
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
docs = text_splitter.split_documents(documents)

# Embed and index
embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
db = FAISS.from_documents(docs, embeddings)

# QA chain
llm = ChatOpenAI(temperature=0, openai_api_key=openai_api_key)
chain = load_qa_chain(llm, chain_type="stuff")

# Interactive CLI
while True:
    query = input("\nAsk a question (or 'exit'): ")
    if query.lower() in ['exit', 'quit']:
        break

    matching_docs = db.similarity_search(query)
    result = chain.run(input_documents=matching_docs, question=query)
    print(f"\nAnswer:\n{result}")
