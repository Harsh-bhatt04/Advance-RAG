from pathlib import Path

from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

from rag_learning.common.embeddings import embeddings

DATA_PATH = Path("data/raw/company_policy.txt")

# Data load
loader = TextLoader(DATA_PATH)
documents = loader.load()

# split the document
splitter = RecursiveCharacterTextSplitter(
    chunk_size = 300,
    chunk_overlap = 50,

)

chunks = splitter.split_documents(documents)
print(f"Number of chunks: {len(chunks)}")

# vector store
vectorstore = Chroma.from_documents(
    documents = chunks,
    embedding = embeddings
)

# query 
query = "What is the $500 home-office allowance?"


# Retrieval
results = vectorstore.similarity_search(
    query,
    k=3
)

print("\nQuery:")
print(query)

print("\nRetrieved documents using Vector Search:")

for i, doc in enumerate(results, 1):
    print(f"\n--- Document {i} ---")
    print(doc.page_content)