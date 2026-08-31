from pathlib import Path

from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

from rag_learning.common.embeddings import embeddings

DATA_PATH = Path("data/raw/company_policy.txt")

# Data load
loader = TextLoader(DATA_PATH)
documents = loader.load()

# Create small chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size = 120,
    chunk_overlap = 20,
)

chunks = splitter.split_documents(documents)

print(f"Number of chunks: {len(chunks)}")

# Create vector store
vectorstore = Chroma.from_documents(
    documents = chunks,
    embedding = embeddings,
)

# 4. Query
query = "What are the requirements for getting the home-office allowance?"


# Retrieval
results = vectorstore.similarity_search(
    query,
    k = 5,
)


# 6. Display retrieved chunks
print("\nRetrieved chunks:")

for i, doc in enumerate(results, 1):
    print(f"\n--- Chunk {i} ---")
    print(doc.page_content)