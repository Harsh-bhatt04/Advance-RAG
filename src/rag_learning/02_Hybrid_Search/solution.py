from pathlib import Path

from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_text_splitters import RecursiveCharacterTextSplitter

from rag_learning.common.embeddings import embeddings


DATA_PATH = Path("data/raw/company_policy.txt")

# Data load
loader = TextLoader(DATA_PATH)
documents = loader.load()

#split 
splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=50,
)

chunks = splitter.split_documents(documents)

print(f"Number of chunks: {len(chunks)}")

# Vector Search
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
)

#user query
query = "What is the $500 home-office allowance?"

vector_results = vectorstore.similarity_search(
    query,
    k=5,
)

# BM25 Keyword Search
bm25 = BM25Retriever.from_documents(chunks)
bm25.k = 5

keyword_results = bm25.invoke(query)


# Combine Results
combined_results = []

seen = set()

for doc in vector_results + keyword_results:
    content = doc.page_content

    if content not in seen:
        seen.add(content)
        combined_results.append(doc)


print("\nQuery:")
print(query)

print("\nVector Search Results:")

for i, doc in enumerate(vector_results, 1):
    print(f"\n--- Document {i} ---")
    print(doc.page_content)


print("\nBM25 Keyword Search Results:")

for i, doc in enumerate(keyword_results, 1):
    print(f"\n--- Document {i} ---")
    print(doc.page_content)


print("\nCombined Hybrid Search Results:")

for i, doc in enumerate(combined_results[:5], 1):
    print(f"\n--- Document {i} ---")
    print(doc.page_content)