from pathlib import Path

from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import CrossEncoder

from rag_learning.common.embeddings import embeddings

DATA_PATH = Path("data/raw/company_policy.txt")

# load data
loader = TextLoader(DATA_PATH)
documents = loader.load()

# split documents into chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=50,
)

chunks = splitter.split_documents(documents)

print(f"Number of chunks: {len(chunks)}")

# create vector store
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
)


# User Query
query = "What is the lodging allowance when I am overseas for my job?"

# retrieve documents using vector similarity
results = vectorstore.similarity_search(
    query=query,
    k=5,
)

print("\nBefore re-ranking:")

for i, doc in enumerate(results, 1):
    print(f"\n--- Document {i} ---")
    print(doc.page_content)

# load cross-encoder re-ranker
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

# create query-document pairs
pairs = [
    (query, doc.page_content)
    for doc in results
]

# calculate relevant scores
scores = reranker.predict(pairs)

# Combine document with score
ranked_results = list(zip(results,scores))

ranked_results.sort(
    key=lambda x: x[1],
    reverse=True,
)


# 10. Display re-ranked results
print("\n\nAfter re-ranking:")

for i, (doc, score) in enumerate(ranked_results, 1):
    print(f"\n--- Document {i} | Score: {score:.4f} ---")
    print(doc.page_content)