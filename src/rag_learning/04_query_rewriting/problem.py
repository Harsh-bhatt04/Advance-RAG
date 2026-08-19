from pathlib import Path

from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

from rag_learning.common.embeddings import embeddings

DATA_PATH = Path("data/raw/company_policy.txt")

# Data load

loader = TextLoader(DATA_PATH)
document = loader.load()

# Split document
splitter = RecursiveCharacterTextSplitter(
    chunk_size = 300,
    chunk_overlap = 50,
)

chunks = splitter.split_documents(document)

print(f"Number of Chunks: {len(chunks)} ")

# Create Vector Store
vectorstore = Chroma.from_documents(
    documents = chunks,
    embedding = embeddings
)

# Vague user query
query = "What can I get if I work from home?"

# retrieval using the original query
results = vectorstore.similarity_search(
    query,
    k=5
)

# 6. Display results
print("\nOriginal Query:")
print(query)

print("\nRetrieved documents:")

for i, doc in enumerate(results, 1):
    print(f"\n--- Document {i} ---")
    print(doc.page_content)