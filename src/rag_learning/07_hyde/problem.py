from pathlib import Path

from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

from rag_learning.common.embeddings import embeddings

DATA_PATH = Path("data/raw/company_policy.txt")

#document load
loader = TextLoader(DATA_PATH)
documents = loader.load()

# Split documents
splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=50,
)

chunks = splitter.split_documents(documents)

print(f"Number of chunks: {len(chunks)}")

# Vector store
vectorstore = Chroma.from_documents(
    documents= chunks,
    embedding = embeddings,
)

# 4. User query
query = "I need some money to get my remote working setup ready. Is there anything available?"


# Normal similarity Search
results = vectorstore.similarity_search(
    query,
    k = 3,
)



# 6. Display results
print("\nOriginal Query:")
print(query)

print("\nRetrieved documents:")

for i, doc in enumerate(results, 1):
    print(f"\n--- Document {i} ---")
    print(doc.page_content)