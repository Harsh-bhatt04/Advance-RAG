from pathlib import Path

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma

from rag_learning.common.embeddings import embeddings

DATA_PATH = Path("data/raw/company_policy.txt")

# Data load
loader= TextLoader(DATA_PATH)
documents = loader.load()


# Split documents
splitter = RecursiveCharacterTextSplitter(
    chunk_size = 300,
    chunk_overlap = 50,
)

chunks = splitter.split_documents(documents)
print(f"Number of chunks: {len(chunks)}")

# vector store
vectorstore = Chroma.from_documents(
    documents = chunks,
    embedding = embeddings,
)

# user query
query = "How much money can I get from my home office?"

# retrieve documents
results = vectorstore.similarity_search(
    query,
    k=3
)

print("\nQuery:")
print(query)

print("\n Retreived documents:")

for i,doc in enumerate(results,1):
    print(f"\n---Document {i}--")
    print(doc.page_content)

