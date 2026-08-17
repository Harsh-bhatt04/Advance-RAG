from pathlib import Path

from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

from rag_learning.common.embeddings import embeddings


DATA_PATH = Path("data/raw/company_policy.txt")


# 1. Load documents
loader = TextLoader(DATA_PATH)
documents = loader.load()

# 2. Split documents
splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=50,
)

chunks = splitter.split_documents(documents)

print(f"Number of chunks: {len(chunks)}")


# 3. Create vector store
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
)


# 4. User query
query = "What is the lodging allowance when I am overseas for my job?"


# 5. Retrieve documents using vector similarity
results = vectorstore.similarity_search(
    query,
    k=5,
)


# 6. Display retrieved documents
print("\nRetrieved documents:")

for i, doc in enumerate(results, 1):
    print(f"\n--- Document {i} ---")
    print(doc.page_content)