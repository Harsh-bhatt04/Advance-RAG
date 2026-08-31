from pathlib import Path

from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_classic.retrievers import ParentDocumentRetriever
from langchain_classic.storage import InMemoryStore
from langchain_text_splitters import RecursiveCharacterTextSplitter

from rag_learning.common.embeddings import embeddings


DATA_PATH = Path("data/raw/company_policy.txt")


# 1. Load documents
loader = TextLoader(DATA_PATH)
documents = loader.load()


# 2. Create parent chunks
parent_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
)


# 3. Create small child chunks
child_splitter = RecursiveCharacterTextSplitter(
    chunk_size=120,
    chunk_overlap=20,
)


# 4. Vector store for CHILD chunks
vectorstore = Chroma(
    collection_name="parent_child_demo",
    embedding_function=embeddings,
)


# 5. Store for PARENT chunks
store = InMemoryStore()


# 6. Parent-Child Retriever
retriever = ParentDocumentRetriever(
    vectorstore=vectorstore,
    docstore=store,
    child_splitter=child_splitter,
    parent_splitter=parent_splitter,
)


# 7. Add documents
retriever.add_documents(documents)


print(f"Number of original documents: {len(documents)}")


# 8. Query
query = "What are the requirements for getting the home-office allowance?"


# 9. Retrieve PARENT documents
results = retriever.invoke(query)


# 10. Display results
print("\nRetrieved parent documents:")

for i, doc in enumerate(results, 1):
    print(f"\n--- Parent Document {i} ---")
    print(doc.page_content)