from pathlib import Path

from langchain_community.document_loaders import TextLoader
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

from rag_learning.common.embeddings import embeddings
from rag_learning.common.llm import llm


DATA_PATH = Path("data/raw/company_policy.txt")

loader = TextLoader(DATA_PATH)
documents = loader.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=50,
)

chunks = splitter.split_documents(documents)

vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
)

query = "How much is the international hotel allowance?"

results = vectorstore.similarity_search(query, k=3)

context = "\n\n".join(doc.page_content for doc in results)

prompt = f"""
Answer the question using the context below.

Context:
{context}

Question:
{query}
"""

response = llm.invoke(prompt)

print("\nAnswer:")
print(response.content)