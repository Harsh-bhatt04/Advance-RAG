from pathlib import Path

from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter

from rag_learning.common.embeddings import embeddings
from rag_learning.common.llm import llm


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
query = "I need some money to get my remote working setup ready. Is there anything available?"


# 5. Generate hypothetical document
hyde_prompt = ChatPromptTemplate.from_template("""
Write a short hypothetical passage that would answer the
user's question.

The passage should sound like a company policy document.

Do not explain that it is hypothetical.
Do not mention this prompt.
Just write the passage.

Question:
{query}
""")

hypothetical_document = (
    hyde_prompt
    | llm
).invoke({"query": query}).content.strip()


print("\nOriginal Query:")
print(query)

print("\nHypothetical Document:")
print(hypothetical_document)


# 6. Search using the hypothetical document
results = vectorstore.similarity_search(
    hypothetical_document,
    k=3,
)


# 7. Display retrieved documents
print("\nRetrieved documents using HyDE:")

for i, doc in enumerate(results, 1):
    print(f"\n--- Document {i} ---")
    print(doc.page_content)