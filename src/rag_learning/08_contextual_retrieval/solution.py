from pathlib import Path

from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter

from rag_learning.common.embeddings import embeddings
from rag_learning.common.llm import llm


DATA_PATH = Path("data/raw/company_policy.txt")

# Data load
loader = TextLoader(DATA_PATH)
documents = loader.load()


# Split document
splitter = RecursiveCharacterTextSplitter(
    chunk_size=150,
    chunk_overlap=20,
)

chunks = splitter.split_documents(documents)

print(f"Number of chunks: {len(chunks)}")


# Context setting
context_prompt = ChatPromptTemplate.from_template("""
You are given a document and one chunk from that document.

Write a short piece of context that explains what the chunk is
about and how it relates to the overall document.

Do not add facts that are not present in the document.

Document:
{document}

Chunk:
{chunk}

Return only the contextual description.
""")


contextualized_chunks = []

full_document = documents[0].page_content

for chunk in chunks[:3]:
    context = (
        context_prompt | llm
    ).invoke({
        "document": full_document,
        "chunk": chunk.page_content,
    }).content.strip()

    contextualized_content = f"""
Context:
{context}

Chunk:
{chunk.page_content}
""".strip()

    chunk.page_content = contextualized_content
    contextualized_chunks.append(chunk)

# Vector store

vectorstore = Chroma.from_documents(
    documents=contextualized_chunks,
    embedding=embeddings,
)

# user query
query = "How much money can I get for my home office?"

# Retrieval 
results = vectorstore.similarity_search(query, k=3)

print("\nQuery:")
print(query)

print("\nRetrieved documents using Contextual Retrieval:")

for i, doc in enumerate(results, 1):
    print(f"\n--- Document {i} ---")
    print(doc.page_content)