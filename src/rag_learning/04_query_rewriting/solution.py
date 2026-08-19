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
query = "Can I take my setup money?"

# 5. Rewrite the query for retrieval
rewrite_prompt = ChatPromptTemplate.from_template("""
You are rewriting a user's query for retrieval from an internal
company policy knowledge base.

Rewrite the query using concise terms that are likely to appear
in company policy documents.

Do not introduce new concepts or make assumptions that are not
present in the user's question.

Do not answer the question.
Return only the rewritten search query.

User query:
{query}
""")

rewritten_query = (
    rewrite_prompt
    | llm
).invoke({"query": query}).content.strip()


print("\nOriginal Query:")
print(query)

print("\nRewritten Query:")
print(rewritten_query)


# 6. Retrieve using original query
original_results = vectorstore.similarity_search(
    query,
    k=3,
)


# 7. Retrieve using rewritten query
rewritten_results = vectorstore.similarity_search(
    rewritten_query,
    k=3,
)


# 8. Compare results
print("\n\nResults using ORIGINAL query:")

for i, doc in enumerate(original_results, 1):
    print(f"\n--- Document {i} ---")
    print(doc.page_content)


print("\n\nResults using REWRITTEN query:")

for i, doc in enumerate(rewritten_results, 1):
    print(f"\n--- Document {i} ---")
    print(doc.page_content)