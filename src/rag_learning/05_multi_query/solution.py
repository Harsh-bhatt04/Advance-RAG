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
    chunk_size = 300,
    chunk_overlap = 50,
)

chunks = splitter.split_documents(documents)
print(f"Number of Chunks: {len(chunks)}")

# create vector store
vectorstore = Chroma.from_documents(
    documents = chunks,
    embedding = embeddings
)

# Query
query = "Can I take my setup money?"


# 5. Generate multiple search queries
multi_query_prompt = ChatPromptTemplate.from_template("""
You are generating search queries for an internal company policy
knowledge base.

The knowledge base contains policies about:
- business travel and accommodation
- remote work
- home-office equipment allowances
- employee leave
- business expenses
- travel approval

Generate exactly 3 different search queries for the user's question.

Use terms likely to appear in these company policies.
Do not invent unrelated concepts.
Explore different possible interpretations of ambiguous words.

Return exactly one query per line.
Do not use numbering, bullets, or explanations.

User question:
{query}
""")

response = (
    multi_query_prompt
    | llm
).invoke({"query": query})

generated_queries = [
    line.strip().lstrip("- ").strip()
    for line in response.content.strip().splitlines()
    if line.strip()
]

print("\nOriginal Query:")
print(query)

print("\nGenerated Queries:")

for generated_query in generated_queries:
    print(f"- {generated_query}")


# 6. Retrieve documents for every generated query
all_results = []

for generated_query in generated_queries:
    results = vectorstore.similarity_search(
        generated_query,
        k=3,
    )

    all_results.extend(results)


# 7. Remove duplicate documents
unique_results = {}

for doc in all_results:
    unique_results[doc.page_content] = doc


# 8. Display combined results
print("\nCombined Unique Retrieved Documents:")

for i, doc in enumerate(unique_results.values(), 1):
    print(f"\n--- Document {i} ---")
    print(doc.page_content)