from pathlib import Path

from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter

from rag_learning.common.llm import llm
from rag_learning.config import GOOGLE_API_KEY
from rag_learning.common.embedding import embeddings

DATA_PATH = Path("data/raw/company_policy.txt")

#load documents
loader = TextLoader(DATA_PATH)
document_loaders = loader.load()

#split documents into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=50

)

chunk = text_splitter.split_documents(document_loaders)
print(f"Number of chunks: {len(chunk)}")

#create vector DB
vectorstore = Chroma.from_documents(
    documents=chunk,
    embedding=embeddings,
)


#Retrieval
query = "What is the lodging allowance when I am overseas for my job?"
results = vectorstore.similarity_search(query, k=3)

print("\nRetrieved documents:\n")

for i, doc in enumerate(results, 1):
    print(f"--- Document {i} ---")
    print(doc.page_content)


context = "\n\n".join(
    doc.page_content
    for doc in results
)

prompt = ChatPromptTemplate.from_template(
    """
    Answer the question using only the provided context.

    Context:
    {context}

    Question:
    {question}
    """
)

chain = prompt | llm

response = chain.invoke({
    "context": context,
    "question": query,
})

print("\nAnswer:")
print(response.content)