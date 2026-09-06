from pathlib import Path

from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.tools import tool
from langchain.agents import create_agent

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


@tool
def search_company_policy(query: str) -> str:
    """Search the company policy documents for relevant information."""

    results = vectorstore.similarity_search(query, k=3)

    return "\n\n".join(
        f"Document {i}:\n{doc.page_content}"
        for i, doc in enumerate(results, 1)
    )


agent = create_agent(
    model=llm,
    tools=[search_company_policy],
    system_prompt="""
You are a company policy assistant.

Decide when you need to search the company policy.

If the question requires information from the company policy,
use the search_company_policy tool.

After retrieving information, answer the user using the
retrieved policy.

Do not invent policy information.
""",
)


query = "How much is the international hotel allowance?"

result = agent.invoke({
    "messages": [
        {
            "role": "user",
            "content": query,
        }
    ]
})

print("\nAnswer:")
print(result["messages"][-1].content)