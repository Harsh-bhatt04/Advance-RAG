from langchain_google_genai import GoogleGenerativeAIEmbeddings

from rag_learning.config import GOOGLE_API_KEY


embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    google_api_key=GOOGLE_API_KEY,
)