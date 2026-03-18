from dotenv import load_dotenv
from langchain_text_splitters import CharacterTextSplitter
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()


with open("cat-facts.txt") as f:
    facts = f.read()


text_splitter = CharacterTextSplitter(
    chunk_size = 1000,
    chunk_overlap = 200,
    length_function = len
)
# Create Documents (Chunks) From File
texts = text_splitter.create_documents([facts])

# Get Embeddings Model
embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")

# Initialize ChromaDB as Vector Store
vector_store = Chroma(
    collection_name="test_collection",
    embedding_function=embeddings
)

# Save Document Chunks to Vector Store
ids = vector_store.add_documents(texts)


results = vector_store.similarity_search(
    'How many people are bitten by cats in the U.S. annually?',
    k=2
)

# Print Resulting Chunks
for res in results:
    print(f"* {res.page_content} [{res.metadata}]\n\n")
