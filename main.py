from dotenv import load_dotenv
from langchain_text_splitters import CharacterTextSplitter
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import GoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough


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


# results = vector_store.similarity_search(
#     'How many people are bitten by cats in the U.S. annually?',
#     k=2
# )

# # Print Resulting Chunks
# for res in results:
#     print(f"* {res.page_content} [{res.metadata}]\n\n")


# Set Chroma Vector Store as the Retriever
retriever = vector_store.as_retriever()

llm = GoogleGenerativeAI(model="gemini-3-flash-preview")

# Create Document Parsing Function to String
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


# Create the Prompt Template
prompt_template = """Use the context provided to answer 
the user's question below. If you do not know the answer 
based on the context provided, tell the user that you do 
not know the answer to their question based on the context
provided and that you are sorry.

context: {context}

question: {query}

answer: """


custom_rag_prompt = PromptTemplate.from_template(prompt_template)

rag_chain = (
    {"context": retriever | format_docs, "query": RunnablePassthrough()}
    | custom_rag_prompt
    | llm
    | StrOutputParser()
)

result = rag_chain.invoke(
  "Do you know how many people are bitten by cats in the U.S. every year?"
)

print(result)