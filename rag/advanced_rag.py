from dotenv import load_dotenv
from langchain_text_splitters import CharacterTextSplitter
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import GoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_community.document_compressors import FlashrankRerank


load_dotenv("../.env")


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

# Set Chroma Vector Store as the Retriever
retriever = vector_store.as_retriever()

llm = GoogleGenerativeAI(model="gemini-3-flash-preview")


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

####################################################################
# We want to modify user's query before send it to vector store.
# it is called Pre-retrieval Query Rewriting
#
#
def rewrite_query(query: str, llm: GoogleGenerativeAI):
    rewrite_prompt = f"""
      You are a helpful assistant that takes a user's query and
      turns it into a short statement or paragraph so that it can
      be used in a semantic similarity search on a vector database
      to return the most similar chunks of content based on the
      rewritten query. Please make no comments, just return the
      rewritten query.
      
      query: {query}

      ai: """

    retrieval_query = llm.invoke(rewrite_prompt)

    return retrieval_query


# Flash Rerank Compressor for Post-retrieval Rerank
compressor = FlashrankRerank()

# Update Retriever -> Compression Retriever
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=retriever
)


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


query = "Does cat have more bones than average human?"
# query = "Is 10 more than 8?"

# Advanced RAG: Pre-retrieval Query Rewrite
retrival_query = rewrite_query(query, llm)

# Retrieval w/ Post-retrieval Reranking
docs = retriever.invoke(retrival_query)

# Format Docs for Context String
context = format_docs(docs)

# Prompt Template
final_prompt = custom_rag_prompt.format(context=context, query=query)

result = llm.invoke(final_prompt)

print(result)
