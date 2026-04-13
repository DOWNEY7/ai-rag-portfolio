import os
from dotenv import load_dotenv

try:
    from langfuse.langchain import CallbackHandler as LangfuseHandler
    HAS_LANGFUSE = True
except ImportError:
    HAS_LANGFUSE = False

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_cohere import CohereRerank

try:
    from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
except (ModuleNotFoundError, ImportError):
    from langchain_community.retrievers.contextual_compression import ContextualCompressionRetriever

from langchain_core.prompts import ChatPromptTemplate

try:
    from langchain.chains import create_retrieval_chain
    from langchain.chains.combine_documents import create_stuff_documents_chain
except (ModuleNotFoundError, ImportError):
    from langchain_community.chains import create_retrieval_chain
    from langchain_community.chains.combine_documents import create_stuff_documents_chain

def get_rag_chain():
    """Builds and returns the LangChain RAG pipeline."""
    # Load environment variables
    load_dotenv()
    
    # Prioritize OpenRouter API Key if available, else fallback to standard OpenAI key
    api_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY or OPENAI_API_KEY is missing from the .env file.")

    # 1. Initialize exact Embeddings we used for ingestion
    embeddings = OpenAIEmbeddings(
        openai_api_base="https://openrouter.ai/api/v1",
        openai_api_key=api_key,
        model="text-embedding-3-small"
    )

    # 2. Connect to local ChromaDB and set up Retriever (fetch top 20 chunks initially)
    if not os.path.exists("./chroma_db"):
         raise RuntimeError("Error: ./chroma_db not found. Run src/ingest.py first.")
         
    vectorstore = Chroma(
        persist_directory="./chroma_db",
        embedding_function=embeddings
    )
    base_retriever = vectorstore.as_retriever(search_kwargs={"k": 20})

    # 3. Initialize Cohere Re-ranker and Contextual Compression Retriever
    cohere_api_key = os.getenv("COHERE_API_KEY")
    if not cohere_api_key:
        raise ValueError("COHERE_API_KEY is missing from the .env file.")
         
    compressor = CohereRerank(cohere_api_key=cohere_api_key, model="rerank-english-v3.0", top_n=5)
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=base_retriever
    )

    # 4. Initialize the LLM via OpenRouter
    llm = ChatOpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
        model="openai/gpt-4o-mini",
        temperature=0.0
    )

    # 5. Create a Strict System Prompt Template
    system_prompt = (
        "You are an assistant for question-answering tasks. "+
        "Answer the user's question using ONLY the provided context. "+
        "If you don't know the answer based on the context, say you don't know. "+
        "Keep your answer concise and informative. \n\n"
        "Context: {context}"
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}")
    ])

    # 6. Connect chunks to LLM to parse answer, and tie it to the compression retriever
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(compression_retriever, question_answer_chain)
    
    return rag_chain

def run_retrieval_pipeline(query: str):
    print(f"--- Processing Query: '{query}' ---")
    print("Constructing RAG Chain with Compression Retriever...")
    rag_chain = get_rag_chain()

    # Initialize Langfuse callback handler if available
    callbacks = []
    langfuse_handler = None
    if HAS_LANGFUSE:
        try:
            langfuse_handler = LangfuseHandler()
            callbacks.append(langfuse_handler)
            print("Langfuse tracing enabled.")
        except Exception:
            print("Langfuse not configured, skipping tracing.")

    # 7. Execute the query
    print("Executing query...\n")
    config = {"callbacks": callbacks} if callbacks else {}
    response = rag_chain.invoke({"input": query}, config=config)

    # 8. Print the results
    print("=" * 60)
    print("AI ANSWER:\n")
    print(response["answer"])
    print("\n" + "=" * 60)
    
    print("\nCITATIONS (Retrieved Context Metadata):")
    for i, doc in enumerate(response["context"]):
        source_file = doc.metadata.get("source", "Unknown Source")
        print(f"  [{i+1}] Source File: {source_file}")

    # Flush Langfuse to ensure all traces are sent before exit
    if langfuse_handler is not None:
        if hasattr(langfuse_handler, 'flush'):
            langfuse_handler.flush()
        elif hasattr(langfuse_handler, 'client'):
            langfuse_handler.client.flush()
        print("\nLangfuse trace flushed successfully.")

if __name__ == "__main__":
    # Test Query
    test_query = "How do I use prompt caching in Claude?"
    run_retrieval_pipeline(test_query)