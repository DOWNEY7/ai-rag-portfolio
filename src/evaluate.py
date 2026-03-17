import os
import sys

# Ensure src/ is in the python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
from datasets import Dataset
from ragas import evaluate
from ragas.metrics.collections import faithfulness, answer_relevancy
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# Import RAG chain from our retrieve script
from src.retrieve import get_rag_chain

def main():
    load_dotenv()
    api_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("Missing API key.")

    # 1. Prepare Golden Dataset (Claude Context)
    questions = [
        "What are some best practices for tool use or function calling with Claude?",
        "How does prompt caching work in Claude and what are its benefits?",
        "How should I structure a system prompt for Claude to get the best results?"
    ]

    ground_truths = [
        "Best practices for tool use include providing clear descriptions, using strongly typed parameters, handling errors gracefully, and asking for user confirmation before taking destructive actions.",
        "Prompt caching allows Claude to remember frequently used context across api calls. Benefits include lower latency and reduced costs for repetitive tasks.",
        "A system prompt should assign a role, provide clear instructions, establish rules and constraints, and give examples of desired output behavior."
    ]

    # 2. Collect Answers and Contexts from our RAG Pipeline
    print("Initializing our live RAG pipeline...")
    rag_chain = get_rag_chain()

    answers = []
    contexts = []

    print("Running RAG pipeline to generate answers for evaluation...")
    for q in questions:
        print(f"  Querying: '{q}'")
        response = rag_chain.invoke({"input": q})
        answers.append(response["answer"])
        
        # Extract raw text from context documents
        retrieved_contexts = [doc.page_content for doc in response["context"]]
        contexts.append(retrieved_contexts)

    # 3. Format as HuggingFace Dataset
    # ragas expects a specific dictionary format
    data = {
        "user_input": questions,    # Changed from "question" in newer versions
        "response": answers,        # Changed from "answer"
        "retrieved_contexts": contexts, # Changed from "contexts" 
        "reference": ground_truths  # Changed from "ground_truth"
    }
    
    # We maintain standard fallback keys as well just in case legacy fallback applies
    data["question"] = questions
    data["answer"] = answers
    data["contexts"] = contexts
    data["ground_truth"] = ground_truths

    dataset = Dataset.from_dict(data)

    # 4. Configure Ragas with OpenRouter
    print("Configuring Ragas evaluator with OpenRouter GPT-4o-mini...")
    evaluator_llm = ChatOpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
        model="openai/gpt-4o-mini",
        temperature=0.0
    )
    evaluator_embeddings = OpenAIEmbeddings(
        openai_api_base="https://openrouter.ai/api/v1",
        openai_api_key=api_key,
        model="text-embedding-3-small"
    )

    # 5. Run Evaluation
    print("Starting Ragas evaluation on Faithfulness and Answer Relevancy...")
    result = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy],
        llm=LangchainLLMWrapper(evaluator_llm),
        embeddings=LangchainEmbeddingsWrapper(evaluator_embeddings)
    )

    # 6. Print Results
    print("\n" + "="*60)
    print("📊 RAGAS EVALUATION RESULTS")
    print("="*60)
    print(result)
    try:
        df = result.to_pandas()
        print("\nDetailed breakdown:")
        print(df[["question", "faithfulness", "answer_relevancy"]])
    except Exception as e:
        print("Note: DataFrame parsing failed, use raw output above.", e)
        
if __name__ == "__main__":
    main()
