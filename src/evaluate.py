import os
import sys
import asyncio

# Ensure src/ is in the python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
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

    # 3. Configure evaluator LLM and Embeddings
    print("Configuring Ragas evaluator with OpenRouter GPT-4o-mini...")
    evaluator_llm = ChatOpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
        model="openai/gpt-4o-mini",
        temperature=0.0,
    )
    evaluator_embeddings = OpenAIEmbeddings(
        openai_api_base="https://openrouter.ai/api/v1",
        openai_api_key=api_key,
        model="text-embedding-3-small",
    )

    # 4. Set up metrics manually (bypass evaluate() validation issue)
    # Import metrics and wrap LLM/embeddings for ragas
    from ragas.metrics._faithfulness import Faithfulness
    from ragas.metrics._answer_relevance import AnswerRelevancy
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.dataset_schema import SingleTurnSample
    from ragas.run_config import RunConfig

    wrapped_llm = LangchainLLMWrapper(evaluator_llm)
    wrapped_embeddings = LangchainEmbeddingsWrapper(evaluator_embeddings)

    metric_faithfulness = Faithfulness(llm=wrapped_llm)
    metric_answer_relevancy = AnswerRelevancy(
        llm=wrapped_llm, embeddings=wrapped_embeddings
    )

    # Initialize metrics
    run_config = RunConfig()
    metric_faithfulness.init(run_config)
    metric_answer_relevancy.init(run_config)

    # 5. Score each sample individually
    print("Starting Ragas evaluation on Faithfulness and Answer Relevancy...")
    results = []

    for i, (q, a, ctx, gt) in enumerate(zip(questions, answers, contexts, ground_truths)):
        print(f"\n  Scoring sample {i+1}/{len(questions)}: '{q[:60]}...'")

        sample = SingleTurnSample(
            user_input=q,
            response=a,
            retrieved_contexts=ctx,
            reference=gt,
        )

        # Score each metric
        f_score = metric_faithfulness.single_turn_score(sample)
        ar_score = metric_answer_relevancy.single_turn_score(sample)

        results.append({
            "question": q,
            "faithfulness": f_score,
            "answer_relevancy": ar_score,
        })
        print(f"    faithfulness={f_score:.4f}, answer_relevancy={ar_score:.4f}")

    # 6. Print Results
    print("\n" + "=" * 60)
    print("RAGAS EVALUATION RESULTS")
    print("=" * 60)

    avg_faith = sum(r["faithfulness"] for r in results) / len(results)
    avg_ar = sum(r["answer_relevancy"] for r in results) / len(results)

    print(f"\n  Average Faithfulness:      {avg_faith:.4f}")
    print(f"  Average Answer Relevancy: {avg_ar:.4f}")

    print("\nDetailed breakdown:")
    print(f"  {'Question':<65} {'Faith':>8} {'Relev':>8}")
    print(f"  {'-'*65} {'-'*8} {'-'*8}")
    for r in results:
        q_short = r['question'][:62] + '...' if len(r['question']) > 65 else r['question']
        print(f"  {q_short:<65} {r['faithfulness']:>8.4f} {r['answer_relevancy']:>8.4f}")


if __name__ == "__main__":
    main()
