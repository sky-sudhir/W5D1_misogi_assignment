# raga_eval.py

from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
from datasets import Dataset

def evaluate_single_sample(question: str, contexts: list[str], answer: str) -> dict:
    """
    Evaluates a single query-answer pair using RAGAS metrics.

    Args:
        question (str): User's medical query
        contexts (list[str]): Retrieved chunks
        answer (str): LLM-generated answer

    Returns:
        dict: Scores for all RAGAS metrics
    """

    data = {
        "question": [question],
        "contexts": [contexts],
        "answer": [answer],
        "ground_truth": [None],  # Optional for now
    }

    dataset = Dataset.from_dict(data)

    scores = evaluate(
        dataset,
        metrics=[
            faithfulness,
            answer_relevancy,
            # context_precision,
            # context_recall,
        ],
    )

    return {
        "faithfulness": scores["faithfulness"],
        "answer_relevancy": scores["answer_relevancy"],
        # "context_precision": scores["context_precision"].tolist()[0],
        # "context_recall": scores["context_recall"].tolist()[0],
    }
