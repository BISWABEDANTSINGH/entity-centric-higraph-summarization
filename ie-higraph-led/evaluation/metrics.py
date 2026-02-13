import evaluate
from bert_score import score as bert_score

rouge = evaluate.load("rouge")

def compute_metrics(predictions, references):
    rouge_scores = rouge.compute(
        predictions=predictions,
        references=references,
        use_stemmer=True
    )

    P, R, F1 = bert_score(
        predictions,
        references,
        lang="en"
    )

    return {
        "rouge1": rouge_scores["rouge1"],
        "rouge2": rouge_scores["rouge2"],
        "rouge3": rouge_scores.get("rouge3", 0.0),
        "rougeL": rouge_scores["rougeL"],
        "bertscore": F1.mean().item()
    }
