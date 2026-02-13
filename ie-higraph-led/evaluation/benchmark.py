import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from evaluation.metrics import compute_metrics
from models.ie_higraph_led import IEHiGraphLED
from utils.data_loader import load_test_data

BASELINE_MODELS = [
    "facebook/bart-large-cnn",
    "google/pegasus-cnn_dailymail",
    "t5-large",
    "allenai/led-base-16384",
    "google/flan-t5-large",
    "mistralai/Mixtral-8x7B-Instruct",
    "facebook/bart-large",
    "t5-base",
    "google/mt5-large",
    "google/long-t5-tglobal-base"
]

def generate_summary(model, tokenizer, text):
    inputs = tokenizer(text, return_tensors="pt",
                       truncation=True,
                       max_length=4096)

    summary_ids = model.generate(
        **inputs,
        max_length=512,
        num_beams=4
    )

    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)


def evaluate_model(model_name, texts, references):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    predictions = []
    for text in texts:
        pred = generate_summary(model, tokenizer, text)
        predictions.append(pred)

    return compute_metrics(predictions, references)


def main():
    texts, references = load_test_data()

    results = {}

    # Baselines
    for model_name in BASELINE_MODELS:
        print(f"Evaluating {model_name}")
        scores = evaluate_model(model_name, texts, references)
        results[model_name] = scores

    # Proposed model
    print("Evaluating IE-HiGraph-LED")
    tokenizer = AutoTokenizer.from_pretrained("allenai/led-base-16384")
    model = IEHiGraphLED()

    predictions = []
    for text in texts:
        inputs = tokenizer(text, return_tensors="pt",
                           truncation=True,
                           max_length=4096)
        summary_ids = model.led.generate(
            **inputs,
            max_length=512,
            num_beams=4
        )
        predictions.append(
            tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        )

    results["IE-HiGraph-LED"] = compute_metrics(predictions, references)

    print("\nFinal Results:\n")
    for model, score in results.items():
        print(model, score)


if __name__ == "__main__":
    main()
