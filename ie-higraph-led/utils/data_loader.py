import json

def load_test_data():
    with open("data/sample_dataset.json") as f:
        data = json.load(f)

    texts = [item["document"] for item in data]
    references = [item["summary"] for item in data]

    return texts, references


def load_dataset():
    # Placeholder for HuggingFace dataset loading
    return {}
