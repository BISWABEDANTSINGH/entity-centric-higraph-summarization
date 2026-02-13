import spacy
from collections import defaultdict
import re

# Load spaCy English model
# Make sure to run: python -m spacy download en_core_web_sm
nlp = spacy.load("en_core_web_sm")


def normalize_entity(text: str) -> str:
    """
    Normalize entity text to improve cross-document matching.
    """
    text = text.strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


def extract_entities(document: str):
    """
    Extract named entities from a document.
    
    Returns:
        entities: list of dict {
            "text": entity text,
            "label": entity type,
            "start_char": start position,
            "end_char": end position
        }
    """
    doc = nlp(document)

    entities = []
    for ent in doc.ents:
        entities.append({
            "text": ent.text,
            "normalized": normalize_entity(ent.text),
            "label": ent.label_,
            "start_char": ent.start_char,
            "end_char": ent.end_char
        })

    return entities


def group_entities_across_documents(documents):
    """
    Group identical normalized entities across multiple documents.

    Args:
        documents: list of document strings

    Returns:
        entity_map: dict
            normalized_entity -> {
                "mentions": list of occurrences,
                "count": total frequency
            }
    """
    entity_map = defaultdict(lambda: {"mentions": [], "count": 0})

    for doc_id, document in enumerate(documents):
        ents = extract_entities(document)

        for ent in ents:
            key = ent["normalized"]
            entity_map[key]["mentions"].append({
                "doc_id": doc_id,
                "text": ent["text"],
                "label": ent["label"],
                "start_char": ent["start_char"],
                "end_char": ent["end_char"]
            })
            entity_map[key]["count"] += 1

    return entity_map


def compute_entity_centrality(entity_map):
    """
    Compute simple centrality score based on cross-document frequency.

    Returns:
        centrality_scores: dict
            normalized_entity -> score
    """
    centrality_scores = {}

    max_freq = max(v["count"] for v in entity_map.values()) if entity_map else 1

    for entity, data in entity_map.items():
        centrality_scores[entity] = data["count"] / max_freq

    return centrality_scores
