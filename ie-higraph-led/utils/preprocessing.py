import torch
from transformers import AutoTokenizer
from utils.entity_extraction import group_entities_across_documents

MAX_INPUT_LENGTH = 4096
MODEL_NAME = "allenai/led-base-16384"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


def concatenate_documents(documents):
    """
    Concatenate multi-document cluster with separators.
    """
    return " </s> ".join(documents)


def hierarchical_segment(documents):
    """
    Segment documents into paragraph-level units
    for hierarchical encoding.
    """
    segments = []

    for doc in documents:
        paragraphs = doc.split("\n\n")
        segments.extend(paragraphs)

    return segments


def build_entity_mask(input_text, entity_map):
    """
    Create token-level mask marking entity positions.
    
    Returns:
        entity_mask: torch tensor [seq_len]
    """
    encoding = tokenizer(
        input_text,
        return_offsets_mapping=True,
        truncation=True,
        max_length=MAX_INPUT_LENGTH
    )

    offsets = encoding["offset_mapping"]
    entity_mask = torch.zeros(len(offsets))

    for entity, data in entity_map.items():
        for mention in data["mentions"]:
            start_char = mention["start_char"]
            end_char = mention["end_char"]

            for idx, (start, end) in enumerate(offsets):
                if start >= start_char and end <= end_char:
                    entity_mask[idx] = 1.0

    return entity_mask


def preprocess_cluster(documents):
    """
    Full preprocessing pipeline for one document cluster.

    Returns:
        {
            input_ids,
            attention_mask,
            entity_mask,
            entity_map
        }
    """

    # Step 1: Concatenate
    concatenated = concatenate_documents(documents)

    # Step 2: Extract entity structure
    entity_map = group_entities_across_documents(documents)

    # Step 3: Tokenize
    encoding = tokenizer(
        concatenated,
        truncation=True,
        padding="max_length",
        max_length=MAX_INPUT_LENGTH,
        return_tensors="pt"
    )

    # Step 4: Entity mask
    entity_mask = build_entity_mask(concatenated, entity_map)

    return {
        "input_ids": encoding["input_ids"],
        "attention_mask": encoding["attention_mask"],
        "entity_mask": entity_mask,
        "entity_map": entity_map
    }
