# entity-centric-higraph-summarization
Entity-Centric Graph-Augmented Hierarchical Transformer for Faithful Long-Context Multi-Document Abstractive Summarization. Official implementation of IE-HiGraph-LED.
# IE-HiGraph-LED
IE-HiGraph-LED

Entity-Centric Graph-Augmented Hierarchical Transformer for Faithful Long-Context Multi-Document Summarization

IE-HiGraph-LED is a structured entity-aware hierarchical graph transformer designed for faithful long-context multi-document abstractive summarization.

Unlike conventional concatenation-based models, IE-HiGraph-LED explicitly models cross-document entity interactions through graph attention and hierarchical encoding, improving factual consistency, discourse coherence, and interpretability.

🔍 Key Contributions

Entity-Centric Graph Modeling
Constructs cross-document entity graphs using NER and graph attention to model semantic interactions.

Hierarchical Long-Context Encoding
Leverages LED-based hierarchical encoding for efficient 4K–16K token processing.

Structured Interpretability
Exposes entity centrality scores, attention distributions, and discourse transition flows.

Faithfulness-Oriented Optimization
Enhances factual consistency through entity-aware representation fusion.

📊 Benchmark Results

The model is evaluated against 10 strong baselines including:

BART

PEGASUS

T5

LED

Long-T5

Mixtral

FLAN-T5

mT5

Evaluation metrics:

ROUGE-1

ROUGE-2

ROUGE-3

ROUGE-L

BERTScore

IE-HiGraph-LED demonstrates improved entity-level attention allocation and reduced discourse perplexity compared to long-context baselines.
