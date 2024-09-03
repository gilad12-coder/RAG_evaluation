import nltk
from typing import Dict, Tuple
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
from collections import Counter
import math
from loguru import logger
from sentence_transformers import SentenceTransformer, util
import os

# Ensure necessary NLTK data is downloaded
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)


def calculate_rouge_scores(reference: str, hypothesis: str) -> Dict[str, float]:
    """Calculate ROUGE scores for a given reference and hypothesis."""
    logger.debug(f"Calculating ROUGE scores for texts of lengths {len(reference)} and {len(hypothesis)}")
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
    scores = scorer.score(reference, hypothesis)
    logger.info(
        f"ROUGE scores calculated: ROUGE-1 = {scores['rouge1'].fmeasure:.4f}, ROUGE-L = {scores['rougeL'].fmeasure:.4f}")
    return {
        'rouge1': scores['rouge1'].fmeasure,
        'rougeL': scores['rougeL'].fmeasure
    }


def calculate_bleu_score(reference: str, hypothesis: str, n: int = 4) -> float:
    """Calculate a simplified BLEU score for a given reference and hypothesis."""
    logger.debug(f"Calculating BLEU score for texts of lengths {len(reference)} and {len(hypothesis)}")

    def ngrams(text, n):
        return [' '.join(text[i:i + n]) for i in range(len(text) - n + 1)]

    ref_tokens = reference.split()
    hyp_tokens = hypothesis.split()

    if len(hyp_tokens) == 0:
        logger.warning("Hypothesis is empty, BLEU score is 0")
        return 0

    # Calculate brevity penalty
    bp = min(1, math.exp(1 - len(ref_tokens) / len(hyp_tokens)))
    logger.debug(f"Brevity penalty: {bp:.4f}")

    # Calculate n-gram precisions
    precisions = []
    for i in range(1, n + 1):
        ref_ngrams = Counter(ngrams(ref_tokens, i))
        hyp_ngrams = Counter(ngrams(hyp_tokens, i))
        if len(hyp_ngrams) == 0:
            precisions.append(0)
        else:
            matches = sum((ref_ngrams & hyp_ngrams).values())
            total = sum(hyp_ngrams.values())
            precisions.append(matches / total)
        logger.debug(f"{i}-gram precision: {precisions[-1]:.4f}")

    # Smooth very small precisions
    precisions = [max(p, 1e-10) for p in precisions]

    # Calculate geometric mean of precisions
    log_avg_precision = sum(math.log(p) for p in precisions) / len(precisions)
    bleu_score = bp * math.exp(log_avg_precision)
    logger.info(f"BLEU score calculated: {bleu_score:.4f}")
    return bleu_score


def calculate_meteor_score(reference: str, hypothesis: str) -> float:
    """Calculate METEOR score for a given reference and hypothesis."""
    logger.debug(f"Calculating METEOR score for texts of lengths {len(reference)} and {len(hypothesis)}")
    tokenized_reference = nltk.word_tokenize(reference)
    tokenized_hypothesis = nltk.word_tokenize(hypothesis)
    meteor_score_value = meteor_score([tokenized_reference], tokenized_hypothesis)
    logger.info(f"METEOR score calculated: {meteor_score_value:.4f}")
    return meteor_score_value


def calculate_sbert_similarity(reference: str, hypothesis: str) -> float:
    """Calculate semantic similarity between reference and hypothesis using SentenceBERT."""
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    logger.debug(f"Calculating SBERT similarity for texts of lengths {len(reference)} and {len(hypothesis)}")

    # Load a pre-trained SentenceBERT model
    model = SentenceTransformer('paraphrase-mpnet-base-v2')

    # Encode sentences to get their embeddings
    reference_embedding = model.encode(reference, convert_to_tensor=True)
    hypothesis_embedding = model.encode(hypothesis, convert_to_tensor=True)

    # Calculate cosine similarity
    similarity = util.pytorch_cos_sim(reference_embedding, hypothesis_embedding)
    sbert_score = similarity.item()

    logger.info(f"SBERT similarity calculated: {sbert_score:.4f}")

    return sbert_score


def calculate_textual_and_semantic_correlation_metrics(reference: str, hypothesis: str) -> Dict[str, float]:
    """Calculate all NLP metrics (BLEU, ROUGE, METEOR, SBERT) for a given reference and hypothesis."""
    logger.info("Starting calculation of textual and semantic correlation metrics")
    rouge_scores = calculate_rouge_scores(reference, hypothesis)
    bleu_score = calculate_bleu_score(reference, hypothesis)
    meteor_score = calculate_meteor_score(reference, hypothesis)
    sbert_similarity = calculate_sbert_similarity(reference, hypothesis)

    metrics = {
        "BLEU": bleu_score,
        "ROUGE-1": rouge_scores['rouge1'],
        "ROUGE-L": rouge_scores['rougeL'],
        "METEOR": meteor_score,
        "SBERT_similarity": sbert_similarity
    }

    logger.info("Textual and semantic correlation metrics calculation completed")
    logger.debug(f"Calculated metrics: {metrics}")
    return metrics
