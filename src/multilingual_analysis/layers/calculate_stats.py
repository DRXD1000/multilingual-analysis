"""Calculating Token language"""
from fast_langdetect import detect
from loguru import logger

def layerwise_lang_stats(hidden_embed_token_level, candidate_langs=None):
    if candidate_langs is None:
        candidate_langs = ["en", "zh", "es", "ru", "de", "fr"]
    lang_stats = {}
    for layer in hidden_embed_token_level:
        lang_stats[layer] = {"total_count": 0}
        for token in hidden_embed_token_level[layer]:
            try:
                lang_pred = detect(token,low_memory=False)
            except ValueError as e:
                continue
            except Exception as e:
                logger.error(f"An error occurred: {e}")
                lang_pred = None
            if lang_pred and lang_pred["score"] > 0.2 and (lang_pred["lang"] in candidate_langs):
                lang_stats[layer]["total_count"] += 1
                if lang_pred["lang"] in lang_stats[layer]:
                    lang_stats[layer][lang_pred["lang"]] += 1
                else:
                    lang_stats[layer][lang_pred["lang"]] = 1
    return lang_stats
