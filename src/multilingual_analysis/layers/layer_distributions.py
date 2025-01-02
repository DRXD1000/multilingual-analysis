"""Layer Distributions."""

def layerwise_lang_distribution(lang_stats, candidate_langs):
    lang_distribution = {}
    for layer in lang_stats:
        lang_distribution[layer] = {}
        for lang in candidate_langs:
            if lang in lang_stats[layer]:
                lang_distribution[layer][lang] = lang_stats[layer][lang] / lang_stats[layer]["total_count"]
            else:
                lang_distribution[layer][lang] = 0
    return lang_distribution


def layerwise_lang_distribution_bi(lang_distribution):
    lang_distribution_bi = {}
    for layer in lang_distribution:
        lang_distribution_bi[layer] = {}
        lang_distribution_bi[layer]["en"] = lang_distribution[layer]["en"]
        lang_distribution_bi[layer]["non-en"] = 1 - lang_distribution_bi[layer]["en"]
    return lang_distribution_bi


def layerwise_lang_distribution_th(lang_distribution):
    lang_distribution_bi = {}
    for layer in lang_distribution:
        lang_distribution_bi[layer] = {}
        lang_distribution_bi[layer]["en"] = lang_distribution[layer]["en"]
        lang_distribution_bi[layer]["zh"] = lang_distribution[layer]["zh"]
        lang_distribution_bi[layer]["non-en-zh"] = 1 - lang_distribution_bi[layer]["en"] - lang_distribution[layer]["zh"]
    return lang_distribution_bi


def average_layerwise_lang_distribution(lst_lang_distribution, candidate_langs=None):
    if candidate_langs is None:
        candidate_langs = ["en", "de"]
    average_lang_distribution = {}
    for lang_distribution in lst_lang_distribution:
        for layer in lang_distribution:
            if layer in average_lang_distribution:
                for lang in lang_distribution[layer]:
                    average_lang_distribution[layer][lang] += lang_distribution[layer][lang]
            else:
                average_lang_distribution[layer] = {}
                for lang in lang_distribution[layer]:
                    average_lang_distribution[layer][lang] = lang_distribution[layer][lang]
    for layer in average_lang_distribution:
        for lang in average_lang_distribution[layer]:
            average_lang_distribution[layer][lang] /= len(lst_lang_distribution)
    for lang in candidate_langs:
        if lang in ("en", "de"):
            average_lang_distribution[0][lang] = 0
        else:
            average_lang_distribution[0][lang] = 1 / (len(candidate_langs) - 2)

    return average_lang_distribution