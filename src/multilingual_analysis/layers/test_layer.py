import logging
import random
import sys

import torch
from langdetect import detect_langs
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(level=logging.INFO)
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

random.seed(112)

model_name = "HuggingFaceTB/SmolLM-135M"

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name,torch_dtype=torch.bfloat16).to(device)


def tracefunc(frame, event, arg, indent=None):
    if indent is None:
        indent = [0]
    if event == "call":
        indent[0] += 2
    elif event == "return":
        indent[0] -= 2
    return tracefunc


def Prompting(model, prompt, candidate_premature_layers):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    hidden_states, outputs = model.generate(
        input_ids=inputs.input_ids, max_new_tokens=64, candidate_premature_layers=candidate_premature_layers, output_hidden_states=True
    )
    # hidden_states, outputs = model.generate(**{'input_ids':inputs.input_ids})
    hidden_embed = {}
    hidden_embed_token_level = {}
    for _i, early_exit_layer in enumerate(candidate_premature_layers):
        hidden_embed[early_exit_layer] = tokenizer.decode(hidden_states[early_exit_layer][0])
        hidden_embed_token_level[early_exit_layer] = [tokenizer.decode(tok) for tok in hidden_states[early_exit_layer][0]]
    answer = tokenizer.decode(outputs[0]).replace("<pad> ", "")
    answer = answer.replace("</s>", "")

    return hidden_embed, hidden_embed_token_level, answer


def layerwise_lang_stats(hidden_embed_token_level, candidate_langs=None):
    if candidate_langs is None:
        candidate_langs = ["en", "zh", "es", "ru", "de", "fr"]
    lang_stats = {}
    for layer in hidden_embed_token_level:
        lang_stats[layer] = {"total_count": 0}
        for token in hidden_embed_token_level[layer]:
            try:
                lang_pred = detect_langs(token)
            except:
                lang_pred = None
            if lang_pred and lang_pred[0].prob > 0.5 and (lang_pred[0].lang in candidate_langs):
                lang_stats[layer]["total_count"] += 1
                if lang_pred[0].lang in lang_stats[layer]:
                    lang_stats[layer][lang_pred[0].lang] += 1
                else:
                    lang_stats[layer][lang_pred[0].lang] = 1
    return lang_stats


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


def plot_lang_distribution(lang_distribution, candidate_langs, candidate_layers, show_plot=False) -> None:
    image_dir = Path.cwd() / "plots"
    image_dir.mkdir(exist_ok=True)

    lang_distribution_matrix = []
    for layer in candidate_layers:
        lang_distribution_matrix.append([lang_distribution[layer][lang] for lang in candidate_langs])
    lang_distribution_matrix = np.array(lang_distribution_matrix).T
    fig, ax = plt.subplots(figsize=(11, 3))
    cmap = sns.color_palette("ch:start=.2,rot=-.3", as_cmap=True)
    sns.heatmap(lang_distribution_matrix, ax=ax, xticklabels=candidate_layers, yticklabels=candidate_langs, cmap=cmap)
    plt.title("Layerwise Language Distribution")
    plt.xlabel("Layer")
    plt.ylabel("Language")
    plt.savefig(image_dir / model_name.replace("/","-") +"_lang_distribution.png")
    if show_plot:
        plt.show()


def main(argv) -> None:
    de_prompts = [
        "Frage: Was sind die besten deutschen Filme? Antwort: ",
        "Frage: Was sind die besten deutschen Bücher? Antwort: ",
        "Frage: Wie kann ich meine Deutschkenntnisse verbessern? Antwort: ",
        "Kann mir jemand ein gutes Restaurant in München empfehlen?",
        "Was sind die besten Sehenswürdigkeiten in Berlin?",
        "Wie kann ich meine Deutschkenntnisse verbessern?",
        "Was sind die besten Tipps für einen guten Schlaf?",
        "Wo kann ich in Hamburg gut shoppen?",
        "Wie kann ich meine Zeit besser nutzen?",
        "Was sind die besten deutschen Serien?",
        "Was sind die besten deutschen Lieder?"
        "Kannst du drei günstige Reiseziele in Österreich für Einzelreisende empfehlen?",
        "Was sind effektive Strategien zur Stressbewältigung?"
    ]

    en_prompts = [
        "What are some popular tourist attractions in New York City?",
        "How can I improve my English writing skills?",
        "Can you recommend three must-read books from the science fiction genre?",
        "What are some effective strategies for time management?",
        "Where can I find authentic Italian cuisine in London?",
        "What are some tips for maintaining a healthy lifestyle?",
        "Can you suggest three classic movies from the 20th century?",
        "How can I develop good public speaking skills?",
        "What are some unique cultural traditions in Japan?",
        "Can you recommend three budget-friendly destinations for solo travelers?"
    ]

    prompts = en_prompts + de_prompts

    candidate_premature_layers = list(
        range(model.config.num_hidden_layers + 1)
    )  # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28,29,30]

    candidate_langs = ["en", "de"]

    candidate_layers = list(
        range(model.config.num_hidden_layers + 1)
    )  # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28,29,30]

    lst_lang_distribution = []
    for prompt in tqdm(prompts):
        hidden_embed, hidden_embed_token_level, answer = Prompting(model, prompt, candidate_premature_layers)
        lang_stats = layerwise_lang_stats(hidden_embed_token_level, candidate_langs)

        # if only draw english and non-english
        # lang_distribution = layerwise_lang_distribution_bi(lang_distribution)

        lang_distribution = layerwise_lang_distribution(lang_stats, candidate_langs)
        lst_lang_distribution.append(lang_distribution)

    # if only draw english and non-english
    # average_lang_distribution = average_layerwise_lang_distribution(lst_lang_distribution, candidate_langs=['en', 'non-en'])
    # plot_lang_distribution(average_lang_distribution, candidate_langs=['en', 'non-en'], candidate_layers=candidate_layers)

    # if draw all languages independently
    average_lang_distribution = average_layerwise_lang_distribution(lst_lang_distribution, candidate_langs)
    plot_lang_distribution(average_lang_distribution, candidate_langs, candidate_layers=candidate_layers)
    # pdb.set_trace()


if __name__ == "__main__":
    # sys.setprofile(tracefunc)
    main(sys.argv[1:])
