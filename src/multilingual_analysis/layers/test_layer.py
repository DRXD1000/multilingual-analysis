
import os
import random
import time
from pathlib import Path

import torch
from datasets import load_dataset
from loguru import logger
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from multilingual_analysis.layers.calculate_stats import layerwise_lang_stats
from multilingual_analysis.layers.layer_distributions import average_layerwise_lang_distribution, layerwise_lang_distribution
from multilingual_analysis.layers.plotting import create_line_plot, plot_lang_distribution

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

random.seed(112)

model_name = "ibm-granite/granite-3.1-1b-a400m-base"#"ibm-granite/granite-3.1-2b-base"#"BSC-LT/salamandra-2b"

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name,torch_dtype=torch.bfloat16).to(device)

if not tokenizer.pad_token:
    tokenizer.pad_token = tokenizer.eos_token

def BatchedPrompting(model, prompts:list[str], candidate_premature_layers:list[int], batch_size:int=10):
    """Processes a batch of prompts to generate hidden embeddings, token-level hidden embeddings, and answers for each prompt using the given model."""
    results = []

    # Process prompts in batches
    for batch_start in tqdm(range(0, len(prompts), batch_size),desc="Processing Prompts"):
        batch_prompts = prompts[batch_start:batch_start + batch_size]
        inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True).to(device)
        hidden_states, outputs = model.generate(
            input_ids=inputs.input_ids,
            max_new_tokens=128,
            candidate_premature_layers=candidate_premature_layers,
            output_hidden_states=True
        )

        for i, _ in enumerate(batch_prompts):
            hidden_embed = {}
            hidden_embed_token_level = {}

            for early_exit_layer in candidate_premature_layers:
                hidden_embed[early_exit_layer] = tokenizer.decode(hidden_states[early_exit_layer][i])
                hidden_embed_token_level[early_exit_layer] = [
                    tokenizer.decode(tok) for tok in hidden_states[early_exit_layer][i]
                ]

            answer = tokenizer.decode(outputs[i],skip_special_tokens=True)

            results.append({
                "hidden_embed": hidden_embed,
                "hidden_embed_token_level": hidden_embed_token_level,
                "answer": answer
            })

    del model
    torch.cuda.empty_cache()

    return results

def main() -> None:
    """Analyze the model."""
    ds = load_dataset("maxidl/no_robots-de")["train"]
    prompts = []
    index = 0
    while True:
        # Check if the first message is a system message or user message
        entry = ds[index]["messages_de"][:2] if ds[index]["messages_de"][0]["role"] == "system" else [ds[index]["messages_de"][0]]

        if tokenizer.chat_template is not None:
        # Calculate the length after tokenization
            length = len(tokenizer.apply_chat_template(entry, tokenize=True))

            # Add to prompts if the length is less than 200
            if length < 200 and tokenizer.chat_template is not None:
                prompts.append(tokenizer.apply_chat_template(entry, tokenize=False))
        else:
            length = len(tokenizer(entry[-1]["content"]))
            if length < 200:
                prompts.append(entry[-1]["content"])

        index += 1

        # Break the loop if we have collected 40 prompts
        if len(prompts) >= 2:
            break

    candidate_premature_layers = list(range(model.config.num_hidden_layers + 1))

    candidate_langs = ["en", "de"]

    candidate_layers = list(range(model.config.num_hidden_layers + 1))

    lst_lang_distribution = []
    logger.info("Starting Processing")
    time_start = time.time()

    batched_results = BatchedPrompting(model, prompts, candidate_premature_layers, 2)

    logger.info(f"Batching took {time.time() - time_start:.2f} seconds")

    language_distributions  = []

    for result in tqdm(batched_results,desc="Analyzing Tokens"):
        lang_stats = layerwise_lang_stats(result["hidden_embed_token_level"], candidate_langs)
        language_distributions.append(lang_stats)
        # if only draw english and non-english
        # lang_distribution = layerwise_lang_distribution_bi(lang_distribution)

        lang_distribution = layerwise_lang_distribution(lang_stats, candidate_langs)
        lst_lang_distribution.append(lang_distribution)

    # if only draw english and non-english
    # average_lang_distribution = average_layerwise_lang_distribution(lst_lang_distribution, candidate_langs=['en', 'non-en'])
    # plot_lang_distribution(average_lang_distribution, candidate_langs=['en', 'non-en'], candidate_layers=candidate_layers)

    # if draw all languages independently
    average_lang_distribution = average_layerwise_lang_distribution(lst_lang_distribution, candidate_langs)

    image_dir = Path.cwd() / "plots"
    image_dir.mkdir(exist_ok=True)
    dist_file_name = model_name.replace("/","-") +"_lang_distribution.png"
    dist_save_path = image_dir  /  dist_file_name

    plot_lang_distribution(average_lang_distribution, candidate_langs, candidate_layers=candidate_layers,save_path=dist_save_path)
    line_file_name = model_name.replace("/","-") +"_line_plot.png"
    line_save_path = image_dir / line_file_name
    create_line_plot(language_distributions,save_path=line_save_path)
    # pdb.set_trace()

if __name__ == "__main__":
    main()
