import os
import torch
from tqdm import tqdm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datasets import load_dataset
from sae_lens import SAE
from transformer_lens import HookedTransformer
from sae_bench.evals.sparse_probing.eval_config import SparseProbingEvalConfig
from typing import List, Optional
import argparse
import json
import sae_bench.sae_bench_utils.general_utils as general_utils
import sae_bench.sae_bench_utils.activation_collection as activation_collection
plt.style.use('ggplot')
from sae_bench.sae_bench_utils.sae_selection_utils import (
    get_saes_from_regex,
)
#get_pretrained_saes_directory

# Determine the available device
device = "cuda:0" if torch.cuda.is_available() else "cpu"

def create_config_and_selected_saes(
    args,
) -> tuple[SparseProbingEvalConfig, list[tuple[str, str]]]:
    config = SparseProbingEvalConfig(
        model_name=args.model_name,
    )

    if args.llm_batch_size is not None:
        config.llm_batch_size = args.llm_batch_size
    else:
        config.llm_batch_size = activation_collection.LLM_NAME_TO_BATCH_SIZE[
            config.model_name
        ]

    if args.llm_dtype is not None:
        config.llm_dtype = args.llm_dtype
    else:
        config.llm_dtype = activation_collection.LLM_NAME_TO_DTYPE[config.model_name]

    if args.sae_batch_size is not None:
        config.sae_batch_size = args.sae_batch_size

    if args.random_seed is not None:
        config.random_seed = args.random_seed

    if args.lower_vram_usage:
        config.lower_vram_usage = True

    selected_saes = get_saes_from_regex(args.sae_regex_pattern, args.sae_block_pattern)
    assert len(selected_saes) > 0, "No SAEs selected"

    releases = set([release for release, _ in selected_saes])

    print(f"Selected SAEs from releases: {releases}")

    for release, sae in selected_saes:
        print(f"Sample SAEs: {release}, {sae}")

    return config, selected_saes

def remove_batch_dim(tensor):
    """
    Removes the first dimension of a tensor if it is size 1.
    """
    return tensor.squeeze(0) if tensor.shape[0] == 1 else tensor

# Standard imports
import os
import torch
from tqdm import tqdm


# %pip install sae-lens==0.1.0
import sys
from transformer_lens import HookedTransformer
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.stats import sem
from statistics import mean


# %pip install sae-lens==0.1.0
import sys
sys.path.append("SAELens")
from sae_lens import SAE
from transformer_lens import HookedTransformer
from transformer_lens import utils
from functools import partial

from datasets import load_dataset
from transformer_lens import HookedTransformer


import argparse
plt.style.use('ggplot')
if torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

from typing import Any, Callable, Dict, List, Optional, Tuple, Union, cast
import pandas as pd


def remove_batch_dim(tensor):
    """
    Removes the first dimension of a tensor if it is size 1, otherwise returns the tensor unchanged
    """
    if tensor.shape[0] == 1:
        return tensor.squeeze(0)
    else:
        return tensor
    
def test_prompt_output(
    prompt: str,
    answer: str,
    model,  # Can't give type hint due to circular imports
    prepend_space_to_answer: bool = True,
    print_details: bool = True,
    prepend_bos: Optional[bool] = None,
    top_k: int = 10,
    
) -> None:
    """Test if the Model Can Give the Correct Answer to a Prompt.

    Intended for exploratory analysis. Prints out the performance on the answer (rank, logit, prob),
    as well as the top k tokens. Works for multi-token prompts and multi-token answers.

    Warning:

    This will print the results (it does not return them).

    Examples:

    >>> from transformer_lens import HookedTransformer, utils
    >>> model = HookedTransformer.from_pretrained("tiny-stories-1M")
    Loaded pretrained model tiny-stories-1M into HookedTransformer

    >>> prompt = "Why did the elephant cross the"
    >>> answer = "road"
    >>> utils.test_prompt(prompt, answer, model)
    Tokenized prompt: ['<|endoftext|>', 'Why', ' did', ' the', ' elephant', ' cross', ' the']
    Tokenized answer: [' road']
    Performance on answer token:
    Rank: 2        Logit: 14.24 Prob:  3.51% Token: | road|
    Top 0th token. Logit: 14.51 Prob:  4.59% Token: | ground|
    Top 1th token. Logit: 14.41 Prob:  4.18% Token: | tree|
    Top 2th token. Logit: 14.24 Prob:  3.51% Token: | road|
    Top 3th token. Logit: 14.22 Prob:  3.45% Token: | car|
    Top 4th token. Logit: 13.92 Prob:  2.55% Token: | river|
    Top 5th token. Logit: 13.79 Prob:  2.25% Token: | street|
    Top 6th token. Logit: 13.77 Prob:  2.21% Token: | k|
    Top 7th token. Logit: 13.75 Prob:  2.16% Token: | hill|
    Top 8th token. Logit: 13.64 Prob:  1.92% Token: | swing|
    Top 9th token. Logit: 13.46 Prob:  1.61% Token: | park|
    Ranks of the answer tokens: [(' road', 2)]

    Args:
        prompt:
            The prompt string, e.g. "Why did the elephant cross the".
        answer:
            The answer, e.g. "road". Note that if you set prepend_space_to_answer to False, you need
            to think about if you have a space before the answer here (as e.g. in this example the
            answer may really be " road" if the prompt ends without a trailing space).
        model:
            The model.
        prepend_space_to_answer:
            Whether or not to prepend a space to the answer. Note this will only ever prepend a
            space if the answer doesn't already start with one.
        print_details:
            Print the prompt (as a string but broken up by token), answer and top k tokens (all
            with logit, rank and probability).
        prepend_bos:
            Overrides self.cfg.default_prepend_bos if set. Whether to prepend
            the BOS token to the input (applicable when input is a string). Models generally learn
            to use the BOS token as a resting place for attention heads (i.e. a way for them to be
            "turned off"). This therefore often improves performance slightly.
        top_k:
            Top k tokens to print details of (when print_details is set to True).

    Returns:
        None (just prints the results directly).
    """
    if prepend_space_to_answer and not answer.startswith(" "):
        answer = " " + answer
    # GPT-2 often treats the first token weirdly, so lets give it a resting position
    prompt_tokens = model.to_tokens(prompt, prepend_bos=prepend_bos)
    answer_tokens = model.to_tokens(answer, prepend_bos=False)
    tokens = torch.cat((prompt_tokens, answer_tokens), dim=1)
    prompt_str_tokens = model.to_str_tokens(prompt, prepend_bos=prepend_bos)
    answer_str_tokens = model.to_str_tokens(answer, prepend_bos=False)
    prompt_length = len(prompt_str_tokens)
    answer_length = len(answer_str_tokens)
    # if print_details:
    #     print("Tokenized prompt:", prompt_str_tokens)
    #     print("Tokenized answer:", answer_str_tokens)
    logits = remove_batch_dim(model(tokens))
    probs = logits.softmax(dim=-1)
    answer_ranks = []
    for index in range(prompt_length, prompt_length + answer_length):
        answer_token = tokens[0, index]
        answer_str_token = answer_str_tokens[index - prompt_length]
        # Offset by 1 because models predict the NEXT token
        token_probs = probs[index - 1]
        sorted_token_probs, sorted_token_values = token_probs.sort(descending=True)
        # Janky way to get the index of the token in the sorted list - I couldn't find a better way?
        correct_rank = torch.arange(len(sorted_token_values))[
            (sorted_token_values == answer_token).cpu()
        ].item()
        answer_ranks.append((answer_str_token, correct_rank))
        df = pd.DataFrame()
        if print_details:
            # String formatting syntax - the first number gives the number of characters to pad to, the second number gives the number of decimal places.
            # rprint gives rich text printing
            for i in range(top_k):
                # if print_details:
                #     print(
                #         f"Top {i}th token. Logit: {logits[index-1, sorted_token_values[i]].item():5.2f} Prob: {sorted_token_probs[i].item():6.2%} Token: |{model.to_string(sorted_token_values[i])}|"
                #     )
                # append df
                df[model.to_string(sorted_token_values[i]) + "_logit"] = [logits[index-1, sorted_token_values[i]].item()]
                df[model.to_string(sorted_token_values[i]) + "_prob"] = [sorted_token_probs[i].item()]
    return df

def calc_diag(exp, model, saes, template):
    target_word = exp["target_word"]
    example_prompt1 = template.format(context=exp["context_1"], target_word=target_word)
    example_prompt2 = template.format(context=exp["context_2"], target_word=target_word)
    # print(example_prompt1)
    # print(example_prompt2)
    # print(target_word)
    c1_maxlist = []
    c2_maxlist = []
    tokens1 = model.to_tokens(example_prompt1)[0]
    to_str_tokens1 = model.to_str_tokens(example_prompt1, prepend_bos=True)
    if " " + target_word in to_str_tokens1:
        id_target_word1 = [i for i, x in enumerate(to_str_tokens1) if x == " " + target_word][-1]
        # print("id_target_word1", id_target_word1)
    else:
        # print("not found")
        return None
    target_word_token1 = tokens1[id_target_word1]
    token2 = model.to_tokens(example_prompt2)[0]
    to_str_tokens2 = model.to_str_tokens(example_prompt2, prepend_bos=True)
    if " " + target_word in to_str_tokens2:
        id_target_word2 = [i for i, x in enumerate(to_str_tokens2) if x == " " + target_word][-1]
        # print("id_target_word2", id_target_word2)
    else:
        # print("not found")
        return None
    target_word_token2 = token2[id_target_word2]
    if target_word_token1 != target_word_token2:
        # print("not same toknen")
        return None
    for sae in saes:
        logits, cache = model.run_with_cache(example_prompt1, prepend_bos=True)
        tokens = model.to_tokens(example_prompt1)
        sae_out = sae(cache[sae.cfg.hook_name])
        sae_act = sae.encode(cache[sae.cfg.hook_name])
            
        df = pd.DataFrame(sae_act.detach().cpu().numpy()[0][0])
        for i in range(1, sae_act.shape[1]):
            df = pd.concat([df, pd.DataFrame(sae_act.detach().cpu().numpy()[0][i])], axis=1)
        df.columns = [f"{model.to_str_tokens(example_prompt1, prepend_bos=True)[i]}" for i in range(sae_act.shape[1])]
        df.index = [f"{i}" for i in range(sae_act.shape[-1])]
        for i in range(1, sae_act.shape[1]):
        # plt.scatter(range(sae_act.shape[-1]), df[f"{model.to_str_tokens(example_prompt, prepend_bos=True)[i]}"], label=f"{model
            if target_word in model.to_str_tokens(example_prompt1, prepend_bos=True)[i]:
                try:
                    c1_maxlist.append(df[f"{model.to_str_tokens(example_prompt1, prepend_bos=True)[i]}"].iloc[:, -1].argmax())
                    break
                except:
                    # print("error")
                    return None

    for sae in saes:
        logits, cache = model.run_with_cache(example_prompt2, prepend_bos=True)
        tokens = model.to_tokens(example_prompt2)
        sae_out = sae(cache[sae.cfg.hook_name])
        sae_act = sae.encode(cache[sae.cfg.hook_name])

        df = pd.DataFrame(sae_act.detach().cpu().numpy()[0][0])
        for i in range(1, sae_act.shape[1]):
            df = pd.concat([df, pd.DataFrame(sae_act.detach().cpu().numpy()[0][i])], axis=1)
        df.columns = [f"{model.to_str_tokens(example_prompt2, prepend_bos=True)[i]}" for i in range(sae_act.shape[1])]
        df.index = [f"{i}" for i in range(sae_act.shape[-1])]
        for i in range(1, sae_act.shape[1]):
        # plt.scatter(range(sae_act.shape[-1]), df[f"{model.to_str_tokens(example_prompt, prepend_bos=True)[i]}"], label=f"{model
            if target_word in model.to_str_tokens(example_prompt2, prepend_bos=True)[i]:
                try:
                    c2_maxlist.append(df[f"{model.to_str_tokens(example_prompt2, prepend_bos=True)[i]}"].iloc[:, -1].argmax())
                    break
                except:
                    # print("error")
                    return None

    if len(c1_maxlist) != len(c2_maxlist) and len(c1_maxlist) != len(saes):
        # print(len(c1_maxlist), len(c2_maxlist), len(saes))
        return None

    max_enc_weights1 = []
    max_dec_weights1 = []
    max_enc_weights2 = []
    max_dec_weights2 = []

    for (layer_i, sae), (max_ind1, max_ind2) in zip(enumerate(saes), zip(c1_maxlist, c2_maxlist)):
        max_enc_weight1 = sae.W_enc[:, max_ind1]
        max_dec_weight1 = sae.W_dec[max_ind1, :]
        max_enc_weight2 = sae.W_enc[:, max_ind2]
        max_dec_weight2 = sae.W_dec[max_ind2, :]
        # cos 類似度　torch
        cos_sim = torch.nn.functional.cosine_similarity(max_enc_weight1, max_dec_weight2, dim=0)
        max_enc_weights1.append(max_enc_weight1)
        max_enc_weights2.append(max_enc_weight2)
        max_dec_weights1.append(max_dec_weight1)
        max_dec_weights2.append(max_dec_weight2)

    # cos sim matrix
    cos_sim_matrix_enc = torch.zeros(len(saes), len(saes))
    if not( len(max_enc_weights1) == len(max_enc_weights2) == len(saes)):
        # print(len(max_enc_weights1), len(max_enc_weights2), len(saes))
        return None

    for layer_i in range(len(saes)):
        for layer_j in range(len(saes)):
            cos_sim = torch.nn.functional.cosine_similarity(max_enc_weights1[layer_i], max_enc_weights2[layer_j], dim=0)
            cos_sim_matrix_enc[layer_i, layer_j] = cos_sim

    cos_sim_matrix_dec = torch.zeros(len(saes), len(saes))
    for layer_i in range(len(saes)):
        for layer_j in range(len(saes)):
            cos_sim = torch.nn.functional.cosine_similarity(max_dec_weights1[layer_i], max_dec_weights2[layer_j], dim=0)
            cos_sim_matrix_dec[layer_i, layer_j] = cos_sim

    # px.imshow(cos_sim_matrix_enc, title="W_enc cos similarity").show()
    # px.imshow(cos_sim_matrix_dec, title="W_dec cos similarity").show()
    # 対角成分
    cos_sim_matrix_enc_diag = torch.diag(cos_sim_matrix_enc).detach().cpu().numpy()
    cos_sim_matrix_dec_diag = torch.diag(cos_sim_matrix_dec).detach().cpu().numpy()
    # px.line(cos_sim_matrix_enc_diag, title="W

    return cos_sim_matrix_enc_diag, cos_sim_matrix_dec_diag

def calc_f_value(exp, model, saes, template):
    target_word = exp["target_word"]
    example_prompt1 = template.format(context=exp["context_1"], target_word=target_word)
    example_prompt2 = template.format(context=exp["context_2"], target_word=target_word)
    # print(example_prompt1)
    # print(example_prompt2)
    # print(target_word)
    c1_maxlist = []
    c2_maxlist = []
    tokens1 = model.to_tokens(example_prompt1)[0]
    to_str_tokens1 = model.to_str_tokens(example_prompt1, prepend_bos=True)
    if " " + target_word in to_str_tokens1:
        id_target_word1 = [i for i, x in enumerate(to_str_tokens1) if x == " " + target_word][-1]
        # print("id_target_word1", id_target_word1)
    else:
        # print("not found")
        return None
    target_word_token1 = tokens1[id_target_word1]
    token2 = model.to_tokens(example_prompt2)[0]
    to_str_tokens2 = model.to_str_tokens(example_prompt2, prepend_bos=True)
    if " " + target_word in to_str_tokens2:
        id_target_word2 = [i for i, x in enumerate(to_str_tokens2) if x == " " + target_word][-1]
        # print("id_target_word2", id_target_word2)
    else:
        # print("not found")
        return None
    target_word_token2 = token2[id_target_word2]
    if target_word_token1 != target_word_token2:
        # print("not same toknen")
        return None
    for sae in saes:
        logits, cache = model.run_with_cache(example_prompt1, prepend_bos=True)
        tokens = model.to_tokens(example_prompt1)
        sae_out = sae(cache[sae.cfg.hook_name])
        sae_act = sae.encode(cache[sae.cfg.hook_name])

        df = pd.DataFrame(sae_act.detach().cpu().numpy()[0][0])
        for i in range(1, sae_act.shape[1]):
            df = pd.concat([df, pd.DataFrame(sae_act.detach().cpu().numpy()[0][i])], axis=1)
        df.columns = [f"{model.to_str_tokens(example_prompt1, prepend_bos=True)[i]}" for i in range(sae_act.shape[1])]
        df.index = [f"{i}" for i in range(sae_act.shape[-1])]
        for i in range(1, sae_act.shape[1]):
            if target_word in model.to_str_tokens(example_prompt1, prepend_bos=True)[i]:
                try:
                    c1_maxlist.append(df[f"{model.to_str_tokens(example_prompt1, prepend_bos=True)[i]}"].iloc[:, -1].argmax())
                    break
                except:
                    # print("error")
                    return None

    for sae in saes:
        logits, cache = model.run_with_cache(example_prompt2, prepend_bos=True)
        tokens = model.to_tokens(example_prompt2)
        sae_out = sae(cache[sae.cfg.hook_name])
        sae_act = sae.encode(cache[sae.cfg.hook_name])

        df = pd.DataFrame(sae_act.detach().cpu().numpy()[0][0])
        for i in range(1, sae_act.shape[1]):
            df = pd.concat([df, pd.DataFrame(sae_act.detach().cpu().numpy()[0][i])], axis=1)
        df.columns = [f"{model.to_str_tokens(example_prompt2, prepend_bos=True)[i]}" for i in range(sae_act.shape[1])]
        df.index = [f"{i}" for i in range(sae_act.shape[-1])]
        for i in range(1, sae_act.shape[1]):
        # plt.scatter(range(sae_act.shape[-1]), df[f"{model.to_str_tokens(example_prompt, prepend_bos=True)[i]}"], label=f"{model
            if target_word in model.to_str_tokens(example_prompt2, prepend_bos=True)[i]:
                try:
                    c2_maxlist.append(df[f"{model.to_str_tokens(example_prompt2, prepend_bos=True)[i]}"].iloc[:, -1].argmax())
                    break

                except:
                    # print("error")
                    return None

    if len(c1_maxlist) != len(c2_maxlist) and len(c1_maxlist) != len(saes):
        # print(len(c1_maxlist), len(c2_maxlist), len(saes))
        return None

    same_num = np.zeros(len(c1_maxlist))
    # print(c1_maxlist)
    # print(c2_maxlist)
    for i in range(len(c1_maxlist)):
        if c1_maxlist[i] == c2_maxlist[i]:
            same_num[i] = 1
    # print(same_num)

    return  same_num

def calc_l0_norm(exp, model, saes, template):
    target_word = exp["target_word"]
    example_prompt1 = template.format(context=exp["context_1"], target_word=target_word)
    example_prompt2 = template.format(context=exp["context_2"], target_word=target_word)
    # print(example_prompt1)
    # print(example_prompt2)
    # print(target_word)
    sae_act = []
    for sae in saes:
        logits, cache = model.run_with_cache(example_prompt1, prepend_bos=True)
        sae_out = sae.encode(cache[sae.cfg.hook_name])
        sae_act.append(sae_out)
    sae_act = torch.cat(sae_act, dim=0)
    l0_norm = (sae_act > 0).float().sum(-1).mean()
    return l0_norm.detach().cpu().numpy()
    
def calc_mse(exp, model, saes, template):
    target_word = exp["target_word"]
    example_prompt1 = template.format(context=exp["context_1"], target_word=target_word)
    example_prompt2 = template.format(context=exp["context_2"], target_word=target_word)
    # print(example_prompt1)
    # print(example_prompt2)
    # print(target_word)
    mse_list = []
    for sae in saes:
        logits, cache = model.run_with_cache(example_prompt1, prepend_bos=True)
        sae_out = sae(cache[sae.cfg.hook_name])
        # mse = torch.norm(sae_out - cache[sae.cfg.hook_name], p=2).mean()
        mse = torch.nn.functional.mse_loss(sae_out, cache[sae.cfg.hook_name], reduction="none").sum(-1).mean()
        mse_list.append(mse)
    mse = torch.stack(mse_list)
    return mse.detach().cpu().numpy()
    


def prep_model(model_name, sae_release, sae_object_or_id, device):

    model = HookedTransformer.from_pretrained(model_name, device = device)

    # pretrain

    # sae, cfg_dict, sparsity = SAE.from_pretrained(
    #     release = "gpt2-small-res-jb", # see other options in sae_lens/pretrained_saes.yaml
    #     sae_id = "blocks.0.hook_resid_pre", # won't always be a hook point
    #     device = device
    # )
    sae_id, sae, sparsity = general_utils.load_and_format_sae(
    sae_release, sae_object_or_id, device
    )
    
    return model, sae
    

def run_eval(model_name, sae_release, sae_object_or_id, output_folder):
    device = "cuda"
    model, sae = prep_model(model_name, sae_release, sae_object_or_id, device)
    template = "{context} The {target_word} means"
    wic_dataset = load_dataset('gouki510/Wic_data_for_SAE-Eval')
    saes = [sae]
    
    results = {}
    
    # W similarity
    print("-"*10)
    print("W_enc cos similarity")
    print("-"*10)
    l0_norm_list = []
    mse_list = []
    for label in range(2):
        print(f"label:{label}")
        exp1_df = pd.DataFrame(wic_dataset["test"])
        exp1_df = exp1_df[exp1_df["label"] == label]
        cos_sim_list = []
        cnt = 0
        for i in tqdm(range(len(exp1_df))):
            exp1 = exp1_df.iloc[i]
            cos_sim = calc_diag(exp1, model, saes, template)
            if cos_sim is not None:
                cos_sim_list.append(cos_sim)
                l0_norm_list.append(calc_l0_norm(exp1, model, saes, template))
                mse_list.append(calc_mse(exp1, model, saes, template))
                cnt+=1
        cos_sim = np.array(cos_sim_list)[:,0]
        mean_std_cossim = np.mean(cos_sim, axis=0), np.std(cos_sim, axis=0)
        fig, ax = plt.subplots(1,2)
        layer = list(range(len(mean_std_cossim[0])))
        ax[0].bar(layer, mean_std_cossim[0], yerr=mean_std_cossim[1])
        ax[0].set_title("W_enc cos similarity")
        results[f"W_enc cos similarity label{label}"] = mean_std_cossim[0].mean()
        cos_sim = np.array(cos_sim_list)[:,1]
        mean_std_cossim = np.mean(cos_sim, axis=0), np.std(cos_sim, axis=0)
        ax[1].bar(layer, mean_std_cossim[0], yerr=mean_std_cossim[1])
        ax[1].set_title("W_dec cos similarity")
        ax[0].set_xlabel("layer")
        ax[1].set_xlabel("layer")
        ax[0].set_ylabel("cos similarity")
        ax[1].set_ylabel("cos similarity")
        fig.savefig(os.path.join(output_folder, f"wic_eval_sim_label{label}.png"))
        results[f"W_dec cos similarity label{label}"] = mean_std_cossim[0].mean()
    
    mse = np.array(mse_list)
    l0_norm = np.array(l0_norm_list)
    mean_std_mse = np.mean(mse, axis=0), np.std(mse, axis=0)
    plt.bar(layer, mean_std_mse[0], yerr=mean_std_mse[1])
    plt.title("MSE")
    plt.xlabel("layer")
    plt.ylabel("MSE")
    plt.savefig(os.path.join(output_folder, f"wic_eval_mse.png"))
    results[f"MSE"] = mean_std_mse[0].mean()
    results[f"l0_norm"] = l0_norm.mean()
    print("MSE", mean_std_mse[0].mean())
    print("L0_norm", l0_norm.mean())
    # F value
    exp1_df = pd.DataFrame(wic_dataset["test"])
    exp1_df = exp1_df[exp1_df["label"] == 0]
    fp_list = []
    tn_list = []
    cnt_0 = 0
    print("-"*10)
    print("calc_f_value")
    print("-"*10)
    for i in tqdm(range(len(exp1_df))):
        exp1 = exp1_df.iloc[i]
        FP = calc_f_value(exp1, model, saes, template)
        if FP is not None:
            fp_list.append(FP)
            TN = 1 - FP
            tn_list.append(TN)
            cnt_0+=1
    exp1_df = pd.DataFrame(wic_dataset["test"])
    exp1_df = exp1_df[exp1_df["label"] == 1]
    fn_list = []
    tp_list = []
    cnt_1 = 0
    for i in tqdm(range(len(exp1_df))[:cnt_0]):
        exp1 = exp1_df.iloc[i]
        TP = calc_f_value(exp1, model, saes, template)
        if TP is not None:  
            tp_list.append(TP)
            FN = 1 - TP 
            fn_list.append(FN)
            cnt_1+=1 
    tp = np.sum(tp_list)
    fp = np.sum(fp_list)
    tn = np.sum(tn_list)
    fn = np.sum(fn_list)
    print("TP", tp)
    print("FP", fp)
    print("TN", tn)
    print("FN", fn)
    f_value =  (2*tp)/(2*tp + fp + fn)
    print("f_value", f_value)
    recall = tp/(tp+fn)
    precision = tp/(tp+fp)
    print("recall", recall)
    print("precision", precision)
    results[f"f_value"] = f_value.mean()
    results[f"recall"] = recall.mean()
    results[f"precision"] = precision.mean()
    df = pd.DataFrame([tp.T, fp.T, tn.T, fn.T], index=["TP", "FP", "TN", "FN"])
    df = df.T
    plt.figure()
    df.plot.bar()
    plt.savefig(os.path.join(output_folder, f"wic_eval_tp_fp_tn_fn.png"))
    df["recall"] = df["TP"]/(df["TP"]+df["FN"])
    df["precision"] = df["TP"]/(df["TP"]+df["FP"])
    df["f_value"] =  (2*df["TP"])/(2*df["TP"] + df["FP"] + df["FN"])
    plt.figure()
    df.plot.bar(y=["recall", "precision", "f_value"])
    plt.savefig(os.path.join(output_folder, f"wic_eval_recall_precision_fvalue.png"))
    df["W_enc_cos_sim_label0"] = results["W_enc cos similarity label0"]
    df["W_dec_cos_sim_label0"] = results["W_dec cos similarity label0"]
    df["W_enc_cos_sim_label1"] = results["W_enc cos similarity label1"]
    df["W_dec_cos_sim_label1"] = results["W_dec cos similarity label1"]
    df["MSE"] = results["MSE"]
    df["l0_norm"] = results["l0_norm"]
    df.to_csv(os.path.join(output_folder, f"wic_eval_result2.csv"))
    

def arg_parser():
    parser = argparse.ArgumentParser(description="Run sparse probing evaluation")
    parser.add_argument("--random_seed", type=int, default=None, help="Random seed")
    parser.add_argument("--model_name", type=str, required=True, help="Model name")
    parser.add_argument(
        "--sae_regex_pattern",
        type=str,
        required=True,
        help="Regex pattern for SAE selection",
    )
    parser.add_argument(
        "--sae_block_pattern",
        type=str,
        required=True,
        help="Regex pattern for SAE block selection",
    )
    parser.add_argument(
        "--output_folder",
        type=str,
        default="eval_results/sparse_probing",
        help="Output folder",
    )
    parser.add_argument(
        "--force_rerun", action="store_true", help="Force rerun of experiments"
    )
    parser.add_argument(
        "--clean_up_activations",
        action="store_true",
        help="Clean up activations after evaluation",
    )
    parser.add_argument(
        "--save_activations",
        action="store_false",
        help="Save the generated LLM activations for later use",
    )
    parser.add_argument(
        "--llm_batch_size",
        type=int,
        default=None,
        help="Batch size for LLM. If None, will be populated using LLM_NAME_TO_BATCH_SIZE",
    )
    parser.add_argument(
        "--llm_dtype",
        type=str,
        default=None,
        choices=[None, "float32", "float64", "float16", "bfloat16"],
        help="Data type for LLM. If None, will be populated using LLM_NAME_TO_DTYPE",
    )
    parser.add_argument(
        "--sae_batch_size",
        type=int,
        default=None,
        help="Batch size for SAE. If None, will be populated using default config value",
    )
    parser.add_argument(
        "--lower_vram_usage",
        action="store_true",
        help="Lower GPU memory usage by doing more computation on the CPU. Useful on 1M width SAEs. Will be slower and require more system memory.",
    )

    return parser

def main():
    args = arg_parser().parse_args()

    os.makedirs(args.output_folder, exist_ok=True)
    saes = get_saes_from_regex(args.sae_regex_pattern, args.sae_block_pattern)
    results = run_eval(args.model_name, saes[0][0], saes[0][1], args.output_folder)
    print("Evaluation Results:", results)

if __name__ == "__main__":
    main()
