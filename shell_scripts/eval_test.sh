#/bin/bash

python -m sae_bench.evals.ps_eval.main \
    --sae_regex_pattern "sae_bench_pythia70m_sweep_standard_ctx128_0712" \
    --sae_block_pattern "blocks.4.hook_resid_post__trainer_10" \
    --model_name pythia-70m-deduped

# python -m sae_bench.evals.sparse_probing.main \
#     --sae_regex_pattern "sae_bench_pythia70m_sweep_standard_ctx128_0712" \
#     --sae_block_pattern "blocks.4.hook_resid_post__trainer_10" \
#     --model_name pythia-70m-deduped

# python -m sae_bench.evals.absorption.main \
#     --sae_regex_pattern "sae_bench_pythia70m_sweep_standard_ctx128_0712" \
#     --sae_block_pattern "blocks.4.hook_resid_post__trainer_10" \
#     --model_name pythia-70m-deduped

# python -m sae_bench.evals.autointerp.main \
#     --sae_regex_pattern "sae_bench_pythia70m_sweep_standard_ctx128_0712" \
#     --sae_block_pattern "blocks.4.hook_resid_post__trainer_10" \
#     --model_name pythia-70m-deduped

# python -m sae_bench.evals.core.main \
#     --sae_regex_pattern "sae_bench_pythia70m_sweep_standard_ctx128_0712" \
#     --sae_block_pattern "blocks.4.hook_resid_post__trainer_10" \
#     --model_name pythia-70m-deduped

# python -m sae_bench.evals.mdl.main \
#     --sae_regex_pattern "sae_bench_pythia70m_sweep_standard_ctx128_0712" \
#     --sae_block_pattern "blocks.4.hook_resid_post__trainer_10" \
#     --model_name pythia-70m-deduped


# python -m sae_bench.evals.ravel.main \
#     --sae_regex_pattern "sae_bench_pythia70m_sweep_standard_ctx128_0712" \
#     --sae_block_pattern "blocks.4.hook_resid_post__trainer_10" \
#     --model_name pythia-70m-deduped


# python -m sae_bench.evals.scr_and_tpp.main \
#     --sae_regex_pattern "sae_bench_pythia70m_sweep_standard_ctx128_0712" \
#     --sae_block_pattern "blocks.4.hook_resid_post__trainer_10" \
#     --model_name pythia-70m-deduped


# python -m sae_bench.evals.unlearning.main \
#     --sae_regex_pattern "sae_bench_gemma-2-2b_topk_width-2pow14_date-1109" \
#     --sae_block_pattern "blocks.5.hook_resid_post__trainer_2" \
#     --model_name gemma-2-2b-it

