import torch
import os
import sys
sys.path.append("SAELens")
from sae_lens import LanguageModelSAERunnerConfig, SAETrainingRunner

if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

print("Using device:", device)
os.environ["TOKENIZERS_PARALLELISM"] = "false"


from transformer_lens import HookedTransformer
from dataclasses import dataclass
from argparse import ArgumentParser
import yaml

@dataclass
class Train_config:
    model_name_or_path: str 
    output_dir: str
    total_training_steps: int
    batch_size: int
    hook_name: str
    hook_layer: int
    d_in: int
    dataset_path: str
    expansion_factor: int
    lr: float
    l1_coefficient: float
    context_size: int
    wandb_project: str
    device: str
    lp_norm: float
    activation_fn: str
    use_quadratic_activation: bool
    use_latent_norm: bool
    use_node_popup : bool
    init_dec_orthogonal: bool
    dec_orthogonal_coefficient: float
    use_ghost_grads: bool
    datadir: list[str] | None = None
    W_restart: bool = False
    k: int = 1
    jump: float = 0.001
    normalize_sae_decoder: bool = False
    scale_sparsity_penalty_by_decoder_norm: bool = False
    decoder_heuristic_init: bool = False
    init_encoder_as_decoder_transpose: bool = False

def main(config):
    model = HookedTransformer.from_pretrained(
    config.model_name_or_path,
    )  # This will wrap huggingface models and has lots of nice utilities.
    total_training_steps = config.total_training_steps
    batch_size = config.batch_size
    total_training_tokens = total_training_steps * batch_size

    lr_warm_up_steps = 0
    # lr_decay_steps = total_training_steps // 5  # 20% of training
    lr_decay_steps = 0
    l1_warm_up_steps = 0  # 5% of training

    cfg = LanguageModelSAERunnerConfig(
        # Data Generating Function (Model + Training Distibuion)
        model_name=config.model_name_or_path,
        hook_name=config.hook_name,
        hook_layer=config.hook_layer,
        d_in=config.d_in,
        dataset_path=config.dataset_path,
        datadir=config.datadir,
        is_dataset_tokenized=True,
        streaming=True,  # we could pre-download the token dataset if it was small.
        # SAE Parameters
        mse_loss_normalization=None,  # We won't normalize the mse loss,
        expansion_factor=config.expansion_factor,  # We'll expand the features by 16x
        b_dec_init_method="zeros",  # The geometric median can be used to initialize the decoder weights.
        apply_b_dec_to_input=False,  # We won't apply the decoder weights to the input.
        normalize_sae_decoder=config.normalize_sae_decoder,
        scale_sparsity_penalty_by_decoder_norm=config.scale_sparsity_penalty_by_decoder_norm, # True
        decoder_heuristic_init=config.decoder_heuristic_init, # True
        init_encoder_as_decoder_transpose=config.init_encoder_as_decoder_transpose,
        activation_fn=config.activation_fn,
        use_quadratic_activation=config.use_quadratic_activation,
        use_latent_norm=config.use_latent_norm,
        use_node_popup=config.use_node_popup,
        init_dec_orthogonal=config.init_dec_orthogonal,
        dec_orthogonal_coefficient=config.dec_orthogonal_coefficient,
        W_restart=config.W_restart,
        # normalize_activations=True,
        # Training Parameters
        lr=config.lr,  # we'll use a constant learning rate.
        adam_beta1=0.9,  # adam params (default, but once upon a time we experimented with these.)
        adam_beta2=0.999,
        # lr_scheduler_name="constant",  # constant learning rate with warmup. Could be better schedules out there.
        lr_scheduler_name="cosineannealingwarmrestarts",
        lr_warm_up_steps=lr_warm_up_steps,  # this can help avoid too many dead features initially.
        lr_decay_steps=lr_decay_steps,  # this will help us avoid overfitting.
        l1_coefficient=config.l1_coefficient,  # the L1 penalty (and not a Lp for p < 1)
        l1_warm_up_steps=l1_warm_up_steps,  # this can help avoid too many dead features initially.
        lp_norm=config.lp_norm,# the L1 penalty (and not a Lp for p < 1)
        train_batch_size_tokens=batch_size,
        context_size=config.context_size,
        # Activation Store Parameters
        n_batches_in_buffer=64,  # controls how many activations we store / shuffle.
        training_tokens=total_training_tokens,  # 100 million tokens is quite a few, but we want to see good stats. Get a coffee, come back.
        store_batch_size_prompts=16,
        # Resampling protocol
        feature_sampling_window=1000,  # this controls our reporting of feature sparsity stats
        dead_feature_window=1000,  # would effect resampling or ghost grads if we were using it.
        dead_feature_threshold=1e-4,  # would effect resampling or ghost grads if we were using it.
        # ghost grad
        use_ghost_grads=config.use_ghost_grads,
        # WANDB
        log_to_wandb=True,  # always use wandb unless you are just testing code.
        wandb_project=config.wandb_project,
        wandb_log_frequency=30,
        eval_every_n_wandb_logs=20,
        # Misc
        device=config.device,
        seed=42,
        n_checkpoints=10,
        checkpoint_path=os.path.join(config.output_dir,f"l1_{config.l1_coefficient}_expansion_{config.expansion_factor}"),
        dtype="float32",
        k=config.k,
        jump=config.jump,
        # from_pretrained_path="/content/checkpoints/i34i0h4u/38096896"
        )
    # look at the next cell to see some instruction for what to do while this is running.
    sparse_autoencoder = SAETrainingRunner(cfg).run()
    
    
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, default="tiny-stories-1L-21M")
    parser.add_argument("--output_dir", type=str, default="output")
    parser.add_argument("--total_training_steps", type=int, default=200000)
    parser.add_argument("--batch_size", type=int, default=8192)
    parser.add_argument("--hook_name", type=str, default="blocks.0.hook_mlp_out")
    parser.add_argument("--hook_layer", type=int, default=0)
    parser.add_argument("--d_in", type=int, default=1024)
    parser.add_argument("--dataset_path", type=str, default="NeelNanda/pile-10k" )#default="apollo-research/roneneldan-TinyStories-tokenizer-gpt2")
    parser.add_argument("--expansion_factor", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)   
    parser.add_argument("--l1_coefficient", type=float, default=5)
    parser.add_argument("--context_size", type=int, default=256)
    parser.add_argument("--wandb_project", type=str, default="wic_sae-ICLR2025-rebttal")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--lp_norm", type=float, default=1.0)
    parser.add_argument("--use_quadratic_activation", action="store_true")
    parser.add_argument("--activation_fn", type=str, default="relu")
    parser.add_argument("--use_latent_norm", action="store_true")
    parser.add_argument("--use_node_popup", action="store_true")
    parser.add_argument("--init_dec_orthogonal", action="store_true")
    parser.add_argument("--dec_orthogonal_coefficient", type=float, default=0.0)
    parser.add_argument("--use_ghost_grads", action="store_true")
    parser.add_argument("--datadir", type=str, nargs="*", default=None)
    parser.add_argument("--W_restart", action="store_true")
    parser.add_argument("--k", type=int, default=1)
    parser.add_argument("--jump", type=float, default=0.001)
    parser.add_argument("--normalize_sae_decoder", action="store_true")
    parser.add_argument("--scale_sparsity_penalty_by_decoder_norm", action="store_true")
    parser.add_argument("--decoder_heuristic_init", action="store_true")
    parser.add_argument("--init_encoder_as_decoder_transpose", action="store_true")
    
    args = parser.parse_args()
    
    main(Train_config(**vars(args)))