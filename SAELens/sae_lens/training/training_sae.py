"""Most of this is just copied over from Arthur's code and slightly simplified:
https://github.com/ArthurConmy/sae/blob/main/sae/model.py
"""

import os
from dataclasses import dataclass
from typing import Any, Optional

import einops
import torch
from jaxtyping import Float
from torch import nn

from sae_lens.config import LanguageModelSAERunnerConfig
from sae_lens.sae import SAE, SAEConfig
from sae_lens.toolkit.pretrained_sae_loaders import (
    load_pretrained_sae_lens_sae_components,
)
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import torch
import math

from sae_lens.popup import GetSubnetNode

@dataclass
class TrainStepOutput:
    sae_in: torch.Tensor
    sae_out: torch.Tensor
    feature_acts: torch.Tensor
    loss: torch.Tensor  # we need to call backwards on this
    mse_loss: float
    l1_loss: float
    ghost_grad_loss: float
    dec_orthogonal_loss: float


@dataclass
class TrainingSAEConfig(SAEConfig):

    # Sparsity Loss Calculations
    l1_coefficient: float
    lp_norm: float = 0.0
    use_ghost_grads: bool = False
    normalize_sae_decoder: bool = False
    noise_scale: float = 0.0
    decoder_orthogonal_init: bool = False
    mse_loss_normalization: Optional[str] = None
    decoder_heuristic_init: bool = False
    init_encoder_as_decoder_transpose: bool = False
    scale_sparsity_penalty_by_decoder_norm: bool = False
    use_quadratic_activation: bool = False
    use_latent_norm: bool = False
    use_node_popup: bool = False
    dec_orthogonal_coefficient: float = 0.0
    init_dec_orthogonal: bool = False
    W_restart: bool = False
    k: int = 1
    jump: float = 0.001
    @classmethod
    def from_sae_runner_config(
        cls, cfg: LanguageModelSAERunnerConfig
    ) -> "TrainingSAEConfig":

        return cls(
            # base confg
            d_in=cfg.d_in,
            d_sae=cfg.d_sae,  # type: ignore
            dtype=cfg.dtype,
            device=cfg.device,
            model_name=cfg.model_name,
            hook_name=cfg.hook_name,
            hook_layer=cfg.hook_layer,
            hook_head_index=cfg.hook_head_index,
            activation_fn_str=cfg.activation_fn,
            apply_b_dec_to_input=cfg.apply_b_dec_to_input,
            finetuning_scaling_factor=cfg.finetuning_method is not None,
            sae_lens_training_version=cfg.sae_lens_training_version,
            context_size=cfg.context_size,
            dataset_path=cfg.dataset_path,
            prepend_bos=cfg.prepend_bos,
            # Training cfg
            l1_coefficient=cfg.l1_coefficient,
            lp_norm=cfg.lp_norm,
            use_ghost_grads=cfg.use_ghost_grads,
            normalize_sae_decoder=cfg.normalize_sae_decoder,
            noise_scale=cfg.noise_scale,
            decoder_orthogonal_init=cfg.decoder_orthogonal_init,
            mse_loss_normalization=cfg.mse_loss_normalization,
            decoder_heuristic_init=cfg.decoder_heuristic_init,
            init_encoder_as_decoder_transpose=cfg.init_encoder_as_decoder_transpose,
            scale_sparsity_penalty_by_decoder_norm=cfg.scale_sparsity_penalty_by_decoder_norm,
            normalize_activations=cfg.normalize_activations,
            W_restart=cfg.W_restart,
            jump=cfg.jump,
        )

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any]) -> "TrainingSAEConfig":
        return TrainingSAEConfig(**config_dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            **super().to_dict(),
            "l1_coefficient": self.l1_coefficient,
            "lp_norm": self.lp_norm,
            "use_ghost_grads": self.use_ghost_grads,
            "normalize_sae_decoder": self.normalize_sae_decoder,
            "noise_scale": self.noise_scale,
            "decoder_orthogonal_init": self.decoder_orthogonal_init,
            "init_encoder_as_decoder_transpose": self.init_encoder_as_decoder_transpose,
            "mse_loss_normalization": self.mse_loss_normalization,
            "decoder_heuristic_init": self.decoder_heuristic_init,
            "scale_sparsity_penalty_by_decoder_norm": self.scale_sparsity_penalty_by_decoder_norm,
            "normalize_activations": self.normalize_activations,
            "W_restart": self.W_restart,
            "jump": self.jump,
        }

    # this needs to exist so we can initialize the parent sae cfg without the training specific
    # parameters. Maybe there's a cleaner way to do this
    def get_base_sae_cfg_dict(self) -> dict[str, Any]:
        return {
            "d_in": self.d_in,
            "d_sae": self.d_sae,
            "activation_fn_str": self.activation_fn_str,
            "apply_b_dec_to_input": self.apply_b_dec_to_input,
            "dtype": self.dtype,
            "model_name": self.model_name,
            "hook_name": self.hook_name,
            "hook_layer": self.hook_layer,
            "hook_head_index": self.hook_head_index,
            "device": self.device,
            "context_size": self.context_size,
            "prepend_bos": self.prepend_bos,
            "finetuning_scaling_factor": self.finetuning_scaling_factor,
            "normalize_activations": self.normalize_activations,
            "dataset_path": self.dataset_path,
            "dataset_trust_remote_code": self.dataset_trust_remote_code,
            "sae_lens_training_version": self.sae_lens_training_version,
            "l1_coefficient": self.l1_coefficient,
            "init_dec_orthogonal": self.init_dec_orthogonal,
            "dec_orthogonal_coefficient": self.dec_orthogonal_coefficient,
            "W_restart": self.W_restart,
            "k": self.k,
            "jump": self.jump,
        }


class TrainingSAE(SAE):
    """
    A SAE used for training. This class provides a `training_forward_pass` method which calculates
    losses used for training.
    """

    cfg: TrainingSAEConfig
    use_error_term: bool
    dtype: torch.dtype
    device: torch.device

    def __init__(self, cfg: TrainingSAEConfig, use_error_term: bool = False):

        base_sae_cfg = SAEConfig.from_dict(cfg.get_base_sae_cfg_dict())
        super().__init__(base_sae_cfg)
        self.cfg = cfg  # type: ignore
        self.use_error_term = use_error_term

        self.initialize_weights_complex()

        # The training SAE will assume that the activation store handles
        # reshaping.
        self.turn_off_forward_pass_hook_z_reshaping()

        self.mse_loss_fn = self._get_mse_loss_fn()
        
        # for l0 norm
        self.d_sae = cfg.d_sae
        self.limit_a, self.limit_b, self.epsilon = -.1, 1.1, 1e-6
        self.qz_loga = Parameter(torch.Tensor(self.d_sae))
        self.temperature = 2/3
        self.floatTensor = torch.FloatTensor if not torch.cuda.is_available() else torch.cuda.FloatTensor
        self.prior_prec = 1.0
        self.lamba = cfg.l1_coefficient
        self.dec_orthogonal_coefficient = cfg.dec_orthogonal_coefficient
        
        # init nodepop
        self.use_node_popup = cfg.use_node_popup
        if self.use_node_popup:
            self.initialize_score_node_popup()

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any]) -> "TrainingSAE":
        return cls(TrainingSAEConfig.from_dict(config_dict))
    
    def initialize_score_node_popup(self):
        
        # initilize node score
        self.node_scores = nn.Parameter(
            torch.zeros((1, self.cfg.d_sae), dtype=self.dtype, device=self.device)
        )
        # if self.cfg.node_popup_init == "kaiming":
        nn.init.kaiming_uniform_(self.node_scores)
        # elif self.cfg.node_popup_init == "uniform":
        #     nn.init.uniform_(self.node_scores, a=-1, b=1)
        # elif self.cfg.node_popup_init == "normal":
        #     nn.init.normal_(self.node_scores, std=0.1)
        # elif self.cfg.node_popup_init == "constant":
        #     nn.init.constant_(self.node_scores, mode="fan_in", nonlinearity="relu")
            
        # prune rate
        self.prune_rate = 0.3
        
    @property
    def clamped_scores(self):
        return self.node_scores.abs()

    def encode(
        self, x: Float[torch.Tensor, "... d_in"]
    ) -> Float[torch.Tensor, "... d_sae"]:
        """
        Calcuate SAE features from inputs
        """
        feature_acts, _ = self.encode_with_hidden_pre(x)
        return feature_acts

    def encode_with_hidden_pre(
        self, x: Float[torch.Tensor, "... d_in"]
    ) -> tuple[Float[torch.Tensor, "... d_sae"], Float[torch.Tensor, "... d_sae"]]:

        # move x to correct dtype
        x = x.to(self.dtype)

        # handle hook z reshaping if needed.
        x = self.reshape_fn_in(x)  # type: ignore

        # apply b_dec_to_input if using that method.
        sae_in = self.hook_sae_input(x - (self.b_dec * self.cfg.apply_b_dec_to_input))

        # handle run time activation normalization if needed
        x = self.run_time_activation_norm_fn_in(x)

        # "... d_in, d_in d_sae -> ... d_sae",
        hidden_pre = self.hook_sae_acts_pre(sae_in @ self.W_enc + self.b_enc)
        hidden_pre_noised = hidden_pre + (
            torch.randn_like(hidden_pre) * self.cfg.noise_scale * self.training
        )
        if self.cfg.use_quadratic_activation:
            feature_acts = self.hook_sae_acts_post(self.activation_fn(hidden_pre_noised)*hidden_pre_noised)
        else:
            feature_acts = self.hook_sae_acts_post(self.activation_fn(hidden_pre_noised))
        
        if self.use_node_popup:
            feature_mask = GetSubnetNode.apply(self.clamped_scores, self.prune_rate)
            feature_acts = feature_acts * feature_mask

        return feature_acts, hidden_pre_noised

    def forward(
        self,
        x: Float[torch.Tensor, "... d_in"],
    ) -> Float[torch.Tensor, "... d_in"]:

        feature_acts, _ = self.encode_with_hidden_pre(x)
        sae_out = self.decode(feature_acts)

        return sae_out

    def training_forward_pass(
        self,
        sae_in: torch.Tensor, # (batch_size, d_i)
        current_l1_coefficient: float,
        dead_neuron_mask: Optional[torch.Tensor] = None,
    ) -> TrainStepOutput:
        
        if self.cfg.W_restart: # dead neuron part restarting
            if dead_neuron_mask is not None:
                with torch.no_grad():
                    # W_dec_new = torch.randn_like(self.W_dec[dead_neuron_mask])
                    W_enc_new = torch.randn_like(self.W_enc[:, dead_neuron_mask])
                    # nn.init.kaiming_uniform_(W_dec_new)
                    nn.init.kaiming_uniform_(W_enc_new)
                    # self.W_dec[dead_neuron_mask] = W_dec_new
                    self.W_enc[:, dead_neuron_mask] = W_enc_new

        # do a forward pass to get SAE out, but we also need the
        # hidden pre.
        feature_acts, _ = self.encode_with_hidden_pre(sae_in)
        if self.cfg.lp_norm == 0:
            # print("feature_acts: ", feature_acts)
            feature_acts = self.sample_activations(feature_acts)
            # print("feature_acts: ", feature_acts)
        if self.cfg.use_latent_norm:
            feature_acts = feature_acts / torch.norm(feature_acts, dim=-1, keepdim=True)
        sae_out = self.decode(feature_acts)

        # MSE LOSS
        per_item_mse_loss = self.mse_loss_fn(sae_out, sae_in)
        mse_loss = per_item_mse_loss.sum(dim=-1).mean()

        # GHOST GRADS
        if self.cfg.use_ghost_grads and self.training and dead_neuron_mask is not None:

            # first half of second forward pass
            _, hidden_pre = self.encode_with_hidden_pre(sae_in)
            ghost_grad_loss = self.calculate_ghost_grad_loss(
                x=sae_in,
                sae_out=sae_out,
                per_item_mse_loss=per_item_mse_loss,
                hidden_pre=hidden_pre,
                dead_neuron_mask=dead_neuron_mask,
            )
        else:
            ghost_grad_loss = 0.0
        
        # dec orthogonal
        if self.cfg.dec_orthogonal_coefficient > 0:
            # dec_orthogonal_loss = self.dec_orthogonal_coefficient * F.mse_loss(self.W_dec@self.W_dec.T, torch.eye(self.cfg.d_sae, dtype=self.dtype, device=self.device))
            dec_orthogonal_loss = self.dec_orthogonal_coefficient * torch.norm(self.W_dec@self.W_dec.T - torch.eye(self.cfg.d_sae, dtype=self.dtype, device=self.device), p="fro")
        else:
            dec_orthogonal_loss = 0.0

        # SPARSITY LOSS
        # either the W_dec norms are 1 and this won't do anything or they are not 1
        # and we're using their norm in the loss function.
        
        # ---- shape memo ----
        # feature_acts: (batch_size, d_sae)
        # W_dec: (d_sae, d_in)
        # ---------------------
        
        weighted_feature_acts = feature_acts * self.W_dec.norm(dim=1)
        
        if self.cfg.lp_norm == 0:
            sparsity = self.l0_reg(weighted_feature_acts)
        else:
            sparsity = weighted_feature_acts.norm(
                p=self.cfg.lp_norm, dim=-1
        )  # sum over the feature dimension

        l1_loss = (current_l1_coefficient * sparsity).mean()

        loss = mse_loss + l1_loss + ghost_grad_loss + dec_orthogonal_loss


        return TrainStepOutput(
            sae_in=sae_in,
            sae_out=sae_out,
            feature_acts=feature_acts,
            loss=loss,
            mse_loss=mse_loss.item(),
            l1_loss=l1_loss.item(),
            ghost_grad_loss=(
                ghost_grad_loss.item()
                if isinstance(ghost_grad_loss, torch.Tensor)
                else ghost_grad_loss
            ),
            dec_orthogonal_loss=(dec_orthogonal_loss.item() if isinstance(dec_orthogonal_loss, torch.Tensor) else dec_orthogonal_loss),
        )
    def cdf_qz(self, x):
        """Implements the CDF of the 'stretched' concrete distribution"""
        xn = (x - self.limit_a) / (self.limit_b - self.limit_a)
        logits = math.log(xn) - math.log(1 - xn)
        return F.sigmoid(logits * self.temperature - self.qz_loga).clamp(min=self.epsilon, max=1 - self.epsilon)
    
    def quantile_concrete(self, x):
        """Implements the quantile, aka inverse CDF, of the 'stretched' concrete distribution"""
        y = F.sigmoid((torch.log(x) - torch.log(1 - x) + self.qz_loga.to(x.device)) / self.temperature)
        return y * (self.limit_b - self.limit_a) + self.limit_a

    def get_eps(self, size):
        """Uniform random numbers for the concrete distribution"""
        eps = self.floatTensor(size).uniform_(self.epsilon, 1-self.epsilon)
        eps = Variable(eps)
        return eps

    def sample_z(self, batch_size, sample=True):
        """Sample the hard-concrete gates for training and use a deterministic value for testing"""
        if sample:
            eps = self.get_eps(self.floatTensor(batch_size, self.d_sae))
            z = self.quantile_concrete(eps)
            return F.hardtanh(z, min_val=0, max_val=1)
        else:  # mode
            pi = F.sigmoid(self.qz_loga).view(1, self.d_sae).expand(batch_size, self.d_sae)
            return F.hardtanh(pi * (self.limit_b - self.limit_a) + self.limit_a, min_val=0, max_val=1)

    def sample_activations(self, x):
        z = self.quantile_concrete(self.get_eps(self.floatTensor(self.d_sae)))
        mask = F.hardtanh(z, min_val=0, max_val=1).to(x.device)
        return mask.view(1, self.d_sae) * x

    def l0_reg(self, activations):
        """Expected L0 norm under the stochastic gates, takes into account and re-weights also a potential L2 penalty"""
        logpw_col = torch.sum(- (.5 * self.prior_prec * activations.pow(2)) - self.lamba, 0)
        logpw = torch.sum((1 - self.cdf_qz(0)).view(1, self.d_sae).to(activations.device) * logpw_col)
        return logpw
    
    # def forward(self, input):
    #     if self.local_rep or not self.training:
    #         z = self.sample_z(input.size(0), sample=self.training)
    #         xin = input.mul(z)
    #         output = xin.mm(self.weights)
    #     else:
    #         weights = self.sample_weights()
    #         output = input.mm(weights)
    #     if self.use_bias:
    #         output.add_(self.bias)
    #     return output

    def calculate_ghost_grad_loss(
        self,
        x: torch.Tensor,
        sae_out: torch.Tensor,
        per_item_mse_loss: torch.Tensor,
        hidden_pre: torch.Tensor,
        dead_neuron_mask: torch.Tensor,
    ) -> torch.Tensor:

        # 1.
        residual = x - sae_out
        l2_norm_residual = torch.norm(residual, dim=-1)

        # 2.
        # ghost grads use an exponentional activation function, ignoring whatever
        # the activation function is in the SAE. The forward pass uses the dead neurons only.
        feature_acts_dead_neurons_only = torch.exp(hidden_pre[:, dead_neuron_mask])
        ghost_out = feature_acts_dead_neurons_only @ self.W_dec[dead_neuron_mask, :]
        l2_norm_ghost_out = torch.norm(ghost_out, dim=-1)
        norm_scaling_factor = l2_norm_residual / (1e-6 + l2_norm_ghost_out * 2)
        ghost_out = ghost_out * norm_scaling_factor[:, None].detach()

        # 3. There is some fairly complex rescaling here to make sure that the loss
        # is comparable to the original loss. This is because the ghost grads are
        # only calculated for the dead neurons, so we need to rescale the loss to
        # make sure that the loss is comparable to the original loss.
        # There have been methodological improvements that are not implemented here yet
        # see here: https://www.lesswrong.com/posts/C5KAZQib3bzzpeyrg/full-post-progress-update-1-from-the-gdm-mech-interp-team#Improving_ghost_grads
        per_item_mse_loss_ghost_resid = self.mse_loss_fn(ghost_out, residual.detach())
        mse_rescaling_factor = (
            per_item_mse_loss / (per_item_mse_loss_ghost_resid + 1e-6)
        ).detach()
        per_item_mse_loss_ghost_resid = (
            mse_rescaling_factor * per_item_mse_loss_ghost_resid
        )

        return per_item_mse_loss_ghost_resid.mean()

    @torch.no_grad()
    def _get_mse_loss_fn(self) -> Any:

        def standard_mse_loss_fn(
            preds: torch.Tensor, target: torch.Tensor
        ) -> torch.Tensor:
            return torch.nn.functional.mse_loss(preds, target, reduction="none")

        def batch_norm_mse_loss_fn(
            preds: torch.Tensor, target: torch.Tensor
        ) -> torch.Tensor:
            target_centered = target - target.mean(dim=0, keepdim=True)
            normalization = target_centered.norm(dim=-1, keepdim=True)
            return torch.nn.functional.mse_loss(preds, target, reduction="none") / (
                normalization + 1e-6
            )

        if self.cfg.mse_loss_normalization == "dense_batch":
            return batch_norm_mse_loss_fn
        else:
            return standard_mse_loss_fn

    @classmethod
    def load_from_pretrained(
        cls,
        path: str,
        device: str = "cpu",
        dtype: str = "float32",
    ) -> "TrainingSAE":

        config_path = os.path.join(path, "cfg.json")
        weight_path = os.path.join(path, "sae_weights.safetensors")

        cfg_dict, state_dict, _ = load_pretrained_sae_lens_sae_components(
            config_path, weight_path, device, dtype
        )

        sae_cfg = TrainingSAEConfig.from_dict(cfg_dict)

        sae = cls(sae_cfg)
        sae.load_state_dict(state_dict)

        return sae

    def initialize_weights_complex(self):
        """ """

        if self.cfg.decoder_orthogonal_init:
            self.W_dec.data = nn.init.orthogonal_(self.W_dec.data.T).T

        elif self.cfg.decoder_heuristic_init:
            self.W_dec = nn.Parameter(
                torch.rand(
                    self.cfg.d_sae, self.cfg.d_in, dtype=self.dtype, device=self.device
                )
            )
            self.initialize_decoder_norm_constant_norm()

        elif self.cfg.normalize_sae_decoder:
            self.set_decoder_norm_to_unit_norm()

        # Then we intialize the encoder weights (either as the transpose of decoder or not)
        if self.cfg.init_encoder_as_decoder_transpose:
            self.W_enc.data = self.W_dec.data.T.clone().contiguous()
        else:
            self.W_enc = nn.Parameter(
                torch.nn.init.kaiming_uniform_(
                    torch.empty(
                        self.cfg.d_in,
                        self.cfg.d_sae,
                        dtype=self.dtype,
                        device=self.device,
                    )
                )
            )

        if self.cfg.normalize_sae_decoder:
            with torch.no_grad():
                # Anthropic normalize this to have unit columns
                self.set_decoder_norm_to_unit_norm()

    ## Initialization Methods
    @torch.no_grad()
    def initialize_b_dec_with_precalculated(self, origin: torch.Tensor):
        out = torch.tensor(origin, dtype=self.dtype, device=self.device)
        self.b_dec.data = out

    @torch.no_grad()
    def initialize_b_dec_with_mean(self, all_activations: torch.Tensor):
        previous_b_dec = self.b_dec.clone().cpu()
        out = all_activations.mean(dim=0)

        previous_distances = torch.norm(all_activations - previous_b_dec, dim=-1)
        distances = torch.norm(all_activations - out, dim=-1)

        print("Reinitializing b_dec with mean of activations")
        print(
            f"Previous distances: {previous_distances.median(0).values.mean().item()}"
        )
        print(f"New distances: {distances.median(0).values.mean().item()}")

        self.b_dec.data = out.to(self.dtype).to(self.device)

    ## Training Utils
    @torch.no_grad()
    def set_decoder_norm_to_unit_norm(self):
        self.W_dec.data /= torch.norm(self.W_dec.data, dim=1, keepdim=True)

    @torch.no_grad()
    def initialize_decoder_norm_constant_norm(self, norm: float = 0.1):
        """
        A heuristic proceedure inspired by:
        https://transformer-circuits.pub/2024/april-update/index.html#training-saes
        """
        # TODO: Parameterise this as a function of m and n

        # ensure W_dec norms at unit norm
        self.W_dec.data /= torch.norm(self.W_dec.data, dim=1, keepdim=True)
        self.W_dec.data *= norm  # will break tests but do this for now.

    @torch.no_grad()
    def remove_gradient_parallel_to_decoder_directions(self):
        """
        Update grads so that they remove the parallel component
            (d_sae, d_in) shape
        """
        assert self.W_dec.grad is not None  # keep pyright happy

        parallel_component = einops.einsum(
            self.W_dec.grad,
            self.W_dec.data,
            "d_sae d_in, d_sae d_in -> d_sae",
        )
        self.W_dec.grad -= einops.einsum(
            parallel_component,
            self.W_dec.data,
            "d_sae, d_sae d_in -> d_sae d_in",
        )
