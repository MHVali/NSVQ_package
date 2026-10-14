import torch
import torch.nn as nn
from torch.distributions import normal
from typing import Tuple
import warnings

class ResidualNSVQ(nn.Module):
    """
    Residual_NSVQ: Differentiable Residual Vector Quantization (RVQ) module that allows
    end-to-end training of vector quantization-based models without any auxiliary losses
    or hyperparameter tunings. The module encompasses codebook replacement function
    which discards unused codebook entries during training to prevent codebook collapse.

    Args:
        - num_embeddings (int): Codebook size (No. of codewords).
        - embedding_dim (int): Dimensionality of embeddings.
        - num_codebooks (int): No. of codebooks involved in quantization.
        - diveq (bool): Whether to use DiVeQ technique for codebook learning. DiVeQ is
            the improved successor of NSVQ method.
        - diveq_noise_var (float): Variance of the directional noise in DiVeQ.
            Ineffective if diveq=False. Recommended diveq_noise_var < 1e-2.
        - replacement_iters (int): Replacement interval (number of training iterations
            to apply codebook replacement). Recommended 50 < replacement_iters < 300.
        - discard_threshold (float): Threshold to discard the codebook entries that are
            used less than this threshold after "replacement_iters" iterations.
            Recommended 0.01 < discard_threshold < 0.05. discard_threshold must be in
            the range of [0,1] such that discard_threshold=0.01 means to discard the
            codebook entries which are used less than 1 percent.
        - perturb_eps (float): Adjusts perturbation/shift magnitude from used codewords
            for codebook replacement.
        - uniform_init (bool): Whether to initialize codebook with uniform distribution.
            If False, the codebook is initialized from a normal distribution.
        - allow_warning (bool): Whether to print the warnings.
        - verbose (bool): Whether to print codebook replacement status.
        - codebook_reset (bool): Whether to apply codebook replacement to avoid
            codebook collapse.

    Returns:
        - z_q (torch.Tensor): Differentiable quantized input/latent. shape (N, D)
        - indices_all (torch.Tensor): Selected codebook indices for all
            codebooks. shape (num_codebooks, N)
        - perplexity_all (torch.Tensor): Perplexity (average codebook usage)
            for all codebooks. shape (num_codebooks, )
    """
    def __init__(
            self,
            num_embeddings: int,
            embedding_dim: int,
            num_codebooks: int,
            diveq: bool = False,
            diveq_noise_var: float = 0.001,
            replacement_iters: int = 100,
            discard_threshold: float = 0.01,
            perturb_eps: float = 1e-9,
            uniform_init: bool = True,
            allow_warning: bool = True,
            verbose: bool = True,
            codebook_reset: bool = True,
    ):
        super().__init__()

        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.num_codebooks = num_codebooks
        self.diveq = diveq
        self.diveq_noise_var = diveq_noise_var
        self.replacement_iters = replacement_iters
        self.discard_threshold = discard_threshold
        self.perturb_eps = perturb_eps
        self.uniform_init = uniform_init
        self.allow_warning = allow_warning
        self.verbose = verbose
        self.codebook_reset = codebook_reset

        self._check_constraints()

        # ---------------- User warnings ----------------
        if allow_warning:
            if self.diveq:
                if diveq_noise_var > 0.01:
                    warnings.warn(f"`diveq_noise_var` is set to {diveq_noise_var}, which is"
                                  f" quite large. Values > 0.01 may overshoot"
                                  f" nearest-neighbor mapping.", UserWarning)
            
            if codebook_reset:
                if replacement_iters < 50:
                    warnings.warn(f"`replacement_iters` is set to"
                                  f" {replacement_iters}, which is quite small."
                                  f" Values < 50 may cause too early and frequent"
                                  f" codebook replacements.", UserWarning)
                elif replacement_iters > 300:
                    warnings.warn(f"`replacement_iters` is set to"
                                  f" {replacement_iters}, which is quite large."
                                  f" Values > 300 may cause too late and sporadic"
                                  f" codebook replacements.", UserWarning)

                if discard_threshold > 0.05:
                    warnings.warn(f"`discard_threshold` is set to"
                                  f" {discard_threshold}, which is quite large."
                                  f" Values > 0.05 may discard a portion of suitable"
                                  f" but rarely used codewords.", UserWarning)

                if perturb_eps > 1e-6:
                    warnings.warn(f"`perturb_eps` is set to {perturb_eps},"
                                  f" which is quite large. Values > 1e-6 may cause big"
                                  f" perturbation/shift from used codewords.",
                                  UserWarning)

        # ---------------- Codebook initialization ----------------
        if uniform_init:
            codebook = torch.rand((self.num_codebooks, self.num_embeddings,
                                   self.embedding_dim)) * (1 / self.num_embeddings)
        else:
            codebook = torch.randn((self.num_codebooks, self.num_embeddings,
                                    self.embedding_dim)) * (1 / self.num_embeddings)

        self.codebook = torch.nn.Parameter(codebook, requires_grad=True)

        # ---------------- Tensors used for codebook replacement ----------------
        self.register_buffer("codebook_usage", torch.zeros(self.num_codebooks,
                                               self.num_embeddings, dtype=torch.int32))
        self.register_buffer("iter_counter", torch.zeros(1, dtype=torch.int32))

    # ---------------- Forward pass (Core API) ----------------
    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            - z (torch.Tensor): input/latent. shape (N, D)

        Returns:
            - z_q (torch.Tensor): Differentiable quantized input/latent. shape (N, D)
            - indices_all (torch.Tensor): Selected codebook indices for all
                codebooks. shape (num_codebooks, N)
            - perplexity_all (torch.Tensor): Perplexity (average codebook usage)
                for all codebooks. shape (num_codebooks, )
        """
        self._check_input(z)

        quantized_input_list = []
        remainder_list = []
        indices_list = []
        perplexity_list = []
        remainder_list.append(z)

        for i in range(self.num_codebooks):
            quantized_input, remainder, indices = self._hard_vq(remainder_list[i],
                                                                self.codebook[i])

            quantized_input_list.append(quantized_input)
            remainder_list.append(remainder)
            indices_list.append(indices)

            # Perplexity Computation
            perplexity = self._compute_perplexity(indices)
            perplexity_list.append(perplexity)

            # Track used indices for codebook replacement
            with torch.no_grad():
                self.codebook_usage[i, indices] += 1

        # Non-differentiable quantized input
        z_hard_quantized = sum(quantized_input_list[:])

        if self.diveq: # Use DiVeQ technique
            direction = z_hard_quantized - z
            random_vectors = (normal.Normal(0, self.diveq_noise_var).sample(z.shape).to(z.device) + direction)
            normalized = random_vectors / torch.linalg.norm(random_vectors, dim=1, keepdim=True).clamp_min(1e-12)
            error_magnitude = torch.linalg.norm(z_hard_quantized - z, dim=1, keepdim=True)
            vq_error = error_magnitude * normalized.detach()

        else: # Use NSVQ technique
            random_vectors = normal.Normal(0, 1).sample(z.shape).to(z.device)
            normalized = random_vectors / torch.linalg.norm(random_vectors, dim=1, keepdim=True).clamp_min(1e-12)
            error_magnitude = torch.linalg.norm(z_hard_quantized - z, dim=1, keepdim=True)
            vq_error = error_magnitude * normalized.detach()

        z_q = z + vq_error # Differentiable quantized input

        # Track used indices for codebook replacement
        if self.codebook_reset:
            with torch.no_grad():
                self.iter_counter += 1
                if self.iter_counter.item() % self.replacement_iters == 0:
                    self._replace_unused_entries() # Applies codebook replacement

        indices_all = torch.stack(indices_list, dim=0)
        perplexity_all = torch.stack(perplexity_list, dim=0)

        return z_q, indices_all, perplexity_all

    # ---------------- Quantization for Inference ----------------
    @torch.no_grad()
    def inference(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Deterministic hard quantization by mapping the input to the nearest codeword.
        Args:
            - z (torch.Tensor): input/latent. shape (N, D)

        Returns:
            - z_q_hard (torch.Tensor): Hard quantized input/latent. shape (N, D)
            - indices_all (torch.Tensor): Selected codebook indices for all
                codebooks. shape (num_codebooks, N)
            - perplexity_all (torch.Tensor): Perplexity (average codebook usage)
                for all codebooks. shape (num_codebooks, )
        """
        self._check_input(z)

        quantized_input_list = []
        remainder_list = []
        indices_list = []
        perplexity_list = []
        remainder_list.append(z)

        for i in range(self.num_codebooks):
            quantized_input, remainder, indices = self._hard_vq(remainder_list[i],
                                                                self.codebook[i])

            quantized_input_list.append(quantized_input)
            remainder_list.append(remainder)
            indices_list.append(indices)

            # Perplexity Computation
            perplexity = self._compute_perplexity(indices)
            perplexity_list.append(perplexity)

        z_q_hard = sum(quantized_input_list[:])

        indices_all = torch.stack(indices_list, dim=0)
        perplexity_all = torch.stack(perplexity_list, dim=0)

        return z_q_hard, indices_all, perplexity_all

    # ---------------- Utility functions ----------------
    def _check_constraints(self,) -> None:
        if self.diveq:
            if self.diveq_noise_var <= 0.0:
                raise ValueError("`diveq_noise_var` must be a positive float value. To have more"
                                 " precise nearest-neighbor assignments, it is recommended"
                                 " that diveq_noise_var < 1e-2.")

        if self.codebook_reset:
            if (self.replacement_iters <= 0) or (type(self.replacement_iters) is not int):
                raise ValueError("`replacement_iters` must be a positive integer value."
                                 " It is recommended that 50 < replacement_iters < 300.")
            if (self.discard_threshold < 0.0) or (self.discard_threshold > 1.0):
                raise ValueError("`discard_threshold` must be in the range of [0,1]. It is"
                                 " recommended that 0.01 < discard_threshold < 0.05,"
                                 " such that discard_threshold=0.01 means to discard the"
                                 " codebook entries which are used less than 1 percent.")

    def _check_input(self, z: torch.Tensor) -> None:
        if z.ndim != 2:
            raise ValueError("Residual_NSVQ input must have the shape of (N, D), where"
                             " N is the No. of input samples,and D is the"
                             " embedding dimensionality.")
        if z.size(1) != self.embedding_dim:
            raise ValueError(f"Residual_NSVQ input.shape[1] must match the embedding"
                             f" dimensionality that is {self.embedding_dim}.")

    @staticmethod
    def _hard_vq(x: torch.Tensor, codebook: torch.Tensor) \
                                    -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        distances = (torch.sum(x.pow(2), dim=1, keepdim=True)
                     + torch.sum(codebook.pow(2), dim=1)
                     - 2 * torch.matmul(x, codebook.t()))
        indices = torch.argmin(distances, dim=1)
        x_quantized = codebook[indices]
        remainder = x - x_quantized
        return x_quantized, remainder, indices

    def _compute_perplexity(self, indices: torch.Tensor) -> float:
        encodings = torch.zeros(indices.shape[0], self.num_embeddings,
                                device=indices.device)
        encodings.scatter_(1, indices.unsqueeze(1), 1)
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        return perplexity

    def _replace_unused_entries(self) -> None:
        with torch.no_grad():

            for i in range(self.num_codebooks):

                usage_ratio = self.codebook_usage[i] / self.replacement_iters
                unused_indices = torch.where(usage_ratio < self.discard_threshold)[0]
                used_indices = torch.where(usage_ratio >= self.discard_threshold)[0]

                if unused_indices.numel() == 0 or used_indices.numel() == 0:
                    self.codebook_usage[i].zero_()
                    continue

                unused_count = unused_indices.numel()
                used_probs = self.codebook_usage[i, used_indices] / torch.sum(
                                                self.codebook_usage[i, used_indices])
                randomly_sampled_indices = used_probs.multinomial(
                                            num_samples=unused_count, replacement=True)
                sampled_indices = used_indices[randomly_sampled_indices]
                used_codebooks = self.codebook[i, sampled_indices].clone()

                self.codebook.data[i, unused_indices] = (used_codebooks + self.perturb_eps
                                            * torch.randn_like(used_codebooks)).clone()
                self.codebook_usage[i].zero_()

                if self.verbose:
                    print("\n***** Replaced " + str(unused_count) + f" codewords of "
                                                                    f"CB{i + 1} *****")

    def extra_repr(self) -> str:
        return (
            f"num_embeddings={self.num_embeddings}, "
            f"embedding_dim={self.embedding_dim}, "
            f"num_codebooks={self.num_codebooks}, "
            f"diveq={self.diveq}, "
            f"diveq_noise_var={self.diveq_noise_var}, "
            f"replacement_iters={self.replacement_iters}, "
            f"discard_threshold={self.discard_threshold}, "
            f"perturb_eps={self.perturb_eps}, "
            f"uniform_init={self.uniform_init}, "
            f"allow_warning={self.allow_warning}, "
            f"verbose={self.verbose}, "
            f"codebook_reset={self.codebook_reset}"
        )