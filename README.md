# Welcome to nsvq
`nsvq` (short for Noise Substitution Vector Quantization) is a tool for implementing and training vector quantization (VQ) in deep neural networks (DNNs), such as a VQ-VAE. It allows end-to-end training of DNNs that contain the non-differentiable VQ module, without any auxiliary losses and hyperparameter tunings. `nsvq` is implemented via PyTorch, and it requires `python >= 3.11` and `torch >= 2.0.0` .

![alt text](https://raw.githubusercontent.com/MHVali/Noise-Substitution-in-Vector-Quantization/main/nsvq_teaser.png)

The `nsvq` method is published as a research paper entitled [*"NSVQ: Noise Substitution in Vector Quantization for Machine Learning"*](https://ieeexplore.ieee.org/document/9696322) in the IEEE Access journal in 2022.

The `nsvq` package includes three different vector quantization (VQ) modules:
1. `from nsvq import NSVQ` optimizes the VQ codebook via NSVQ technique. The NSVQ module is the ordinary VQ method that uses only one codebook for quantization.

VQ variants that use multiple codebooks for vector quantization, i.e., Residual VQ and Product VQ:

2. `from nsvq import ResidualNSVQ` optimizes the Residual VQ codebooks via NSVQ technique.
3. `from nsvq import ProductNSVQ` optimizes the Product VQ codebooks via NSVQ technique.

> [!TIP]
> [DiVeQ](https://github.com/AaltoML/DiVeQ) is an improved successor of NSVQ method, which reduces the stocahsticity of nearest-neighbor assingments. To use DiVeQ technique, you can simply set `diveq=True` inside each of NSVQ, ResidualNSVQ, and ProductNSVQ modules.

# Installation

You can install `nsvq` through `pip` by running:

```bash
pip install nsvq
```

After installing `nsvq`, you can verify the installation and package details by running:

```bash
python -m pip show nsvq
```

# Usage Example

Before using `nsvq`, you have to install it using `pip install nsvq`.

Below you see a minimal example of how to import and use the `NSVQ` optimization method as a vector quantizer in a model.

```python
from nsvq import NSVQ
vector_quantizer = NSVQ(num_embeddings=512, # codebook size
                         embedding_dim=256   # embedding dimension
                         )

z = torch.randn(100, 256)

# when training
z_quantized, indices, perplexity = vector_quantizer(z) # shapes: (100, 256), (100, ), (1)

# at inference
z_q_hard, indices, perplexity = vector_quantizer.inference(z) # shapes: (100, 256), (100, ), (1)
```

- `vector_quantizer` is the vector quantization module that will be used for building the model.
- `num_embeddings` and `embedding_dim` are the codebook size and dimension of each codebook entry, respectively. In the following, you can find the list of all input parameters in different vector quantization modules incorporated in `nsvq` package.

- `z_quantized` is the differentiable quantized version of input z used when training
- `z_q_hard` is the non-differentiable (hard) quantized version of input z used at inference
- `indices` is the selected codebook indices
- `perplexity` is the codebook perplexity/usage (entropy of selected indices)

In the `example` directory of the [GitHub for nsvq package](https://github.com/MHVali/NSVQ_package), we provide a code example of how vector quantization modules in `nsvq` can be used in a vector quantized variational autoencoder (VQ-VAE). You can create the required environment to run the code by running:

```bash
cd example  #change directory to the example folder
conda create --name nsvq_example python=3.11
conda activate nsvq_example
pip install -r requirements.txt
```

Then, you can train the VQ-VAE model by running:

```bash
python train.py
```

# Tensor shape requirements
All quantization modules in `nsvq` package take input with the shape of (N, D), where N is the number of latent vectors and D is the embedding dimension. Usually for image data, the input of vector quantizer has the shape of (B, C, H, W), where B: batch size, C: input channels, H and W: Height and Width. In this case, to use `nsvq` quantization modules, you need permutation and reshaping of the latent z:

```python
# extract latent z from input image x
z = self.encoder(x)

# first do the permutation and reshaping
z_permuted = z.permute(0, 2, 3, 1) # Convert BCHW -> BHWC
z_shape = z_permuted.shape
z_flattened = z_permuted.view(-1, embedding_dim)  # Flatten the latents

# apply vector quantization using any module in nsvq package
z_quantized, indices, perplexity = self.vector_quantizer(z)

# apply permutation and reshaping back
z_quantized = (z_quantized.view(z_shape)).permute(0, 3, 1, 2) # Convert BHWC -> BCHW

# reconstruct the image from quantized latents
x_recon = self.decoder(z_quantized)
```

# List of Parameters
Here, we provide the list of parameters that are used as inputs for three different vector quantization modules included in `nsvq` package.

- `num_embeddings` (integer): Codebook size or the number of codewords in the codebook.
- `embedding_dim` (integer): Dimensionality of each codebook entry or codeword.
- `diveq` (bool): Whether to use DiVeQ technique for codebook learning. DiVeQ is the improved successor of NSVQ method.
- `diveq_noise_var` (float): Variance of the directional noise in DiVeQ. Ineffective if `diveq=False`.
- `replacement_iters` (integer): Number of training iterations to apply codebook replacement.
- `discard_threshold` (float): Threshold to discard the codebook entries that are used less than this threshold after *replacement_iters* iterations.
- `perturb_eps` (float): Adjusts perturbation/shift magnitude from used codewords for codebook replacement.
- `uniform_init` (bool): Whether to use uniform initialization. If False, codebook is initialized from a normal distribution.
- `verbose` (bool): Whether to print codebook replacement status, i.e., to print how many unused codewords are replaced.
- `codebook_reset` (bool): Whether to apply codebook replacement to avoid codebook collapse.
- `allow_warning` (bool): Whether to print the warnings. The warnings will warn if the user inserts unusual values for the parameters.
- `num_codebooks` (integer): Number of codebooks to be used for quantization in VQ variants of Residual VQ and Product VQ. All the codebooks will have the same size and dimensionality.

# Important Notes about Parameters

1. **Codebook Replacement:** Note that to prevent *codebook collapse*, we include a codebook replacement function (in cases where it is required) inside different quantization modules. Codebook replacement function is called after each `replacement_iters` training iterations, and it replaces the codewords which are used less than `discard_threshold` with perturbation of actively used codewords which are shifted by `perturb_eps` magnitude. If `verbose=True`, the status of how many unused codewords are replaced will be printed by the module. Note that the number of unused codewords should decrease over training, and it might take a while.

2. **Variants of Vector Quantization:** Residual VQ and Product VQ are two variants of vector quantization, which are included in the `nsvq` package. These variants utilize multiple codebooks for quantization, where `num_codebooks` determines the number of codebooks used in these VQ variants.

For further details about different vector quantization modules in the `nsvq` package and their corresponding parameters, please see the details provided in the Python codes in `src` directory of the `nsvq` package.

# Citation
If this package contributed to your work, please consider citing **NSVQ** as

```
@article{vali2022nsvq,
  title={{NSVQ}: Noise Substitution in Vector Quantization for Machine Learning},
  author={Vali, Mohammad Hassan and B{\"a}ckstr{\"o}m, Tom},
  journal={IEEE Access},
  volume={10},
  pages={13598--13610},
  year={2022}
}
```

and **DiVeQ** as

```
@InProceedings{vali2026diveq,
    title={{DiVeQ}: {D}ifferentiable Vector Quantization Using the Reparameterization Trick},
    author={Vali, Mohammad Hassan and B{\"a}ckstr{\"o}m, Tom and Solin, Arno},
    booktitle={International Conference on Learning Representations (ICLR)},
    year={2026}
}
```

# License
`nsvq` was developed by <span property="cc:attributionName">Mohammad Hassan Vali</span>, a former member of the <a href="https://www.aalto.fi/en/department-of-information-and-communications-engineering/speech-interaction-technology" target="_blank">Speech Interaction Technology research group from Aalto University</a> and is licensed under MIT license.