# PS-Eval: Evaluation Framework for Sparse Autoencoders (SAEs)

paper link : https://arxiv.org/abs/2501.06254  

![image](https://github.com/gouki510/PS-Eval/blob/main/fig1_v5.png)
## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/gouki510/PS-Eval.git
   cd PS-Eval
   ```

2. Docker build
    ```
    cd docker
    docker build -t ps_eval .
    docker run -it --gpus all -v $(pwd):/workspace ps_eval
    ```

## Dataset

The evaluation is built on the **Word-in-Context (WiC)** dataset. The modified dataset, PS-Eval, provides:
- **Poly-contexts**: Sentences where the target word has different meanings.
- **Mono-contexts**: Sentences where the target word has the same meaning.

Download the dataset from [Hugging Face](https://huggingface.co/datasets/gouki510/Wic_data_for_SAE-Eval).

## Usage

### 1. Training SAEs
Train an SAE with the provided configurations:
```bash
python train.py 
    --model_name_or_path gpt2-small 
    --d_in 768 
    --expansion_factor 32 
    --hook_name blocks.9.hook_resid_pre 
    --hook_layer 4 
    --batch_size 1024 
    --dataset_path pasinit/xlwic 
    --l1_coefficient 0.05 
    --datadir xlwic_en_de xlwic_en_bg xlwic_en_da xlwic_en_et xlwic_en_fa xlwic_en_fr xlwic_en_hr xlwic_en_it xlwic_en_ja xlwic_en_nl xlwic_en_ko xlwic_en_zh \
    --use_ghost_grads --output_dir your_output_dir --total_training_steps 20000
```

### 2. Evaluation
Evaluate the trained SAE using PS-Eval:
```bash
python wic_eval.py --sae_path your_sae_checkpoint_path
```

### 3. Notebook evaluation
You can also use the notebook to evaluate the SAE.
`GPT2_PS_Eval.ipynb`


## Citation

If you use this code or framework in your research, please cite:
```
@article{minegishi2024ps_eval,
  title={Rethinking Evaluation of Sparse Autoencoders Through the Representation of Polysemous Words},
  author={Gouki Minegishi, Hiroki Furuta, Yusuke Iwasawa, Yutaka Matsuo},
  journal={arXiv preprint arXiv:...},
  year={2024}
}
```
