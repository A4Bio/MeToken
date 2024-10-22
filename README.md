# MeToken: Uniform Micro-Environment Token Boosts Post-Translational Modification Prediction

This repository contains the open-source implementation of the paper "MeToken: Uniform Micro-Environment Token Boosts Post-Translational Modification Prediction." The MeToken model leverages both sequence and structural information to accurately predict post-translational modification (PTM) types at specific sites on proteins. By tokenizing the micro-environment of each amino acid, MeToken captures the complex factors influencing PTMs, addressing limitations of sequence-only models and improving prediction performance, especially for rare PTM types.

## Table of Contents

<!-- - [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
  - [Inference](#inference)
  - [Testing](#testing)
- [License](#license)
- [Citations](#citations)  -->

## Introduction
Post-translational modifications (PTMs) are crucial for regulating protein function and interactions. Accurately predicting PTM sites and their types helps understand biological processes and disease mechanisms. Traditional computational approaches mainly focus on sequence motifs for PTM prediction, often neglecting the role of protein structure. 

**MeToken** addresses these limitations by integrating both sequence and structural information into unified tokens that represent the micro-environment of each amino acid. The model leverages a large-scale sequence-structure PTM dataset and uses uniform sub-codebooks to handle the long-tail distribution of PTM types, ensuring robust performance even for rare PTMs.

## Features
- **🚀 Integration of Sequence and Structure:** MeToken tokenizes the local micro-environment of amino acids, combining sequence motifs and 3D structural information.
- **⚡ Support for Multiple PTM Types:** The model is designed to predict a wide range of PTM types, including rare modifications.

## Installation
1. **Clone the repository:**
   ```bash
   git clone https://github.com/your_username/MeToken.git
   cd MeToken
   ```

2. **Install dependencies:**
    ```bash
    conda env create -f environment.yml
    conda activate metoken
    ```

3. **Download the pretrained model:**

    We provide a pretrained model for MeToken. [Download it here]() and place it in the pretrained_models directory.

## Usage

### Inference

To perform PTM prediction on a single PDB file, follow these steps:

1. **Run the inference script:**

```bash
python inference.py --pdb_file_path examples/Q16613.pdb --predict_indices 31 79 114
```

- `--pdb_file_path`: Path to the input PDB file (e.g., `examples/Q16613.pdb`).
- `--predict_indices`: A list of residue indices for which PTM prediction should be made.

2. **Optional arguments:**

- `--checkpoint_path`: Specify the path to the model checkpoint (default is `pretrained_model/checkpoint.ckpt`).
- `--output_json_path`: Path to save prediction results in JSON format (default is output/predict.json).
- `--output_hdf5_path`: Path to save prediction results in HDF5 format (default is output/predict.hdf5).

3. **Example Output**: The script will print predictions for the specified positions:

```python
PTM type at position 31 is phosphorylation.
PTM type at position 79 is acetylation.
PTM type at position 114 is ubiquitination.
```

### Testing

You can evaluate the model using predefined test datasets.

1. Set the test dataset path in args within `quick_test.ipynb`. Available test sets:

- ./data_test/large_scale_dataset/
- ./data_test/generalization/PTMint_dataset/
- ./data_test/generalization/qPTM_dataset/

2. Run the test notebook:

```bash
jupyter notebook quick_test.ipynb
```

This will provide performance metrics and model evaluation results.

## References

For a complete description of the method, see:

```text
TBD
```

## Contact

Please submit any bug reports, feature requests, or general usage feedback as a github issue or discussion.

- Cheng Tan (tancheng@westlake.edu.cn)
- Zhenxiao Cao (alancao@stu.xjtu.edu.cn)

## License

This project is licensed under the MIT License. See the `LICENSE` file for more details.