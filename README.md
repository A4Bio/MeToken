# METOKEN: UNIFORM MICRO-ENVIRONMENT TOKEN BOOSTS POST-TRANSLATIONAL MODIFICATION PREDICTION
## Introduction
The open-source code of paper "MeToken: Uniform Micro-environment Token Boosts Post-Translational Modification Prediction"
## Usage
### Interface
1. Place all the proteins that require inference into data_inference
2. Set inference position in proteins `inference_pos` in `args` The format is a nested list, with each sublist corresponding to the inference positions within each protein. Example: `'inference_pos':[[4,6],[35,56],[79,114]]` 
3. Due to the limitations of our model, **our model can only be used for predicting the type of PTM at confirmed PTM sites.** If you need to infer both the presence of PTM and its type simultaneously, you can set `with_null_ptm=1` in `args`, but we cannot guarantee the accuracy of the model in this mode.
4. Run quick_inference.ipynb <br>
In our directory, we perform inference on the protein with the Uniprot ID Q16613, and the PTMint database indicates phosphorylation modification at position 31. We set `inference_pos` as `[[31]]`, the result is `[{31: 16}]`, according to the numbering system in our paper, 16 represents phosphorylation modification.
### Test
1. Set test sets `path` in `args`, options are `'./data_test/large_scale_dataset/'`   `'./data_test/generalization/PTMint_dataset/'` `'./data_test/generalization/qPTM_dataset/'`
2. Run quick_test.ipynb