import re
import torch
import argparse
from Bio import PDB
from omegaconf import OmegaConf
from src.metoken_model import MeToken_Model
from src.datasets.featurizer import featurize


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="PTM prediction from PDB files.")
    parser.add_argument(
        '--pdb_file_path', type=str, required=True,
        help="Path to the input PDB file."
    )
    parser.add_argument(
        '--checkpoint_path', type=str, default='pretrained_model/checkpoint.ckpt',
        help="Path to the model checkpoint file."
    )
    parser.add_argument(
        '--output_json_path', type=str, default='output/predict.json',
        help="Path to save the prediction results in JSON format. Default is 'output/predict.json'."
    )
    parser.add_argument(
        '--output_hdf5_path', type=str, default='output/predict.hdf5',
        help="Path to save the prediction results in HDF5 format. Default is 'output/predict.hdf5'."
    )
    parser.add_argument(
        '--predict_indices', type=int, nargs='+', default=[31],
        help="List of residue indices for PTM prediction."
    )
    return parser.parse_args()


def extract_pdb_id(file_path):
    """Extract PDB ID from the file path."""
    match = re.search(r'([^/]+)(?=\.pdb)', file_path)
    if match:
        return match.group(0)
    else:
        raise ValueError(f"Could not extract PDB ID from file path: {file_path}")


def get_seq_str(pdb_file_path, chain_id='A'):
    parser = PDB.PDBParser(QUIET=True)

    if pdb_file_path.endswith('.pdb'):
        pdb_id = extract_pdb_id(pdb_file_path)
        structure = parser.get_structure(pdb_id, pdb_file_path)
        
        seq = ''
        coords_chain_A = {'N_chain_A': [], 'C_chain_A': [], 'CA_chain_A': [], 'O_chain_A': []}
        
        for model in structure:
            for chain in model:
                if chain.id == chain_id:
                    for residue in chain:
                        if PDB.is_aa(residue):
                            seq += PDB.Polypeptide.three_to_one(residue.resname)
                            for atom in residue:
                                if atom.id == 'N':
                                    coords_chain_A['N_chain_A'].append(atom.coord.tolist())
                                elif atom.id == 'C':
                                    coords_chain_A['C_chain_A'].append(atom.coord.tolist())
                                elif atom.id == 'CA':
                                    coords_chain_A['CA_chain_A'].append(atom.coord.tolist())
                                elif atom.id == 'O':
                                    coords_chain_A['O_chain_A'].append(atom.coord.tolist())        
        protein_data = {
            "id": pdb_id,
            "seq": seq,
            "coords_chain_A": coords_chain_A
        }
    else:
        raise "The PDB file path is invalid."
    return protein_data


def apply_ptm_indices(protein_data, ptm_indices):
    seq_length = len(protein_data["seq"])
    
    if -1 in ptm_indices:
        ptm = [1] * seq_length
    else:
        ptm = [0] * seq_length
        for index in ptm_indices:
            if 0 <= index < seq_length:
                ptm[index] = 1

    protein_data["ptm"] = ptm
    return protein_data


def main():
    args = parse_arguments()
    try:
        protein_data = get_seq_str(args.pdb_file_path)
        protein_data = apply_ptm_indices(protein_data, args.predict_indices)

        checkpoint = torch.load(args.checkpoint_path)
        params = OmegaConf.load('configs/MeToken.yaml')
        model = MeToken_Model(params)
        model.load_state_dict(checkpoint)

        data = featurize([protein_data])
        result = model(data)
        preds = result['log_probs'].argmax(dim=-1).cpu()

        for pos in args.predict_indices:
            print(f'PTM type at the position {pos} is {preds[pos]}.\n')
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}")


if __name__ == '__main__':
    main()