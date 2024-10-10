
from rdkit import Chem
import torch
from config import config
import numpy as np
import sys
from RL.environment.environment import QEDRewardMolecule, RLMol


from RL.config import config as rl_config
from RL.utils import utils

def build_molecules(batch, atom_pred_matrix, rl=False, agent=None): 
    results = np.array([0, 0, 0]) # valid, invalid, reconstruction error
    qeds = []
    for i in range(len(atom_pred_matrix)): 
        new_results = graph_represenatation_to_molecule(atom_pred_matrix[i], batch[i], rl, agent)
        results += new_results[0]
        qeds.append(new_results[1])
    
    return results

def graph_represenatation_to_molecule(atom_pred_matrix, edge_preds, rl, agent):

    mol = Chem.RWMol()
    node_to_idx = {}
    for i in range(len(atom_pred_matrix)):
        pred = torch.argmax(atom_pred_matrix[i])
        if pred > 4: 
            continue
        atomic_number = config.ATOMIC_NUMBER[pred]
        a = Chem.Atom(atomic_number)
        molIdk = mol.AddAtom(a) # returns index of added atom in molecule
        node_to_idx[i] = molIdk # maps current index "i" to the atom index "molIdk"

    # num_nodes = len(atom_types)
    # adj_matrix = triu_to_dense(edge_preds, num_nodes)

    for iy, row in enumerate(torch.argmax(edge_preds, dim=-1)):
        for ix, bond in enumerate(row):  
            if ix <= iy: 
                continue

            if bond == 0: 
                continue
            else:
                if bond == 1:
                    bond_type = Chem.rdchem.BondType.SINGLE
                elif bond == 2:
                    bond_type = Chem.rdchem.BondType.DOUBLE
                elif bond == 3:
                    bond_type = Chem.rdchem.BondType.TRIPLE
                elif bond == 4:
                    bond_type = Chem.rdchem.BondType.AROMATIC
                try: 
                    mol.AddBond(node_to_idx[iy], node_to_idx[ix], bond_type)
                except KeyError as e:
                    print(f"Edge predicted between missing atom prediction at postion {ix, iy}", e) 
                    
                    return np.array([0,0,1])
                except Exception as e: 
                    print("An error occurred while constructing molecule from prediction", e)
                    
                    return np.array([0,0,1])

    # Convert RWMol to mol and Smiles
    mol = mol.GetMol()
    smiles = Chem.MolToSmiles(mol)

    if rl:
        environment = RLMol(
        discount_factor=rl_config.DISCOUNT_FACTOR,
        atom_types=set(rl_config.ATOM_TYPES),
        init_mol=smiles,
        allow_removal=rl_config.ALLOW_REMOVAL,
        allow_no_modification=rl_config.ALLOW_NO_MODFICATION,
        allow_bonds_between_rings=rl_config.ALLOW_BOND_BETWEEN_RINGS,
        allowed_ring_sizes=set(rl_config.ALLOWED_RING_SIZES),
        max_steps=rl_config.MAX_STEPS,
        )

        mol_res = []
        for steps in range(rl_config.MAX_STEPS):
            steps_left = rl_config.MAX_STEPS - environment.num_steps_taken

            # Compute a list of all possible valid actions. (Here valid_actions stores the states after taking the possible actions)
            valid_actions = list(environment.get_valid_actions())

            # Append each valid action to steps_left and store in observations.
            observations = np.vstack(
                [
                    np.append(
                        utils.get_fingerprint(
                            act, rl_config.FINGERPRINT_LENGTH, rl_config.FINGERPRINT_RADIUS
                            ),steps_left,) 
                        for act in valid_actions
                ]
            )  # (num_actions, fingerprint_length)

            observations_tensor = torch.Tensor(observations)
      
            a = agent.get_action(observations_tensor, 0)

            action = valid_actions[a]
        
            result = environment.step(action)
            next_state, reward, done = result
            try: 
                mol = Chem.MolFromSmiles(next_state)
                Chem.SanitizeMol(mol)
                qed = Chem.QED.qed(mol)
            except: 
                qed = 0
            
            mol_res.append((next_state, qed))
        
        mols, qeds = zip(*mol_res)
        index = np.argmax(np.array(qeds))
        best_mol = mols[index]

        try:
            mol = Chem.MolFromSmiles(best_mol)
            Chem.SanitizeMol(mol)
            return np.array([1,0,0]), qeds[index]

        except:
            return np.array([0,1,0]), qeds[index]
        
    else:
        try:
            Chem.SanitizeMol(mol)
            qed = Chem.QED.qed(mol) 
            return np.array([1,0,0]), qed
        except: 
            return np.array([0,1,0]), 0
    
         
    # # TODO: Visualize and save (use deepchem smiles_to_image)
    # return np.array([1,0,0]), qed


def graph_represenatation_to_molecule1(atom_pred_matrix, edge_preds):

    mol = Chem.RWMol()

    node_to_idx = {}
    for i in range(len(atom_pred_matrix)):
        pred = torch.argmax(atom_pred_matrix[i])
        if pred > 4: 
            continue
        atomic_number = config.ATOMIC_NUMBER[pred]
        a = Chem.Atom(atomic_number)
        molIdk = mol.AddAtom(a) # returns index of added atom in molecule
        node_to_idx[i] = molIdk # maps current index "i" to the atom index "molIdk"

    
    # num_nodes = len(atom_types)
    # adj_matrix = triu_to_dense(edge_preds, num_nodes)

    for ix, row in enumerate(torch.argmax(edge_preds, dim=-1)):
        for iy, bond in enumerate(row):  
            if iy <= ix: 
                continue

            if bond == 0: 
                continue
            else:
                if bond == 1:
                    bond_type = Chem.rdchem.BondType.SINGLE
                elif bond == 2:
                    bond_type = Chem.rdchem.BondType.DOUBLE
                elif bond == 3:
                    bond_type = Chem.rdchem.BondType.TRIPLE
                elif bond == 4:
                    bond_type = Chem.rdchem.BondType.AROMATIC
                mol.AddBond(node_to_idx[ix], node_to_idx[iy], bond_type)
        
    # Convert RWMol to mol and Smiles
    mol = mol.GetMol()
    smiles = Chem.MolToSmiles(mol)

    try:
        Chem.SanitizeMol(mol)
    except:
        smiles = None

    # TODO: Visualize and save (use deepchem smiles_to_image)
    return smiles, mol