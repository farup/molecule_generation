from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem import Descriptors
from rdkit.Chem.Scaffolds import MurckoScaffold

from rdkit import DataStructs
from rdkit.Chem import AllChem
from rdkit.Chem import RDConfig
import numpy as np
import sys
import os

sys.path.append(os.path.join(RDConfig.RDContribDir, "SA_Score"))
import sascorer

from RL.config import config



mfpgen = rdFingerprintGenerator.GetMorganGenerator(radius=config.FINGERPRINT_RADIUS,fpSize=config.FINGERPRINT_LENGTH)


def get_graph_embedding(smiles): 
    
    mol =Chem.MolFromSmiles("O=C(CN)N")
    Chem.rdmolops.GetAdjacencyMatrix(mol)
    


def get_fingerprint(smiles, fingerprint_length, fingerprint_radius):
    """Get Morgan Fingerprint of a specific SMILES string.

  Args:
    smiles: String. The SMILES string of the molecule.
    hparams: tf.contrib.training.HParams. Hyper parameters.

  Returns:
    np.array. shape = [hparams.fingerprint_length]. The Morgan fingerprint.
  """
    if smiles is None:
        return np.zeros((config.FINGERPRINT_LENGTH,))
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        return np.zeros((config.FINGERPRINT_LENGTH))
    
    fingerprint = mfpgen.GetFingerprint(molecule)
    # fingerprint = AllChem.GetMorganFingerprintAsBitVect(
    #     molecule, config.FINGERPRINT_RADIUS, config.FINGERPRINT_LENGTH
    # )
    arr = np.zeros((1,))
    # ConvertToNumpyArray takes ~ 0.19 ms, while
    # np.asarray takes ~ 4.69 ms
    DataStructs.ConvertToNumpyArray(fingerprint, arr)
    return arr


def atom_valences(atom_types):
    """Creates a list of valences corresponding to atom_types.

  Note that this is not a count of valence electrons, but a count of the
  maximum number of bonds each element will make. For example, passing
  atom_types ['C', 'H', 'O'] will return [4, 1, 2].

  Args:
    atom_types: List of string atom types, e.g. ['C', 'H', 'O'].

  Returns:
    List of integer atom valences.
  """
    periodic_table = Chem.GetPeriodicTable()
    return [
        max(list(periodic_table.GetValenceList(atom_type))) for atom_type in atom_types
    ]