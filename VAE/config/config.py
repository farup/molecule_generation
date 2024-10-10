import torch

ATOMIC_SYMBOL = ["H", "C", "N", "O", "F"]
ATOMIC_NUMBER = [1, 6, 7, 8, 9]

NUM_NODE_FEATURES = 11
NUM_ATOM_TYPES = 5
NUM_EDGES_TYPE = 4

MAX_GRAPH_SIZE = 20

ENC_DIM = 100
LATENT_DIM = 128

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ROOT = r"C:\Users\tnf\Documents\NTNU\2024\WaterlooInternship\final_project\data\qm9"

# Training

TRAIN_TEST_SPLIT = 0.9
BATCH_SIZE = 4
LEARNING_RATE = 1e-4
EPOCH = 10
