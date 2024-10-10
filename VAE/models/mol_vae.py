


import torch.nn as nn 
import torch
import torch.nn.functional as F
import os

import torch_geometric.nn as pyg_nn
from torch_geometric.datasets import QM9
from torch_geometric.loader import DataLoader
from torch_geometric.nn import global_max_pool, Set2Set
import sys 

from tqdm import tqdm
# sys.path.append(r"C:\Users\tnf\Documents\NTNU\2024\WaterlooInternship\final_project")

from RL.agent import Agent
from RL.config import config as rl_config
from config import config

agent = Agent(rl_config.FINGERPRINT_LENGTH +1, 1, config.DEVICE)
from VAE.utils.utils_build_molecule import build_molecules

sys.path.append(r"C:\Users\tnf\Documents\NTNU\2024\WaterlooInternship\final_project")


from utils.utils_loss_fn import vae_loss


class Mol_VAE(nn.Module): 
    def __init__(self, input_dim, enc_dim, latent_dim, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        output_atoms = int(config.MAX_GRAPH_SIZE*(config.NUM_ATOM_TYPES +1))
        output_edges = int((config.MAX_GRAPH_SIZE * (config.MAX_GRAPH_SIZE-1) /2) * config.NUM_EDGES_TYPE)

        # Encoder 
        self.input_dim = input_dim 
        self.enc_dim = enc_dim
        self.latent_dim = latent_dim
        self.enc_tConv1 = pyg_nn.TransformerConv(input_dim, enc_dim, edge_dim=config.NUM_EDGES_TYPE)
        self.enc_b1 = pyg_nn.BatchNorm(enc_dim)

        self.pool = Set2Set(enc_dim, 4)
        self.max_pool = global_max_pool

        # mean and log variance
        self.enc_mu = nn.Linear(enc_dim*2, latent_dim)
        self.enc_logvar = nn.Linear(enc_dim*2, latent_dim)

        # Decoder 
        self.dec_lin_atoms = nn.Linear(latent_dim, output_atoms)
        self.dec_lin_edges = nn.Linear(latent_dim, output_edges)

        # Agent
        self.agent = Agent(rl_config.FINGERPRINT_LENGTH + 1, 1, config.DEVICE)
        self.agent.load_model_parameters(rl_config.SAVE_MODEL_PATH, "dqn_it_30000.pt")

    def encoder(self, x, edge_index, edge_attr, batch_index): 
        x = self.enc_tConv1(x,  edge_index, edge_attr)
        x = self.enc_b1(x)
        x = F.relu(x)
        x = self.pool(x, batch_index)

        mu, logvar = self.enc_mu(x), self.enc_logvar(x)
        return mu, logvar
    
    def decoder_advanced(self, z, batch_index): 

        atom_pred_list = []
        edge_pred_list = []

        for graph_id in torch.unique(batch_index):
            graph_z = z[graph_id]
            atom_preds, edge_preds, = self.decoder(graph_z)
            atom_pred_list.append(atom_preds)
            edge_pred_list.append(edge_preds)

        return atom_pred_list, edge_pred_list


    def decoder(self, z):
        atoms = self.dec_lin_atoms(z)
        edges = self.dec_lin_edges(z)
        return atoms, edges
        
    def reparameterization(self, mu, logvar): 
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(mu)
        return eps.mul(std).add_(mu)
        
    def forward(self, batch):
        # DataBatch(x=[16, 11], edge_index=[2, 24], edge_attr=[24, 4], y=[4, 19], pos=[16, 3], idx=[4], name=[4], z=[16], batch=[16], ptr=[5])
        x, edge_index, edge_attr, batch_index = batch.x, batch.edge_index, batch.edge_attr, batch.batch
        mu, logvar = self.encoder(x, edge_index, edge_attr, batch_index)
        z = self.reparameterization(mu, logvar)
        atom_preds, edge_preds = self.decoder(z)
        #atom_pred_lists, edge_pred_lists = self.decoder_advanced(z,batch_index)
        return atom_preds, edge_preds, z, mu, logvar
    

    def sample_mols(self, num=100, rl=False): 

        for _ in tqdm(range(num)):

            z = torch.randn(1, self.latent_dim).to(config.DEVICE)
        
            atom_preds, edge_preds = self.decoder(z)


            edge_pred_matrix = edge_preds.unsqueeze(1).reshape(-1, int((config.MAX_GRAPH_SIZE * (config.MAX_GRAPH_SIZE - 1))/ 2), config.NUM_EDGES_TYPE)

            atom_pred_matrix = atom_preds.unsqueeze(1).reshape(-1, config.MAX_GRAPH_SIZE, config.NUM_ATOM_TYPES+1)
                    
            batch = []
            triu_indices = torch.triu_indices(config.MAX_GRAPH_SIZE, config.MAX_GRAPH_SIZE, offset=1)

            for graph in edge_preds:
                placeholder_matrix = torch.zeros((config.MAX_GRAPH_SIZE, config.MAX_GRAPH_SIZE, config.NUM_EDGES_TYPE)).to(config.DEVICE)

                edge_pred_matrix = graph.unsqueeze(1).reshape(-1, int((config.MAX_GRAPH_SIZE * (config.MAX_GRAPH_SIZE - 1))/ 2), config.NUM_EDGES_TYPE)
                
                placeholder_matrix[triu_indices[0], triu_indices[1]] = edge_pred_matrix
                
                batch.append(placeholder_matrix)

            batch = torch.stack(batch, dim=0)
            out = build_molecules(batch, atom_pred_matrix, rl, self.agent) / num
            
        return out
    



if __name__ == "__main__": 

    data = QM9(root=config.ROOT)

    train_loader = DataLoader(data, batch_size=2)
    batch = next(iter(train_loader))

    mol_vae = Mol_VAE(input_dim=config.NUM_NODE_FEATURES, enc_dim=config.ENC_DIM, latent_dim=config.LATENT_DIM)
    atom_preds, edge_preds, z, mu, logvar = mol_vae(batch)

    loss = vae_loss(atom_preds, edge_preds, z, mu, logvar, batch)


