import torch.nn as nn 
import torch 

from tqdm import tqdm

from torch_geometric.loader import DataLoader
from torch_geometric.datasets import QM9
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter

import sys

sys.path.append(r"C:\Users\tnf\Documents\NTNU\2024\WaterlooInternship\final_project")
from config import config
from VAE.utils.utils_loss_fn import vae_loss
from VAE.models.mol_vae import Mol_VAE

from VAE.utils.utils_build_molecule import build_molecules, graph_represenatation_to_molecule


data = QM9(root=config.ROOT)

split = int(len(data) * config.TRAIN_TEST_SPLIT)

train_loader = DataLoader(data[:split], batch_size=config.BATCH_SIZE)
eval_loader = DataLoader(data[split:], batch_size=config.BATCH_SIZE)

mol_vae = Mol_VAE(input_dim=config.NUM_NODE_FEATURES, enc_dim=config.ENC_DIM, latent_dim=config.LATENT_DIM).to(config.DEVICE)
optimizer = Adam(mol_vae.parameters(), lr=config.LEARNING_RATE)


def train(): 
    writer = SummaryWriter()
    mol_vae.train()
    for epoch in tqdm(range(config.EPOCH), desc="VAE"): 
        pbar = tqdm(train_loader, desc=f"TRAIN EPOCH {epoch}")
       
        for idx, batch in enumerate(pbar): 
            optimizer.zero_grad()
            atom_preds, edge_preds, z, mu, logvar = mol_vae(batch.to(config.DEVICE))
            atom_loss, edge_loss, atom_edge_loss, kl_loss2, kl_loss, out = vae_loss(atom_preds, edge_preds, z, mu, logvar, batch)
            loss = atom_loss + edge_loss + atom_edge_loss + kl_loss2
            loss.backward()
            optimizer.step()
            pbar.set_postfix({"atom_loss" : atom_loss, "edge_loss": edge_loss, "atom_edge_loss": atom_edge_loss})
            writer.add_scalar("Loss/train", loss, idx)
            writer.add_scalar("atom_loss/train", atom_loss, idx)
            writer.add_scalar("edge_loss/train", edge_loss, idx)
            writer.add_scalar("atom_edge_loss/train", atom_edge_loss, idx)
            writer.add_scalar("kl_loss2/train", kl_loss2, idx)
            writer.add_scalar("kl_loss/train", kl_loss, idx)
            writer.add_scalar("valid_recon/train", out[0], idx)
            writer.add_scalar("invalid_recon/train", out[1], idx)
            writer.add_scalar("fail_recon/train", out[2], idx)
            pbar.update(1)

            if idx % 1000 == 0: 
                with torch.no_grad(): 
                    epoch = idx
                    mol_vae.eval()
                    pbar_eval = tqdm(eval_loader, desc=f"EVAL EPOCH {epoch}")
                    # for idx, batch in enumerate(pbar_eval):
                    #     atom_preds, edge_preds, z, mu, logvar = mol_vae(batch)
                    #     atom_loss, edge_loss, atom_edge_loss, kl_loss2, kl_loss, out = vae_loss(atom_preds, edge_preds, z, mu, logvar, batch)
                    #     pbar_eval.update(1)

                    out = mol_vae.sample_mols(100, False)

                    writer.add_scalar("sample_valid_recon/eval", out[0], epoch)
                    writer.add_scalar("sample_invalid_recon/train", out[1], epoch)
                    writer.add_scalar("sample_fail_recon/train", out[2], epoch)

                    out = mol_vae.sample_mols(100, True)

                    writer.add_scalar("sample_rl_valid_recon/eval", out[0], epoch)
                    writer.add_scalar("sample_rl_invalid_recon/train", out[1], epoch)
                    writer.add_scalar("sample_rl_fail_recon/train", out[2], epoch)

                    mol_vae.train()


if __name__ == "__main__":
    train()









