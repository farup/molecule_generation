
from config import config
import torch

from torch.nn.functional import mse_loss
from torch_geometric.utils import to_dense_adj
import torch.nn.functional as F

from VAE.utils.utils_build_molecule import build_molecules, graph_represenatation_to_molecule

def one_hot_to_atomic(matrix): 

    f = lambda x:  F.one_hot(x, (config.NUM_ATOM_TYPES +1))
    matrix.apply_(f)


def get_triangular_uppper(matrix, edge_index_targets, edge_attr_targets): 

    placeholder_matrix = torch.zeros(matrix.shape[0], matrix.shape[1], config.NUM_EDGES_TYPE +1).to(config.DEVICE)
    edge_attr_targets_copy = torch.cat((edge_attr_targets,(torch.argmax(edge_attr_targets, dim=-1)+1).unsqueeze(-1)), dim=-1)
    placeholder_matrix[edge_index_targets[0,:], edge_index_targets[1, :]] = edge_attr_targets_copy

    triu = torch.triu(placeholder_matrix.permute(2,1,0)) # batch of tensors 
    return triu


def kl_div_loss2(z):

    kl_loss = F.kl_div(z.log(), F.softmax(torch.randn_like(z), dim=-1))
    return kl_loss

def kl_div_loss(mu, logstd): 
    """
    Closed formula of the KL divergence for normal distributions
    """
    MAX_LOGSTD = 10
    logstd = logstd.clamp(max=MAX_LOGSTD)
    kl_div = -0.5 * torch.mean(torch.sum(1 + 2* logstd - mu**2 - logstd.exp()**2, dim=1))

    # Limit numeric errors
    kl_div = kl_div.clamp(max=1000)
    return kl_div


def atom_type_loss(atom_pred_matrix, atom_targetss):

    atom_preds_total = torch.sum(atom_pred_matrix[...,:-1], dim=(1, 0))
    atom_targets_total = torch.sum(atom_targetss[:, :5], dim=0)
    
    atom_loss = mse_loss(atom_preds_total, atom_targets_total)

    return atom_loss

def edge_type_loss(edge_pred_matrix, edge_attr_targets): 

    edge_preds_total = torch.sum(edge_pred_matrix, dim=[1, 0])
    edge_targets_total = torch.sum(edge_attr_targets[:-1, ...], dim=[1,2])

    edge_loss = mse_loss(edge_preds_total, edge_targets_total)

    return edge_loss

def atom_edge_pair_loss(atom_pred_matrix, edge_preds, triu_batch_edge_target, atom_targets, batch_index): 

    target_atom_type_edge_connections = torch.matmul(triu_batch_edge_target[:-1, ...], atom_targets[:,:5].unsqueeze(0).repeat(4, 1, 1))
    
    edge__per_atom_type = torch.sum(target_atom_type_edge_connections, dim=[1, 0])

    triu_indices = torch.triu_indices(config.MAX_GRAPH_SIZE, config.MAX_GRAPH_SIZE, offset=1).to(config.DEVICE)

    
    loss = []
    batch = []

    for graph in edge_preds:
        placeholder_matrix = torch.zeros((config.MAX_GRAPH_SIZE, config.MAX_GRAPH_SIZE, config.NUM_EDGES_TYPE)).to(config.DEVICE)

        edge_pred_matrix = graph.unsqueeze(1).reshape(-1, int((config.MAX_GRAPH_SIZE * (config.MAX_GRAPH_SIZE - 1))/ 2), config.NUM_EDGES_TYPE)
        
        placeholder_matrix[triu_indices[0], triu_indices[1]] = edge_pred_matrix
        
        batch.append(placeholder_matrix)

    batch = torch.stack(batch, dim=0)
    pred_atom_type_edge_connections = torch.matmul(batch.permute(0,3,1,2), atom_pred_matrix.unsqueeze(-1).repeat(1,1,1,4).permute(0,3,1,2)[..., :-1]) # batch_sizeXedge_typesXnum_nodesXatom_types
    pred_edge__per_atom_type = torch.sum(pred_atom_type_edge_connections, dim=[-2,1, 0])

    loss = mse_loss(pred_edge__per_atom_type, edge__per_atom_type)


    return loss, batch

def atom_edge_atom_loss(): 
    pass

def extract_target_graph(triu_batch_edge_target,atom_targets, batch_index):

    for graph_id in torch.unique(triu_batch_edge_target): 
        mask = batch_index.eq(graph_id) 
        graph = triu_batch_edge_target[-1, mask][...,mask]

        return graph, atom_targets[mask]


def vae_loss(atom_preds, edge_preds, z, mu, logvar, target_batch):
    atom_targets, edge_index_targets, edge_attr_targets, batch_index = target_batch.x, target_batch.edge_index, target_batch.edge_attr, target_batch.batch 

    edge_pred_matrix = edge_preds.unsqueeze(1).reshape(-1, int((config.MAX_GRAPH_SIZE * (config.MAX_GRAPH_SIZE - 1))/ 2), config.NUM_EDGES_TYPE)

    atom_pred_matrix = atom_preds.unsqueeze(1).reshape(-1, config.MAX_GRAPH_SIZE, config.NUM_ATOM_TYPES+1)

    batch_edge_targets = torch.squeeze(to_dense_adj(edge_index_targets)).to(torch.int64)
    triu_batch_edge_target = get_triangular_uppper(batch_edge_targets, edge_index_targets, edge_attr_targets) # num_edge_types+label_edge_type x num_nodes_in_batch x num_nodes_in_batch

    # edges, atoms = extract_target_graph(triu_batch_edge_target, atom_targets, batch_index)
    # smiles, mol = graph_represenatation_to_molecule(atoms, edges)

    atom_loss = atom_type_loss(atom_pred_matrix, atom_targets)

    edge_loss = edge_type_loss(edge_pred_matrix, triu_batch_edge_target)

    atom_edge_loss, batch = atom_edge_pair_loss(atom_pred_matrix, edge_preds, triu_batch_edge_target, atom_targets, batch_index)

    kl_loss2 = kl_div_loss2(F.softmax(z, dim=-1))

    kl_loss = kl_div_loss(mu, logvar.sqrt())

    out = build_molecules(batch, atom_pred_matrix) / config.BATCH_SIZE

    return atom_loss, edge_loss, atom_edge_loss, kl_loss2, kl_loss, out



    
