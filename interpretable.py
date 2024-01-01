import torch
import torch.nn.functional as F
import numpy as np
import dgl
from model_hetero import HAN
from utils import load_data
from hook import Hook


def contrastive_gradient(model, logits, mask, idx):
    _, indices = torch.max(logits[mask], dim=1)

    idx = torch.from_numpy(idx).to('cuda:0')
    indexes = torch.stack([idx, indices], dim=1)
    saliencys = []

    for index in indexes:
        logits[index[0], index[1]].backward(retain_graph=True)
        saliencys.append(F.relu(model.input_features.grad).sum(dim=1))
        model.input_features.grad = None
    saliencys = torch.stack(saliencys)
    torch.save(saliencys, '.\\out\\saliency.pt')


def grad_cam(model: HAN, logits, mask, idx, subg):
    _, indices = torch.max(logits[mask], dim=1)

    idx = torch.from_numpy(idx).to('cuda:0')
    indexes = torch.stack([idx, indices], dim=1)
    heatmap = []
    for index in indexes:
        logits[index[0], index[1]].backward(retain_graph=True)
        one_row_heats = []
        for i in range(2):
            one_row_heats.append([])
            alphas = torch.mean(model.hooks[i].grad, axis=0)
            for n in range(model.hooks[i].output.shape[0]):
                one_row_heats[i].append(F.relu(alphas @ model.hooks[i].output[n]))
            one_row_heats[i] = torch.stack(one_row_heats[i])
            neighbor = subg[i].successors(index[0])
            nodes = torch.cat((torch.tensor([index[0]]).cuda(), neighbor))
            nodes = torch.unique(nodes)
            one_row_heats[i] = torch.mean(one_row_heats[i][nodes])
        # regularization and softmax
        one_row_heats = torch.stack(one_row_heats)
        min_idx = torch.argmin(one_row_heats)
        one_row_heats[min_idx] *= (1 / one_row_heats[1 - min_idx])
        one_row_heats[1 - min_idx] = 1
        one_row_heats = torch.softmax(one_row_heats, dim=0)
        heatmap.append(one_row_heats)
        model.zero_grad()
    heatmap = torch.stack(heatmap)
    torch.save(heatmap, '.\\out\\softmax.pt')


device = "cuda:0"
default_configure = {
    "lr": 0.005,  # Learning rate
    "num_heads": [8],  # Number of attention heads for node-level attention
    "hidden_units": 8,
    "dropout": 0.6,
    "weight_decay": 0.001,
    "num_epochs": 200,
    "patience": 100,
}

(
    g,
    features,
    labels,
    num_classes,
    train_idx,
    val_idx,
    test_idx,
    _,
    train_mask,
    val_mask,
    test_mask,
    _,
    _
) = load_data("ACMRaw")
interpret_idx = np.load('.\\out\\idx.npy')
# interpret_idx = torch.tensor(interpret_idx).to(device)
interpret_mask = torch.load('.\\out\\mask.pt')

if hasattr(torch, "BoolTensor"):
    train_mask = train_mask.bool()
    val_mask = val_mask.bool()
    test_mask = test_mask.bool()
    interpret_mask = interpret_mask.bool()

features = features.to(device)
labels = labels.to(device)
train_mask = train_mask.to(device)
val_mask = val_mask.to(device)
test_mask = test_mask.to(device)
interpret_mask = interpret_mask.to(device)

# HAN(
#   (layers): ModuleList(
#     (0): HANLayer(
#       (gat_layers): ModuleList(
#         (0): GATConv(
#           (fc): Linear(in_features=1903, out_features=64, bias=False)
#           (feat_drop): Dropout(p=0.6, inplace=False)
#           (attn_drop): Dropout(p=0.6, inplace=False)
#           (leaky_relu): LeakyReLU(negative_slope=0.2)
#         )
#         (1): GATConv(
#           (fc): Linear(in_features=1903, out_features=64, bias=False)
#           (feat_drop): Dropout(p=0.6, inplace=False)
#           (attn_drop): Dropout(p=0.6, inplace=False)
#           (leaky_relu): LeakyReLU(negative_slope=0.2)
#         )
#       )
#       (semantic_attention): SemanticAttention(
#         (project): Sequential(
#           (0): Linear(in_features=64, out_features=128, bias=True)
#           (1): Tanh()
#           (2): Linear(in_features=128, out_features=1, bias=False)
#         )
#       )
#     )
#   )
#   (predict): Linear(in_features=64, out_features=3, bias=True)
# )
model = HAN(
    meta_paths=[["pa", "ap"], ["pf", "fp"]],
    in_size=features.shape[1],
    hidden_size=default_configure["hidden_units"],
    out_size=num_classes,
    num_heads=default_configure["num_heads"],
    dropout=default_configure["dropout"],
).to(device)
g = g.to(device)
meta_paths = [["pa", "ap"], ["pf", "fp"]]
subg = [dgl.metapath_reachable_graph(g, meta_paths[0]), dgl.metapath_reachable_graph(g, meta_paths[1])]

model.load_state_dict(torch.load(".\\para.pth"))
model.eval()
logits = model(g, features)
contrastive_gradient(model, logits, interpret_mask, interpret_idx)
grad_cam(model, logits, interpret_mask, interpret_idx, subg)

np.save('.\\out\\idx.npy', interpret_idx)
torch.save(interpret_mask, '.\\out\\mask.pt')
