import torch
import torch.nn.functional as F

from model_hetero import HAN
from utils import load_data


def contrastive_gradient(model, logits, mask, idx):
    _, indices = torch.max(logits[mask], dim=1)

    idx = torch.from_numpy(idx).to('cuda:0')
    indexes = torch.stack([idx, indices], dim=1)

    saliencys = []

    for index in indexes:
        logits[index[0], index[1]].backward(retain_graph=True)
        saliencys.append(F.relu(model.input_features.grad[mask]).sum(dim=1))
        model.input_features.grad = None
    # print('len of salience = ',len(saliencys))
    # with open('saliency.out', 'w') as f:
    #     for tensor in saliencys:
    #         f.write(f'{tensor}')


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
    train_mask,
    val_mask,
    test_mask,
) = load_data("ACMRaw")

if hasattr(torch, "BoolTensor"):
    train_mask = train_mask.bool()
    val_mask = val_mask.bool()
    test_mask = test_mask.bool()

features = features.to(device)
labels = labels.to(device)
train_mask = train_mask.to(device)
val_mask = val_mask.to(device)
test_mask = test_mask.to(device)

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

model.load_state_dict(torch.load("./para.pth"))
model.eval()
logits = model(g, features)
contrastive_gradient(model, logits, val_mask, val_idx)


