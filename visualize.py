import dgl
import spacy
from utils import load_data
import numpy as np
import torch
import json
import matplotlib as mpl
import tqdm
# import networkx as nx
import matplotlib.pyplot as plt


def generate_saliency_graph(subg, i):
    neighbor_nodes = []
    for j in range(2):
        neighbor_nodes.append(subg[j].successors(i))
        neighbor_nodes[j] = torch.cat((i.reshape(1), neighbor_nodes[j]))
        neighbor_nodes[j] = torch.unique(neighbor_nodes[j])
    graph = dgl.merge(subg)
    sg = dgl.node_subgraph(graph, torch.unique(torch.cat(tuple(neighbor_nodes))))
    mask = (sg.edges()[0] >= sg.edges()[1])
    sg = dgl.remove_edges(sg, sg.edges('eid')[mask])
    return sg, neighbor_nodes


def generate_color(saliencys, i, graph):
    saliency = saliencys[i][graph.ndata['_ID']]
    # set color map
    max_val = torch.max(saliency)
    norm = mpl.colors.Normalize(vmin=0, vmax=max_val.item())
    cmap = mpl.colormaps['OrRd']
    plt_colors = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    nodes_color = [plt_colors.to_rgba(weight) for weight in saliency.cpu()]
    return nodes_color


def generate_metapath_color(w):
    idx = int(w[0] > w[1])
    norm = mpl.colors.Normalize(vmin=w[idx].item(), vmax=w[1 - idx].item())
    cmap = mpl.colormaps['viridis']
    plt_colors = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    weights_color = [plt_colors.to_rgba(weight) for weight in w.cpu().detach().numpy()]
    return weights_color


def generate_subedge_nodes(subg, index):
    for i in range(index.shape[0]):
        nodes = []
        for j in range(2):
            neighbor_nodes = torch.unique(subg[j].successors(i))
            nodes.append(neighbor_nodes)
        nodes = torch.cat((nodes[0], nodes[1]), dim=0)
        nodes = torch.unique(torch.cat((torch.tensor([index[i]]).cuda(), nodes)))
        torch.save(nodes, f'.\\subgraph_nodes\\subgraph{i}.pt')


def extract_title(text):
    spacy.prefer_gpu()
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)

    title = []
    for token in doc:
        # Stop searching when a period, question mark, or exclamation mark is found
        if token.text in [".", "?", "!", "  "]:
            break
        elif (not token.is_stop) and (not token.is_punct):
            title.append(token.text)

    return ' '.join(title)


def generate_paper_name(p_name):
    p_info = []
    for i in tqdm.tqdm(range(len(p_name))):
        p_info.append(extract_title(str(p_name[i][0][0])))
    return p_info


device = 'cuda'

(
    hg,
    features,
    labels,
    num_classes,
    train_idx,
    val_idx,
    test_idx,
    interpret_idx,
    train_mask,
    val_mask,
    test_mask,
    interpret_mask,
    p_name
) = load_data("ACMRaw")

index = np.load('.\\out\\idx.npy')
index = torch.tensor(index).to(device)
metapath_weight = torch.load('.\\out\\metapath.pt').to(device)
saliencys = torch.load('.\\out\\saliency.pt').to(device)
predict = torch.load('.\\out\\predict.pt').to(device)
meta_paths = [["pa", "ap"], ["pf", "fp"]]
hg = hg.to(device)
subg = [dgl.metapath_reachable_graph(hg, meta_paths[0]), dgl.metapath_reachable_graph(hg, meta_paths[1])]

path_name = ["Paper-Author-Paper", "Paper-Field-Paper"]
with open('.\\out\\paper_name.txt', 'r') as f:
    p_info = [line.rstrip() for line in f]

for i in tqdm.tqdm(range(index.shape[0])):
    w = metapath_weight[i]
    # w_color = generate_metapath_color(w)
    # w_color = [
    #     '#{:02x}{:02x}{:02x}'.format(int(w_color[0][0] * 255), int(w_color[0][1] * 255), int(w_color[0][2] * 255)),
    #     '#{:02x}{:02x}{:02x}'.format(int(w_color[1][0] * 255), int(w_color[1][1] * 255), int(w_color[1][2] * 255))]
    g, subg_nodes = generate_saliency_graph(subg, index[i])
    nodes_color = generate_color(saliencys, i, g)
    js_graph = {"nodes": [], "edges": []}
    # nodes
    for j in range(g.nodes().shape[0]):
        nid = g.ndata['_ID'][j]
        meta_path = 0 if nid in subg_nodes[0] else 1
        if nid == index[i]:
            rgb = '#006284'
        else:
            rgb = '#{:02x}{:02x}{:02x}'.format(int(nodes_color[j][0] * 255), int(nodes_color[j][1] * 255),
                                               int(nodes_color[j][2] * 255))
        js_graph['nodes'].append(
            {'id': f'{nid}',
             'description': f'Paper:\n{p_info[nid]}\nMetapath:\n{"Paper-Author-Paper" if meta_path == 1 else "Paper-Field-Paper"}, weight = {w[meta_path]}\nLabel: {labels[nid]}\nPrediction: {predict[nid]}',
             'style': {'keyshape': {'fill': rgb, 'stroke': rgb, 'fillOpacity': 1}}})
    for j in range(g.edges()[0].shape[0]):
        src = g.ndata['_ID'][g.edges()[0][j]]
        dst = g.ndata['_ID'][g.edges()[1][j]]
        js_graph['edges'].append({"source": f'{src}', "target": f'{dst}', 'style': {'keyshape': {'endArrow': False}}})
    with open(f'.\\graphs\\graph-{i}.json', 'w') as f:
        json.dump(js_graph, f)

# def draw_networkx(subg, index, saliencys):
#     # for i in range(index.shape[0]):
#     i = 4
#     nx_g = []
#     for j in range(2):
#         neighbor_nodes = subg[j].successors(index[i])
#         nodes = torch.cat((torch.tensor([index[i]]).cuda(), neighbor_nodes))
#         nodes = torch.unique(nodes)
#         sg = dgl.in_subgraph(subg[j], nodes, relabel_nodes=True, store_ids=True)
#         # mask = (sg.nodes()[0] <= sg.nodes()[1])
#         nx_g.append(dgl.to_networkx(sg.cpu()))
#         origin_labels = {k.item(): v.item() for k, v in zip(sg.nodes(), nodes)}
#         nx.relabel_nodes(nx_g[j], origin_labels, copy=False)
#     nx_g = nx.compose(*nx_g)
#     nx_g = nx_g.to_undirected()
#     nodes = list(nx_g.nodes())
#     saliency = saliencys[i][nodes]
#
#     # set color map
#     max_val = torch.max(saliency)
#     norm = mpl.colors.Normalize(vmin=0, vmax=max_val.item())
#     cmap = mpl.colormaps['OrRd']
#     plt_colors = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
#     nodes_color = [plt_colors.to_rgba(weight) for weight in saliency.cpu()]
#
#     # draw networkx
#     plt.figure(figsize=[12, 12])
#     plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
#     pos = nx.fruchterman_reingold_layout(nx_g, k=0.2)
#     nx.draw_networkx(nx_g, node_color=nodes_color, pos=pos, style='dashed', width=0.1, node_size=500, font_size=10)
#     plt.show()
#     # print(metapath_heatmaps[index[i], :, i])
