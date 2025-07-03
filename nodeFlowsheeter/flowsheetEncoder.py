import numpy as np
import torch
from torch_geometric.data import Data
def FlowsheetEncoder(flowsheet, n_comp, verbose = False, returnTorch = True,
                     types = ['feed', 'output', 'empty', 'dstwu']):
    nodes = flowsheet.nodes
    edges = flowsheet.edges
    streams = flowsheet.streams
    node_types = flowsheet.node_types

    n_components = n_comp
    
    # Normalize flowrates
    max_flowrate = max(streams[edge]['flowrate'] for edge in edges)
    for edge in edges:
        streams[edge]['flowrate'] /= max_flowrate

    # Encode nodes
    encoded_nodes = [get_onehot_types(node, node_types) for node in nodes.values()]

    # Encode edges with streams
    node_name_to_idx = {name: i for i, name in enumerate(nodes.keys())}
    edge_index = np.array([
        [node_name_to_idx[src], node_name_to_idx[dst]]
        for src, dst in edges
    ])

    edge_attr = np.array([
    [streams[edge]['flowrate']] + list(streams[edge]['composition'])
    for edge in edges
    ])


    if verbose:
        np.set_printoptions(formatter={'float': '{: 0.3f}'.format})
        print('\n\n== NODES ==')
        print(f'Node shape: {np.shape(encoded_nodes)}')
        print(f'Nodes: {encoded_nodes}')

        print('\n\n== EDGES ==')
        print(f'Edge shape: {np.shape(edge_index)}')
        print(f'Edges: {edge_index}')

        print('\n\n== EDGE ATTRIBUTES ==')
        print(f'Edge attribute shape: {np.shape(edge_attr)}')
        print(f'Edge attributes: {edge_attr}')

    if returnTorch:
        return torch_data(encoded_nodes, edge_index, edge_attr)
        

def torch_data(nodes, edges, edge_attr):
    # Convert to PyTorch tensors
    nodes = torch.tensor(nodes, dtype=torch.float)
    edges = torch.tensor(edges, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)

    # Create a PyG Data object
    data = Data(x=nodes, edge_index=edges, edge_attr=edge_attr)
    return data


def get_onehot_types(node, node_types):
    
    try:
        type_index = node_types.index(node.node_type)
    except ValueError:
        raise ValueError(f"Unknown node type: {node.node_type}")

    type_vector = [int(i == type_index) for i in range(len(node_types))]
    return type_vector

        



