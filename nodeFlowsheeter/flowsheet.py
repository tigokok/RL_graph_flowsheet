import nodeFlowsheeter.price_functions as pr
import matplotlib.pyplot as plt
import networkx as nx
class Flowsheet():
    def __init__(self, node_types, name = 'Flowsheet name'):
        """Initializes a flowsheet object

        Args:
            name (str, optional): Name of the flowsheet. Defaults to 'Flowsheet name'.
        """
        self.name       =   name
        self.nodes      =   {}
        self.edges      =   []
        self.streams    =   {}
        self.current_id =   0

        self.node_types      =   node_types

    def reset(self):
        """Clears the nodes, edges and streams of the flowsheet object.
        It also sets the current id to 0 such that newly placed nodes start with index 0.
        """
        self.nodes.clear()
        self.edges.clear()
        self.streams.clear()
        self.current_id = 0

    def unique_id(self):
        """A counter for unique ID's
        Running this function updates the current ID to the current ID + 1 and returns the old current ID as a unique ID.

        Returns:
            int: A unique indentifier (ID) to identify a node.
        """
        unique = self.current_id
        self.current_id += 1
        return unique

    def add_unit(self, unit_nodes):
        """Add one or more units to the flowsheet

        Args:
            unit_nodes (Object): A unit node object as given in nodes.py
        """
        if not isinstance(unit_nodes, list):
            unit_nodes = [unit_nodes]

        for node in unit_nodes:
            self.nodes[node.node_id] = node

    def connect(self, source_ids, target_ids):
        """Connects one or more units from source_id to target_id. It connects all source id's to all target id's.

        Args:
            source_ids (Int*): List of node ID's of the source nodes
            target_ids (Int*): List of node ID's of the target nodes
        """
        if not isinstance(source_ids, list):
            source_ids = [source_ids]
        if not isinstance(target_ids, list):
            target_ids = [target_ids]

        for src in source_ids:
            for tgt in target_ids:
                assert src in self.nodes and tgt in self.nodes, f'Node {src} or {tgt} does not exist.'
                self.edges.append((src, tgt))

    def get_empty_nodes_mask(self):
        """Returns a mask of shape [N, 1] with
        - N = amount of nodes
        That is 1 for all nodes of type 'empty'

        Returns:
            Int*: A mask that is 1 for all 'empty' nodes and 0 for all others.
        """
        return [1 if node.node_type == 'empty' else 0 for node in self.nodes.values()]

    def display(self):
        """Prints an overview of the current flowsheet.
        """
        print(f'Flowsheet name: {self.name}')

        print('\nNodes: \n')
        [print(f'Node ID: {node_id:<20} \ttype: {node.node_type:>20}') for node_id, node in self.nodes.items()]      
        
        print('\nConnections: \n')
        [print(f'{edge[0]:<10} -> {edge[1]:>10}') for edge in self.edges]

    def print_streams(self):
        """Prints all streams/edges of the current flowsheet. Not to be used before the simulation.
        """
        for edge in self.edges:
            # Get the stream data for each edge
            flowrate = self.streams[edge]["flowrate"]
            composition = self.streams[edge]["composition"]

            print(f'Stream: {str(edge):<20}, flowrate: {flowrate:<10.3f}, composition: {composition}')

    def generate_stream_table(self):
        """Generate a list of tuples containing edge, flowrate, and composition."""
        stream_data = []
        for edge in self.flowsheet.edges:
            # Get the stream data for each edge
            flowrate = self.streams[edge]["flowrate"]
            composition = self.streams[edge]["composition"]

            # Append the data as a tuple to the list
            stream_data.append((edge, flowrate, composition))

        return stream_data
    
    def estimate_flowsheet_value(self, spec, carbon_tax = False):
        """Estimates the value of the current flowsheet

        Args:
            spec (_type_): _description_
            carbon_tax (bool, optional): _description_. Defaults to False.

        Returns:
            _type_: _description_
        """
        value = 0
        tac = 0
        max_value = 1
        for node_id, node in self.nodes.items():
                match node.node_type:
                    case 'dstwu':
                        val = pr.tac(node)
                        tac += val
                    case 'output':
                        val = pr.output_price(node, spec)
                    case 'empty':
                        val = pr.output_price(node) / 4
                    case 'feed':
                        val = 0
                        max_value = pr.max_value(node)
                    case _:
                        val = 0
                        
                value += val

        return value / max_value, tac
    
    def amount_on_spec(self, spec, count_empty = False):
        on_spec = 0
        for node_id, node in self.nodes.items():
            if node.node_type == 'output' and node.purity > spec:
                on_spec += node.flowrate
            
            if count_empty and node.node_type == 'empty' and node.purity > spec:
                on_spec += node.flowrate
            
            if node.node_type == 'feed':
                flowrate = node.flowrate
                
        return on_spec / flowrate
    
    def display_graph(self):
        """Displays the flowsheet with a left-to-right layout, placing outputs up/down and highlighting columns."""
        from matplotlib.colors import to_rgb
        import matplotlib.pyplot as plt
        import networkx as nx
        from collections import defaultdict, deque

        light_tone = to_rgb("#B8B3E9")
        dark_tone = to_rgb("#5D51CD")
        light_tone_alt = to_rgb("#F3DFC1")
        dark_tone_alt = to_rgb("#DB9E43")

        G = nx.DiGraph()

        # Add nodes with appropriate labels
        for node_id, node in self.nodes.items():
            node_type = node.node_type
            if node_type == 'dstwu':
                label = (
                    f'ID: {node_id}\n'
                    f'Column\n'
                    f'lk: {getattr(node, "lk", "–")}, hk: {getattr(node, "hk", "–")}\n'
                    f'Stages: {getattr(node, "N", "-")}\n'
                    f'RR: {format(getattr(node, "R", "-"), ".3f")}\n'
                    f'η_lk: {format(getattr(node, "recov_lk", "–"), ".3f")}\n'
                    f'η_hk: {format(getattr(node, "recov_hk", "–"), ".3f")}'
                )
            else:
                label = f'ID: {node_id}\n{node_type}'
            G.add_node(node_id, label=label, type=node_type)

        for src, tgt in self.edges:
            G.add_edge(src, tgt)

        # Format edge labels: flowrate and formatted composition
        edge_labels = {}
        for edge in self.edges:
            stream = self.streams.get(edge, {})
            flowrate = stream.get("flowrate")
            comp = stream.get("composition")
            label = ''
            if flowrate is not None:
                label += f'flow: {flowrate:.2f}'
            if comp is not None:
                comp_lines = [f'{format(comp[i], ".3f")}' for i in range(len(comp))]
                label += '\n' + '\n'.join(comp_lines)

            edge_labels[edge] = label


        # Step 1: Compute depth from feeds using BFS
        in_deg = {n: 0 for n in G.nodes}
        for u, v in G.edges:
            in_deg[v] += 1

        depth = {}
        queue = deque([n for n in G.nodes if in_deg[n] == 0])
        for n in queue:
            depth[n] = 0

        while queue:
            node = queue.popleft()
            for succ in G.successors(node):
                depth[succ] = max(depth.get(succ, 0), depth[node] + 1)
                in_deg[succ] -= 1
                if in_deg[succ] == 0:
                    queue.append(succ)

        # Step 2: Assign positions
        pos = {}
        y_tracker = defaultdict(int)
        output_offsets = defaultdict(int)

        for node_id in sorted(G.nodes, key=lambda n: depth[n]):
            node_type = G.nodes[node_id]['type']
            d = depth[node_id]
            if node_type == 'output':
                preds = list(G.predecessors(node_id))
                if preds:
                    x = depth[preds[0]]
                    offset = 1.5 if output_offsets[x] == 0 else -1.5
                    y = pos[preds[0]][1] + offset
                    output_offsets[x] += 1
                    pos[node_id] = (x, y)
                    continue
            x = d
            y = -y_tracker[d]
            pos[node_id] = (x, y)
            y_tracker[d] += 1

        # Step 3: Drawing
        plt.figure(figsize=(10, 6), layout = 'tight')
        node_labels = nx.get_node_attributes(G, 'label')

        node_sizes = []
        node_colors = []
        for node in G.nodes:
            typ = G.nodes[node]['type']
            if typ in ['dstwu']:
                node_sizes.append(4200)
                node_colors.append(light_tone)  # Orange-ish
            else:
                node_sizes.append(3200)
                node_colors.append(light_tone_alt)  # Light blue

        nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=node_colors, edgecolors='black', node_shape='s')
        nx.draw_networkx_edges(G, pos, edge_color='gray', arrows=True, arrowsize=15, width=1.4, connectionstyle="arc3,rad=0.05")
        nx.draw_networkx_labels(G, pos, labels=node_labels, font_color='black', font_size=8)

        # Edge labels: horizontal alignment
        nx.draw_networkx_edge_labels(
            G,
            pos,
            edge_labels=edge_labels,
            font_size=8,
            font_color=dark_tone,
            label_pos=0.5,
            rotate=False
        )

        plt.title(f'Flowsheet: {self.name}', fontsize=14)
        plt.axis('off')
        plt.tight_layout()
        plt.show()

