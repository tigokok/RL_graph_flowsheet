import nodeFlowsheeter.price_functions as pr
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
        max_value = 1
        for node_id, node in self.nodes.items():
                match node.node_type:
                    case 'dstwu':
                        val = pr.tac(node)
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

        return value / max_value
    
    def amount_on_spec(self, spec, count_empty = False):
        on_spec = 0
        for node_id, node in self.nodes.items():
            if node.node_type == 'output' and node.purity > spec:
                on_spec += node.flowrate
            
            if count_empty and node.node_type == 'empty' and node.purity > spec:
                on_spec += node.flowrate
                
        return on_spec / 100
