import numpy as np
from nodeFlowsheeter.unit_operations import *
from nodeFlowsheeter.properties import get_properties

class Simulation:
    def __init__(self, flowsheet, components):
        self.flowsheet = flowsheet
        self.n_comp = len(components)

        # Get component properties
        self.component_properties = get_properties(components)
        self.feeds = [node for node in self.flowsheet.nodes.values() if node.node_type == 'feed']
        self.streams = {}

        self.tolerance = 1e-3
        self.max_iterations = 15

    def weighted_pure_price(self, feed):
        pp_mole = np.array([component['pp_mole'] for component in self.component_properties])
        return np.dot(pp_mole, feed.composition)

    
    def initialize_streams(self):
        """Initialize all streams (edges) in the flowsheet to default unknown values."""
        for edge in self.flowsheet.edges:
            source_node = self.flowsheet.nodes[edge[0]]
            if source_node.node_type == 'feed':
                self.streams[edge] = {
                    "flowrate": source_node.flowrate,
                    "composition": source_node.composition
                }
            else:
                self.streams[edge] = {
                    "flowrate": 0.0,
                    "composition": np.zeros(self.n_comp)
                }
        self.update_streams()

    def update_streams(self):
        self.flowsheet.streams = self.streams

        
    def set_flowsheet(self, flowsheet):
        self.flowsheet = flowsheet

    def run_simulation(self):
        self.initialize_streams()

        converged = False
        iteration_count = 0
        while not converged and iteration_count < self.max_iterations:
            iteration_count += 1
            previous_flows = np.vstack([item['flowrate'] for item in self.streams.values()])
            previous_compositions = np.vstack([item['composition'] for item in self.streams.values()])

            for node_id, node in self.flowsheet.nodes.items():
                input_streams = [self.streams[(src, tgt)] for src, tgt in self.flowsheet.edges if tgt == node_id]
                output_streams = [self.streams[(src, tgt)] for src, tgt in self.flowsheet.edges if src == node_id]
                output_edges = [(src, tgt) for src, tgt in self.flowsheet.edges if src == node_id]

                if len(input_streams) > 1:
                    print('Mixer not yet implemented')
                    print('Likely crash')

                match node.node_type:
                    case 'ideal_dstwu':
                        assert len(input_streams) == 1, f'Unit {node_id} should have 1 input. Check mixer behaviour'
                        assert len(output_streams) == 2, f'Unit {node_id}, type {node.node_type}, should have 2 outputs. Check definition'
                        
                        distillate, bottom = ideal_dstwu(input_streams,
                                                        node.spec,
                                                        node.recov_lk, node.recov_hk)
                        self.streams[output_edges[0]] = distillate
                        self.streams[output_edges[1]] = bottom

                    case 'dstwu':
                        assert len(input_streams) == 1, f'Unit {node_id} should have 1 input. Check mixer behaviour'
                        assert len(output_streams) == 2, f'Unit {node_id}, type {node.node_type}, should have 2 outputs. Check definition'
                        
                        distillate, bottom, params = dstwu_unit(input_streams, 
                                                                node.lk, node.hk, 
                                                                node.recov_lk, node.recov_hk,
                                                                self.component_properties)
                        self.streams[output_edges[0]] = distillate
                        self.streams[output_edges[1]] = bottom
                        node.set_params(params)

                    case 'output':
                        assert len(output_streams) == 0, f'Unit {node_id}, type {node.node_type}, cannot have an outgoing stream.'
                        node.flowrate = input_streams[0]['flowrate']
                        node.composition = input_streams[0]['composition']

                        node.base_price = self.component_properties[int(np.argmax(node.composition))]['pp_mole']
                        node.purity = np.max(node.composition)
                    
                    case 'empty':
                        assert len(output_streams) == 0, f'Unit {node_id}, type {node.node_type}, cannot have an outgoing stream.'
                        node.flowrate = input_streams[0]['flowrate']
                        node.composition = input_streams[0]['composition']

                        node.base_price = self.component_properties[int(np.argmax(node.composition))]['pp_mole']
                        node.purity = np.max(node.composition)

                    case 'feed':
                        node.weighted_pure_price = self.weighted_pure_price(node)
                        
            flows = np.vstack([item['flowrate'] for item in self.streams.values()])
            compositions = np.vstack([item['composition'] for item in self.streams.values()])

            max_flow_diff = np.max(np.abs(flows - previous_flows))
            max_comp_diff = np.max(np.abs(compositions - previous_compositions))

            if max_flow_diff < self.tolerance and max_comp_diff < self.tolerance:
                converged = True
        
        self.update_streams()

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

