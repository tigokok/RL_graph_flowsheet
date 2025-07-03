import numpy as np
from nodeFlowsheeter.flowsheet import *
from nodeFlowsheeter.nodes import *
from nodeFlowsheeter.simulation import Simulation
from nodeFlowsheeter.flowsheetEncoder import FlowsheetEncoder
import torch

# Set parameters
# feed_flow in mol/sec
# Components: choose from properties.py
# Feed_composition: np.array() with sum == 1
feed_flow = 100
components = ['cyclo-pentane', 'n-hexane', 'n-heptane', 'n-octane']
feed_composition = np.array([0.4, 0.3, 0.2, 0.1])


# Make nodes
feed = FeedNode('feed', feed_flow, feed_composition)
C1 = DstwuNode('C1', lk = 0, hk = 1, recov_lk = 0.99, recov_hk = 0.01)
C1_top = OutputNode('C1_top')

C2 = DstwuNode('C2', lk = 1, hk = 2, recov_lk = 0.99, recov_hk = 0.01)
C2_top = EmptyNode('C2_top')
C2_bot = EmptyNode('C2_bot')

# Initialize flowsheet object
node_types = ['feed', 'output', 'empty', 'dstwu']
flowsheet_example = Flowsheet(node_types=node_types,
                              name = 'Example sheet')

# Include nodes in flowsheet
# You can add more than 1!
flowsheet_example.add_unit([feed, C1, C1_top, C2, C2_top, C2_bot])

# Connect nodes
flowsheet_example.connect('feed', 'C1')
flowsheet_example.connect('C1', ['C1_top', 'C2'])
flowsheet_example.connect('C2', ['C2_top', 'C2_bot'])

# Display connections in console
flowsheet_example.display()

# Initialize a simulation object
sim = Simulation(flowsheet = flowsheet_example, components = components)


# Run the flowsheet
sim.run_simulation()

# Estimate value
# Value is normalized by the maximum value one could achieve if all components
# were to be separated perfectly and sold.
# 1 is not achieveable as there's always a separation cost.
print(f'VALUE: {flowsheet_example.estimate_flowsheet_value(spec = 0.9) * 100}% of max sell value')

# Print streams from flowsheet
flowsheet_example.print_streams()

# Get specific variables
debug = 0
if debug:
    print(flowsheet_example.nodes['C2'].R_min)
    print(flowsheet_example.nodes['C2_top'].composition)
    print(flowsheet_example.nodes['C2'].V_top)

    print(flowsheet_example.nodes['C1'].Q)

    #print(flowsheet_example.nodes)
    #print(flowsheet_example.edges)
    #print(flowsheet_example.streams)

# Encode flowsheet for GNN
# This function takes a flowsheet object as input and makes it into a PyG Data object
# used to store graphs
td = FlowsheetEncoder(flowsheet_example, len(components), verbose = True)

# Reset the flowsheet
flowsheet_example.reset()
flowsheet_example.display()