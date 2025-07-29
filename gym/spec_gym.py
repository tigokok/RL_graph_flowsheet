import numpy as np
import gymnasium as gym
from nodeFlowsheeter.flowsheet import Flowsheet
from nodeFlowsheeter.nodes import *
from nodeFlowsheeter.simulation import Simulation

class SpecGym(gym.Env):

    # Initialize gym
    def __init__(self, comp_mode='light',
                 feed_flow = 100,
                 components = ['cyclo-pentane', 'n-hexane', 'n-heptane', 'n-octane'],
                 feed_composition = np.array([0.4, 0.3, 0.2, 0.1]),
                 node_types = ['feed', 'output', 'empty', 'ideal_dstwu']):
                
        ## VARIABLES
        self.comp_mode = comp_mode
        self.feed_flow = feed_flow                    
        self.feed_composition = feed_composition
        self.components = components
        self.node_types = node_types

        # Initialize flowsheet
        self.flowsheet = Flowsheet(self.node_types, name = 'Gym flowsheet')
        self.sim = Simulation(self.flowsheet, self.components)
        self.spec = 0.9
        
        # For plotting
        self.reward_history = []
        self.done = 0
    
    # Gym standard. Unnecessary here.
    def _get_info(self):
        return 0

    # Reset environment
    def reset(self):
        self.flowsheet.reset()

        match self.comp_mode:
            case 'match_init':
                self.feed_composition = self.feed_composition
                
            case 'light':
                self.feed_composition = np.array([0.4, 0.3, 0.2, 0.1])

            case 'light_var':
                initial_composition = np.array([0.4, 0.3, 0.2, 0.1]) 
                dev = np.random.normal(loc=0, scale=0.05, size=len(self.components))
                vec = initial_composition + dev
                self.feed_composition = vec/vec.sum()
            
            case 'heavy':
                self.feed_composition = np.array([0.1, 0.2, 0.3, 0.4]) 

            case 'heavy_var':
                initial_composition = np.array([0.1, 0.2, 0.3, 0.4]) 
                dev = np.random.normal(loc=0, scale=0.05, size=len(self.components))
                vec = initial_composition + dev
                self.feed_composition = vec/vec.sum()

            case 'random':
                vec = np.random.rand(len(self.components)) + 0.1
                self.feed_composition = vec / vec.sum()

        # Place feed node and empty node
        # and connect the feed to the empty node to start
        # TODO: possible create scenario's, like starting with something
        feed = FeedNode(self.flowsheet.unique_id(), self.feed_flow, self.feed_composition)
        first_node = EmptyNode(self.flowsheet.unique_id())
        self.flowsheet.add_unit([feed, first_node])
        self.flowsheet.connect(feed.node_id, first_node.node_id)

        self.sim.initialize_streams()
        
        self.on_spec = 0
        self.done = 0
        info = self._get_info()
        
        # For plotting
        self.reward_history = []
        return self.flowsheet, info
    
    def step(self, action):
        node_id = action[0]
        node_type = action[1]

        # Place new node
        # Currently: only 2 actions
        # possible to extend GNN to different actions, i.e. other node types
        match node_type:
            case 0:
                output = OutputNode(node_id)
                self.flowsheet.add_unit(output)
        
            case 1:
                dstwu = IdealDstwuNode(node_id, self.spec)
                dist = EmptyNode(self.flowsheet.unique_id())
                bot = EmptyNode(self.flowsheet.unique_id())

                self.flowsheet.add_unit([dstwu, dist, bot])
                self.flowsheet.connect(dstwu.node_id, [dist.node_id, bot.node_id])

        # Run simulation
        # Done if fails
        try:
            self.sim.run_simulation()  
        except:
            self.done = 1
    

        # -------------------
        # Calculate value
        reward = 0

        # Reward creating valuable stream
        # Also check empty nodes !!!
        # This rewards placing a column

        if node_type == 1:
            current_on_spec = self.flowsheet.amount_on_spec(self.spec, count_empty=True)
            reward = (current_on_spec - self.on_spec)
            self.on_spec = current_on_spec

            if reward <=0:
                reward -= 0.5
        
        # Reward placing output on spec
        if node_type == 0:
            if output.purity > self.spec:
                reward = (output.flowrate / self.feed_flow)
            else:
                reward = -(output.flowrate / self.feed_flow)



        
        # Append to histories for plotting
        self.reward_history.append(reward)

        # Done = True when no empty nodes left
        if sum(self.flowsheet.get_empty_nodes_mask()) == 0: 
            self.done = 1 

        return self.flowsheet, reward, self.done, self._get_info()
        
    def render(self, mode="text"):
        # TODO
        return ...
