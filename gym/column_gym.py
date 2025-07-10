import numpy as np
import gymnasium as gym
from nodeFlowsheeter.flowsheet import Flowsheet
from nodeFlowsheeter.nodes import *
from nodeFlowsheeter.simulation import Simulation

class ColumnGym(gym.Env):

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
                self.feed_composition = np.array([0.1, 0.2, 0.3, 0.4])

            case 'light_var':
                initial_composition = np.array([0.4, 0.3, 0.2, 0.1]) 
                dev = np.random.normal(loc=0, scale=0.05, size=len(self.components))
                vec = abs(initial_composition + dev)
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
        reward = 0

        node_id = int(action[0])
        node_type = int(action[1])
        lk, hk, recov_lk, recov_hk = action[2:6]

        # Place new node
        # Currently: only 2 actions
        # possible to extend GNN to different actions, i.e. other node types
        match node_type:
            case 0:
                output = OutputNode(node_id)
                self.flowsheet.add_unit(output)
        
            case 1:
                dstwu = DstwuNode(node_id=node_id,
                                lk=lk, 
                                hk=hk, 
                                recov_lk=recov_lk, 
                                recov_hk=recov_hk)
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
    

        # Reward creating valuable stream
        # Also check empty nodes !!!
        # This rewards placing a column

        if node_type == 1:  # column

            in_stream = self.sim.get_incoming_streams(node_id)[0]
            top = self.sim.get_outgoing_streams(node_id)[0]
            bot = self.sim.get_outgoing_streams(node_id)[1]

            in_flow, in_comp = in_stream['flowrate'], in_stream['composition']
            top_flow, top_comp = top['flowrate'], top['composition']
            bot_flow, bot_comp = bot['flowrate'], bot['composition']

            # Identify keys
            top_spec = top_comp[lk]     # LK purity in top stream
            bot_spec = bot_comp[hk]     # HK purity in bottom stream
            
            # Reward for purity (cap at spec target)
            top_purity_reward = min(top_spec / self.spec, 1.02)
            bot_purity_reward = min(bot_spec / self.spec, 1.02)

            reward = max(top_purity_reward, bot_purity_reward)

            reward += top_spec > self.spec
            reward += bot_spec > self.spec

            

            if reward < 0.01:
                reward = -0.5  # discourage wasting steps
            
            print(reward)

            '''
            in_stream = self.sim.get_incoming_streams(node_id)[0]
            top = self.sim.get_outgoing_streams(node_id)[0]
            bot = self.sim.get_outgoing_streams(node_id)[1]

            in_flow, in_comp = in_stream['flowrate'], in_stream['composition']
            top_flow, top_comp = top['flowrate'], top['composition']
            bot_flow, bot_comp = bot['flowrate'], bot['composition']

            # Identify keys
            top_spec = top_comp[lk]     # LK purity in top stream
            bot_spec = bot_comp[hk]     # HK purity in bottom stream

            # Reward for purity (cap at spec target)
            top_purity_reward = min(top_spec / self.spec, 1.0)
            bot_purity_reward = min(bot_spec / self.spec, 1.0)

            # Weight by flow (so large pure streams are better)
            weighted_top = top_purity_reward * top_flow / self.feed_flow
            weighted_bot = bot_purity_reward * bot_flow / self.feed_flow

            # LK and HK flow to the wrong side
            lk_in_bot = bot_flow * bot_comp[lk] / self.feed_flow
            hk_in_top = top_flow * top_comp[hk] / self.feed_flow

            penalty = lk_in_bot + hk_in_top  # total "misplaced" mass

            # Total reward = purity × flow for both streams
            reward = weighted_top + weighted_bot - penalty

            # Optionally penalize unphysical separation (too sharp)
            purity_gap = abs(top_spec - bot_spec)
            if top_spec > 0.95 and bot_spec > 0.95:
                reward -= 1  # overoptimistic separation penalty
            '''

            '''
            current_on_spec = self.flowsheet.amount_on_spec(self.spec, count_empty=True)
            reward = (current_on_spec - self.on_spec)
            self.on_spec = current_on_spec

            if reward <=0:
                reward -= 0.5
            '''


        
        # Append to histories for plotting
        self.reward_history.append(reward)

        # Done = True when no empty nodes left
        if sum(self.flowsheet.get_empty_nodes_mask()) == 0: 
            self.done = 1 

        return self.flowsheet, reward, self.done, self._get_info()
        
    def render(self, mode="text"):
        # TODO
        return ...
