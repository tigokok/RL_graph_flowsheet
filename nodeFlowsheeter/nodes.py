class Node():
    def __init__(self, node_id, node_type):
        self.node_id = node_id
        self.node_type = node_type

class EmptyNode(Node):
    def __init__(self, node_id):
        super().__init__(node_id = node_id, node_type = 'empty')
        self.flowrate = -1
        self.composition = -1
        self.purity = -1
        self.base_price = -1

class FeedNode(Node):
    def __init__(self, node_id, flowrate, composition):
        super().__init__(node_id = node_id, node_type = 'feed')
        self.flowrate = flowrate
        self.composition = composition
        self.weighted_pure_price = -1

class OutputNode(Node):
    def __init__(self, node_id):
        super().__init__(node_id, node_type = 'output')
        self.flowrate = -1
        self.composition = -1
        self.purity = -1
        self.base_price = -1

class MixerNode(Node):
    def __init__(self, node_id):
        super().__init__(node_id = node_id, node_type = 'mixer')

class DstwuNode(Node):
    def __init__(self, node_id, lk=0, hk=1, recov_lk=.99, recov_hk=.01):
        super().__init__(node_id, node_type = 'dstwu')

        self.lk         = lk
        self.hk         = hk
        self.recov_lk   = recov_lk
        self.recov_hk   = recov_hk

        self.R_min      = None
        self.R          = None
        self.N_min      = None
        self.N          = None
        self.Q          = None

        # Distillate vapor flow for column sizing only
        self.V_top = None
        self.M_top = None
        self.T_top = None

        # For reboiler sizing bottom params
        self.M_bot = None
        self.T_bot = None

    def set_params(self, params):
        (self.R_min, self.R, self.N_min, self.N, self.Q, 
        self.V_top, self.M_top, self.T_top,
        self.M_bot, self.T_bot) = params 

class IdealDstwuNode(Node):
    def __init__(self, node_id, spec=0.9, recov_lk=.99, recov_hk=.01):
        super().__init__(node_id, node_type = 'ideal_dstwu')
        
        self.spec = spec
        self.recov_lk = recov_lk
        self.recov_hk = recov_hk

        self.lk = None
        self.hk = None
        

