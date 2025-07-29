import numpy as np
from nodeFlowsheeter.properties import get_properties
R               =   8.314       # Gas constant [J/K*mol]
M_S             =   1800        # Marshall and swift index [2025, estimated]

OP_HOURS        =   8000        # Yearly operating hours
M_WATER         =   0.018       # MW water [kg/mol]
C_STEAM         =   0.018       # Steam price [EUR/kg]
CP              =   4.2         # kj / kg K
C_CW            =   0.000006    # Cooling water price [EUR/kg]
DHV             =   34.794      # Molar heat of condensation of 16bar steam [kJ/mol]
T_COOL_IN       =   30          # Supply cooling water temperature [°C]
T_COOL_OUT      =   40          # Return cooling water temperature [°C]



def columnDiameter(V_top, M_top, T_top, P_top = 1e5, f = 1.6):
    D_eff = np.sqrt((4 * V_top)/(np.pi * f) * np.sqrt(R * T_top * (M_top/1000) / P_top))
    return 1.1 * D_eff

def columnHeight(N, HETP = 0.5, H_0 = 4.0):
    """Calculates column height for distillation column with N stages

    Args:
        N (int): Amount of stages
        HETP (float, optional): Height equivalent of theoretical plate. Defaults to 0.5.
        H_0 (float, optional): Base height. Defaults to 4.

    Returns:
        float: Column height
    """
    return N * HETP + H_0

def columnCosts(height, diameter, f_m = 1.0, f_p = 1.0):
    f_c = f_m + f_p
    return 0.9 * (M_S / 280) * 937.64 * diameter**1.066 * height**0.802 * f_c

def internalCosts(height, diameter, f_m = 1.0, f_int_s = 1.4, f_int_t = 0.0, f_int_m = 0):
    f_int_c = f_int_s + f_int_t + f_int_m
    return 0.9 * (M_S / 280) * 97.24 * diameter**1.55 * height * f_int_c

def reboilerArea(Q_rbl, T_bot, K_rbl = 800, T_steam = 201 + 273):
    assert T_steam > T_bot, f'T_bot {T_bot}  smaller than T_steam {T_steam}, infeasible'
    A = Q_rbl / (K_rbl * (T_steam - T_bot)) * 1000
    return A

def reboilerCosts(Q_rbl, T_bot, F_rbl_c = 0.8):
    A_rbl = reboilerArea(Q_rbl, T_bot)
    return 0.9 * (M_S / 280) * 474.67 * A_rbl**0.65 * F_rbl_c

def condensorArea(Q_cnd, T, K_cnd = 500, T_cool_in = 30 + 273, T_cool_out = 40 + 273):
    lmtd = ((T - T_cool_in) * (T - T_cool_out) * ((T - T_cool_in) + (T - T_cool_out)) / 2)**(1/3)
    A_cnd = Q_cnd / (K_cnd * lmtd) * 1000
    return A_cnd

def condensorCosts(Q_cnd, T_top, F_cnd_c = 0.8):
    A_cnd = condensorArea(Q_cnd, T_top)
    return 0.9 * (M_S / 280) * 474.67 * A_cnd ** 0.65 * F_cnd_c

def condensorOperation(Q_cnd):
    return Q_cnd * C_CW / (CP * (T_COOL_OUT - T_COOL_IN)) 

def reboilerOperation(Q_rbl):
    return Q_rbl * M_WATER * C_STEAM / DHV

def investmentCosts(dstwuNode):
    height = columnHeight(dstwuNode.N)
    diameter = columnDiameter(dstwuNode.V_top, dstwuNode.M_top, dstwuNode.T_top)

    column = columnCosts(height, diameter)
    internals = internalCosts(height, diameter)
    reboiler = reboilerCosts(dstwuNode.Q_bot, dstwuNode.T_bot)
    condensor = condensorCosts(dstwuNode.Q_dist, dstwuNode.T_top)

    return column + internals + reboiler + condensor

def operatingCosts(dstwuNode):
    reboiler = reboilerOperation(dstwuNode.Q_bot)
    condensor = condensorOperation(dstwuNode.Q_dist)
    
    return (reboiler + condensor) * 3600 * OP_HOURS

def tac(dstwuNode):
    return -(investmentCosts(dstwuNode) + operatingCosts(dstwuNode))

def max_value(feedNode):
    mol_sec = feedNode.flowrate * feedNode.weighted_pure_price
    value = mol_sec * 3600 * OP_HOURS
    return value

def output_price(outputNode, spec = 0.90):
    mol_sec = outputNode.flowrate * outputNode.base_price
    value = mol_sec * 3600 * OP_HOURS

    if outputNode.purity < spec:
        value = 0
    
    return value