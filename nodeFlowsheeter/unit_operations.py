import numpy as np
from scipy.optimize import brentq
import torch
import torch.nn as nn

def bulk_distillate_props(component_properties, composition):
    """Calculates molar weighted component properties

    Args:
        component_properties (Dict): Dictionary with component properties
        composition (np.array): Composition of the components

    Returns:
        _type_: _description_
    """
    mw_values = np.array([component['mw'] for component in component_properties])
    t_values = np.array([component['bp'] for component in component_properties])
    return np.dot(composition, mw_values), np.dot(composition, t_values)

def antoine(component_properties, light_key, heavy_key):
    """Calculates absolute component vapor pressure

    Args:
        component_properties (Dict): Dictionary of component properties
        light_key (int): index of light key
        heavy_key (int): index of heavy key

    Returns:
        np.array: vapor pressure for all components
    """
    antoine_properties = [component['psat'] for component in component_properties]
    boiling_points = [component['bp'] for component in component_properties]
    temperature = (boiling_points[light_key] + boiling_points[heavy_key]) / 2
    vp = [np.exp(p[0] + p[1] / (temperature + p[2]) 
                        + p[3] * temperature 
                        + p[4] * np.log(temperature) 
                        + p[5] * temperature ** p[6]) 
                        for p in antoine_properties]
    
    return np.array(vp)

def h_evap(component_properties, composition):
    """Calculates the enthalpy of evaporation

    Args:
        component_properties (Dict): Dictionary with component properties
        composition (np.array): Composition of all components

    Returns:
        np.array: Enthalpy of evaporation for all components in kJ / mol
    """
    dhv_properties = [comp['dhv'] for comp in component_properties]
    _, temp = bulk_distillate_props(component_properties, composition) 

    dhv_values = []
    for p in dhv_properties:
        a, b, c, d, e, f, g = p
        tc = g + 273.15
        tr = temp / tc
        if tr < 1:
            dhv_i = a * (1 - tr) ** (b + c * tr + d * tr ** 2 + e * tr ** 3) * 4.184
        else:
            dhv_i = 0
        dhv_values.append(dhv_i)
    
    return np.dot(composition, dhv_values)

def underwood_theta(z_feed, alpha, light_key, q=1.0, tol=1e-6):
    """Underwood equation to find the minimum reflux ratio

    Args:
        z_feed (np.array): Composition of the feed
        alpha (np.array): Relative volatilities of all components
        alpha (np.array): Relative volatilities of all components
        q (float, optional): Feed quality, 1 is vapor, 0 is liquid. Defaults to 1.0.
        tol (float, optional): tolerance. Defaults to 1e-6.
    """
    def objective(theta):
        return np.sum(alpha * z_feed / (alpha - theta)) - (1 - q)
    
    lower = 1.0001
    upper = alpha[light_key] * 0.999

    theta = brentq(objective, lower, upper, xtol=tol)
    return theta

def underwood(z_feed, z_dist, alpha, light_key):
    """Underwood equation to find the minimum reflux ratio

    Args:
        z_feed (np.array): Composition of the feed
        z_feez_distd (np.array): Composition of the distillate
        alpha (np.array): Relative volatilities of all components
        alpha (np.array): Relative volatilities of all components

    Returns:
        Float: Minimum reflux ratio
    """
    theta = underwood_theta(z_feed, alpha, light_key)
    return np.sum(z_dist * alpha / (alpha - theta)) - 1


def fenske(z_dist_lk, z_dist_hk, z_bot_lk, z_bot_hk, alpha):
    """Fenske equation to find the minimum amount of stages

    Args:
        z_dist_lk (float): fraction of the light key in distillate
        z_dist_hk (float): fraction of the heavy key in distillate
        z_bot_lk (float): fraction of the light key in bottoms
        z_bot_hk (float): fraction of the heavy key in bottoms
        alpha (float): relative volatility of light key to heavy key

    Returns:
        float: minimum amount of stages
    """
    numerator = np.log((z_dist_lk / z_dist_hk) / (z_bot_lk / z_bot_hk))
    denominator = np.log(alpha)

    return numerator / denominator

def gilliland(R_min, N_min, R):
    """Gilliland equation to find actual amount of stages

    Args:
        R_min (float): Minimum reflux ratio (Underwood)
        N_min (float): Minimum amount of stages (Fenske)
        R (float): Actual reflux ratio

    Returns:
        float: Amount of stages
    """
    assert R > R_min, f'Reflux ratio {R} is lower than Rmin = {R_min}'

    X = (R - R_min) / (R + 1)

    # Eduljee approximation
    Y = 1 - np.exp((1 + 54.4*X)/(11 + 117.2*X) * ((X - 1) / np.sqrt(X)))

    N = (N_min + Y) / (1 - Y)
    
    return N

def dstwu_unit(input_streams, light_key, heavy_key, recovery_lk, recovery_hk, component_properties):
    """Calculates the distillate and bottoms stream using the FUG shortcut method for a distllation column

    Args:
        input_streams (list of dict): A list of all input streams
        light_key (int): Index of light key
        heavy_key (int): Index of heavy key
        recovery_lk (float): Recovery of the light component
        recovery_hk (float): Recovery of the heavy component
        component_properties (dict): Dictionary with component properties

    Returns:
        Distillate (dict): Flowrate and composition of the distillate
        Bottoms (dict): Flowrate and composition of the bottom
        Params (list): other parameters necessary for the column
    """
    feed_flow = input_streams[0]['flowrate']
    feed_comp = input_streams[0]['composition']
    component_flows = feed_flow * feed_comp

    assert (light_key + 1 < len(feed_comp)), 'The light key cannot be the least volatile component'

    vp = antoine(component_properties, light_key, heavy_key)

    x_dist = np.zeros(len(feed_comp))
    x_bot = np.zeros(len(feed_comp))

    x_dist[light_key] = recovery_lk * component_flows[light_key]
    x_bot[light_key] = component_flows[light_key] - x_dist[light_key]

    x_dist[heavy_key] = recovery_hk * component_flows[heavy_key]
    x_bot[heavy_key] = component_flows[heavy_key] - x_dist[heavy_key]

    for i in range(len(feed_comp)):
        if i < light_key:
            x_dist[i] = component_flows[i] * 0.99
            x_bot[i] = component_flows[i] - x_dist[i]

        elif i > heavy_key:
            x_bot[i] = component_flows[i] * 0.99
            x_dist[i] = component_flows[i] - x_bot[i]

        elif heavy_key > i > light_key:
            x_bot[i] = component_flows[i] / 2
            x_dist[i] = component_flows[i] / 2


    distillate_flow, bottom_flow = np.sum(x_dist), np.sum(x_bot)
    z_dist, z_bot = x_dist / distillate_flow, x_bot / bottom_flow
    
    rel_vol = vp / vp[heavy_key]
    
    N_min = fenske(z_dist[light_key], z_dist[heavy_key], z_bot[light_key], z_bot[heavy_key], vp[light_key]/vp[heavy_key])
    R_min = underwood(feed_comp, z_dist, rel_vol, light_key)

    R = R_min * 1.1
    N = gilliland(R_min, N_min, R)


    distillate = {'flowrate': distillate_flow, 'composition': z_dist}
    bottom = {'flowrate': bottom_flow, 'composition': z_bot}
    
    # Calculate Q
    Q = 1.1 * (R_min + 1) * h_evap(component_properties, z_bot)

    V_top = distillate_flow * (1 + R)
    M_top, T_top = bulk_distillate_props(component_properties, z_dist)
    M_bot, T_bot = bulk_distillate_props(component_properties, z_bot)

    params = (R_min, R, N_min, N, Q, 
              V_top, M_top, T_top,
              M_bot, T_bot)
    return distillate, bottom, params


def best_fake_split(input_streams, spec=0.90, recovery_lk=0.99, recovery_hk=0.01):
    """Calculates best possible fake split by trying all options of LK/HK

    Args:
        input_streams (list of dict): List of all ingoing streams with flowrate and composition
        spec (float, optional): Specification at which purity is sellable. Defaults to 0.90.
        recovery_lk (float, optional): Reovery of the light component. Defaults to 0.99.
        recovery_hk (float, optional): Recovery of the heavy. Defaults to 0.01.

    Returns:
        Tuple: index of best light key, best heavy key
    """
    feed_flow = input_streams[0]['flowrate']
    feed_comp = input_streams[0]['composition']
    component_flows = feed_flow * feed_comp

    n = len(feed_comp)

    best = None
    max_flow = 0

    for lk in range(n-1):
        for hk in range(lk + 1, n):
            spec_flow = 0
            x_dist = np.zeros(n)
            x_bot = np.zeros(n)

            # Recoveries
            x_dist[lk] = feed_comp[lk] * recovery_lk
            x_bot[lk] = feed_comp[lk] - x_dist[lk]

            x_dist[hk] = feed_comp[hk] * recovery_hk
            x_bot[hk] = feed_comp[hk] - x_dist[hk]

            for i in range(n):
                if i == lk or i == hk:
                    continue
                if i < lk:
                    x_dist[i] = feed_comp[i] * 0.99
                    x_bot[i] = feed_comp[i] - x_dist[i]
                elif i > hk:
                    x_bot[i] = feed_comp[i] * 0.99
                    x_dist[i] = feed_comp[i] - x_bot[i]
                else:
                    x_dist[i] = feed_comp[i] * 0.5
                    x_bot[i] = feed_comp[i] - x_dist[i]

            flow_dist = np.sum(x_dist)
            flow_bot = np.sum(x_bot)

            z_dist = x_dist / flow_dist if flow_dist > 0 else 0
            z_bot = x_bot / flow_bot if flow_bot > 0 else 0

            # Check if distillate meets spec
            if max(z_dist) >= spec:
                spec_flow += flow_dist

            # Check if bottoms meets spec
            if max(z_bot) >= spec and flow_bot > max_flow:
                spec_flow += flow_bot
            
            if spec_flow > max_flow:
                max_flow = spec_flow
                best = (lk, hk)
    return best 

def ideal_dstwu(input_streams, spec=.9, recovery_lk = 0.99, recovery_hk = 0.01):
    """Calculates distillate and bottoms for an imaginary ideal column

    Args:
        input_streams (list of dict): List of all incoming streams
        spec (float, optional): Specification of the column, desired purity. Defaults to .9.
        recovery_lk (float, optional): Recovery of the light component. Defaults to 0.99.
        recovery_hk (float, optional): Recovery of the heavy component. Defaults to 0.01.

    Returns:
        Distillate (dict): Flowrate and composition of the distillate
        Bottoms (dict): Flowrate and composition of the bottom
    """
    feed_flow = input_streams[0]['flowrate']
    feed_comp = input_streams[0]['composition']
    component_flows = feed_flow * feed_comp

    light_key, heavy_key = best_fake_split(input_streams, spec=spec,
                                           recovery_lk=recovery_lk,
                                           recovery_hk=recovery_hk)

    x_dist = np.zeros(len(feed_comp))
    x_bot = np.zeros(len(feed_comp))

    x_dist[light_key] = recovery_lk * component_flows[light_key]
    x_bot[light_key] = component_flows[light_key] - x_dist[light_key]

    x_dist[heavy_key] = recovery_hk * component_flows[heavy_key]
    x_bot[heavy_key] = component_flows[heavy_key] - x_dist[heavy_key]

    for i in range(len(feed_comp)):
        if i < light_key:
            x_dist[i] = component_flows[i] * 0.99
            x_bot[i] = component_flows[i] - x_dist[i]

        elif i > heavy_key:
            x_bot[i] = component_flows[i] * 0.99
            x_dist[i] = component_flows[i] - x_bot[i]

        elif heavy_key > i > light_key:
            x_bot[i] = component_flows[i] / 2
            x_dist[i] = component_flows[i] / 2

    distillate_flow, bottom_flow = np.sum(x_dist), np.sum(x_bot)
    z_dist, z_bot = x_dist / distillate_flow, x_bot / bottom_flow

    distillate = {'flowrate': distillate_flow, 'composition': z_dist}
    bottom = {'flowrate': bottom_flow, 'composition': z_bot}
    
    return distillate, bottom, light_key, heavy_key