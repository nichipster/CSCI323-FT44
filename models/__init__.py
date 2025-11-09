"""
AlphaGo Baseline Models Package

Contains the policy network and complete baseline system
"""

from .policy_net import PolicyNetwork
from .alphago_baseline import AlphaGoBaseline

__all__ = ['PolicyNetwork', 'AlphaGoBaseline']
