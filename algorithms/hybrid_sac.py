"""
Discrete Soft Actor-Critic with Trigger-Conditioned Graph Attention Network
for Microservice Migration.

Key Innovation:
1. TriggerAwareGAT: Physics-aware graph attention that conditions on trigger type
   (PROACTIVE vs REACTIVE) to learn differentiated attention patterns.
2. Discrete SAC: Maximum entropy RL for discrete action spaces, encouraging
   exploration while respecting SA priors.

Architecture Overview:
- TriggerAwareGAT: Graph feature extraction with trigger-conditioned attention
- SAC Actor: Policy network outputting action probabilities
- SAC Critic: Twin Q-networks to mitigate overestimation bias

Actions: [Stay, Follow SA, Nearest] - 3 discrete choices per microservice node

Reference:
- Haarnoja et al., "Soft Actor-Critic Algorithms and Applications", 2018
- Christodoulou, "Soft Actor-Critic for Discrete Action Settings", 2019
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from collections import deque
from tqdm import tqdm
import random
import copy
import math
import time

from core.microservice_dags import MICROSERVICE_DAGS
from core.geo import haversine_distance, find_k_nearest_servers
from core.context import get_trigger_type, TRIGGER_PROACTIVE, TRIGGER_REACTIVE
from core.dag_utils import get_entry_nodes, topological_sort, assign_dag_type, initialize_dag_assignment
from core.reward import build_servers_info, calculate_microservice_reward
from core.state_builder import build_graph_state
from algorithms.sa import microservice_simulated_annealing

FORECAST_HORIZON = 15


# =============================================================================
# Module 1: Trigger-Conditioned Graph Attention Network (TriggerAwareGAT)
# =============================================================================

class TriggerAwareGAT(nn.Module):
    """
    Trigger-Conditioned Graph Attention Network for Microservice Migration.
    
    This module implements a physics-aware graph attention mechanism that
    explicitly conditions on the trigger type (PROACTIVE vs REACTIVE) to learn
    different attention patterns for migration decisions.
    
    Physical Intuition of Trigger-Conditioned Attention:
    ====================================================
    
    The SLA trigger type fundamentally changes the physics of migration:
    
    1. PROACTIVE Trigger [1.0, 0.0]:
       - System has advance warning → time for background state synchronization
       - Stateful nodes can be migrated with LOW COST (async state transfer)
       - Attention should SPREAD across topology to consider global optimization
       - Network learns to be more AGGRESSIVE with stateful node migrations
    
    2. REACTIVE Trigger [0.0, 1.0]:
       - SLA already violated → urgent migration needed
       - Stateful nodes have HIGH COST (synchronous state transfer, downtime)
       - Attention should FOCUS on critical nodes (entry points, high-traffic)
       - Network learns to be CONSERVATIVE with stateful nodes
    
    How Trigger Vector Physically Intervenes in Attention Weights:
    ==============================================================
    
    The trigger context vector modulates attention in three ways:
    
    (A) Feature Enrichment (Line: enriched_features):
        - Trigger vector is concatenated to EVERY node's features
        - This allows the network to learn node-specific responses to triggers
        - e.g., A stateful node with REACTIVE trigger activates different neurons
          than the same node with PROACTIVE trigger
    
    (B) Attention Gating (Line: trigger_gate):
        - A learned gating function produces trigger-specific attention modulation
        - PROACTIVE: Higher gate values → softer attention (more uniform distribution)
        - REACTIVE: Lower gate values → sharper attention (focus on critical paths)
        - Mathematical form: attention_scores *= sigmoid(W_gate @ trigger_context)
    
    (C) Stateful-Trigger Interaction (Line: stateful_trigger_bias):
        - Learned bias term that specifically models the stateful × trigger interaction
        - REACTIVE + stateful: Strong negative bias → avoid unnecessary attention
        - PROACTIVE + stateful: Positive bias → safe to consider for migration
        - This captures the asymmetric migration cost structure
    
    Parameters
    ----------
    node_feat_dim : int
        Dimension of per-node features (default: 3 for [image_mb, state_mb, is_stateful])
    trigger_dim : int
        Dimension of trigger context.
        - v3.6 and earlier: 2 for one-hot [PROACTIVE, REACTIVE]
        - v3.7+: 3 for [PROACTIVE, REACTIVE, risk_ratio] with continuous risk feature
    hidden_dim : int
        Hidden layer dimension for attention computation
    num_heads : int
        Number of attention heads (default: 2 for multi-perspective attention)
    output_dim : int
        Output embedding dimension per node
    dropout : float
        Dropout rate for regularization
    """
    
    def __init__(self, node_feat_dim=3, trigger_dim=3, hidden_dim=64,
                 num_heads=2, output_dim=64, dropout=0.1):
        super().__init__()
        
        self.node_feat_dim = node_feat_dim
        self.trigger_dim = trigger_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.output_dim = output_dim
        
        # Input dimension after trigger concatenation
        enriched_dim = node_feat_dim + trigger_dim  # 3 + 2 = 5
        
        # ============================================================
        # Stage 1: Feature Projection
        # Project enriched node features to hidden space
        # ============================================================
        self.node_projection = nn.Sequential(
            nn.Linear(enriched_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        
        # ============================================================
        # Stage 2: Multi-Head Attention Parameters
        # Each head learns different aspects of the topology
        # ============================================================
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"
        self.head_dim = hidden_dim // num_heads
        
        # Query, Key, Value projections for multi-head attention
        self.W_query = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_key = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_value = nn.Linear(hidden_dim, hidden_dim, bias=False)
        
        # ============================================================
        # Stage 3: Physics-Aware Attention Modulation
        # These components implement the trigger-conditioned attention
        # ============================================================
        
        # (A) Trigger-based attention temperature scaling
        # Controls attention sharpness: PROACTIVE → softer, REACTIVE → sharper
        self.trigger_temperature = nn.Sequential(
            nn.Linear(trigger_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_heads),
            nn.Softplus(),  # Ensure positive temperature
        )
        
        # (B) Trigger-based attention gating
        # Modulates overall attention strength based on trigger type
        self.trigger_gate = nn.Sequential(
            nn.Linear(trigger_dim, hidden_dim),
            nn.Sigmoid(),
        )
        
        # (C) Stateful-Trigger interaction bias
        # Learned parameter capturing the asymmetric cost structure
        # Shape: (num_heads,) - different bias per attention head
        self.stateful_trigger_bias = nn.Parameter(torch.zeros(num_heads))
        
        # ============================================================
        # Stage 4: Output Projection
        # Combine multi-head outputs and project to final embedding
        # ============================================================
        self.output_projection = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim),
        )
        
        # Residual connection scaling (learnable)
        self.residual_scale = nn.Parameter(torch.ones(1) * 0.1)
        
        # Store computed embeddings for downstream use
        self._node_embeddings = None
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using Xavier/Glorot initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, node_features, adj_matrix, trigger_context):
        """
        Compute trigger-conditioned graph attention embeddings.
        
        Parameters
        ----------
        node_features : torch.Tensor, shape (N, 3)
            Per-node features: [image_mb/200, state_mb/256, is_stateful]
        adj_matrix : torch.Tensor, shape (N, N)
            Normalized adjacency matrix with traffic weights
        trigger_context : torch.Tensor, shape (3,)
            v3.7: [proactive_flag, reactive_flag, risk_ratio]
            - PROACTIVE: [1.0, 0.0, risk_ratio] where risk_ratio ∈ [0,1]
            - REACTIVE:  [0.0, 1.0, 1.0] (always max risk)
            
            risk_ratio captures continuous distance-to-SLA information:
            - Low risk (e.g., 0.3): plenty of time, no rush to migrate
            - High risk (e.g., 0.9): urgent, should consider migration
        
        Returns
        -------
        node_embeddings : torch.Tensor, shape (N, output_dim)
            Trigger-conditioned node embeddings
        
        Physical Interpretation of Forward Pass (v3.7 Enhanced):
        ========================================================
        
        The forward pass implements the following physics-aware computation:
        
        1. ENRICHMENT: Each node receives the global trigger context
           → Node features become trigger-aware AND risk-aware
           → v3.7: risk_ratio allows gradual JIT decision making
        
        2. PROJECTION: Features are projected to a shared latent space
           → Enables comparison across heterogeneous node types
        
        3. ATTENTION: Multi-head attention with trigger modulation
           → PROACTIVE + low risk: Very broad attention, favor stability
           → PROACTIVE + high risk: Focus shifting to migration readiness
           → REACTIVE: Always focused attention on critical nodes
        
        4. AGGREGATION: Weighted message passing along edges
           → Information flows according to traffic-weighted topology
        
        5. OUTPUT: Final embeddings encode both local features and
           trigger-conditioned global context with risk awareness
        """
        n_nodes = node_features.shape[0]
        device = node_features.device
        
        # ================================================================
        # Step 1: Trigger-Aware Feature Enrichment
        # ================================================================
        # Broadcast trigger context to all nodes: (2,) → (N, 2)
        trigger_broadcast = trigger_context.unsqueeze(0).expand(n_nodes, -1)
        
        # Concatenate: (N, 3) + (N, 2) → (N, 5)
        # This is the KEY step where trigger information enters the node representation
        enriched_features = torch.cat([node_features, trigger_broadcast], dim=1)
        
        # Project to hidden space: (N, 5) → (N, hidden_dim)
        h = self.node_projection(enriched_features)
        
        # ================================================================
        # Step 2: Multi-Head Query, Key, Value Computation
        # ================================================================
        # Q, K, V projections: (N, hidden_dim) each
        Q = self.W_query(h)
        K = self.W_key(h)
        V = self.W_value(h)
        
        # Reshape for multi-head: (N, num_heads, head_dim)
        Q = Q.view(n_nodes, self.num_heads, self.head_dim)
        K = K.view(n_nodes, self.num_heads, self.head_dim)
        V = V.view(n_nodes, self.num_heads, self.head_dim)
        
        # ================================================================
        # Step 3: Trigger-Conditioned Attention Score Computation
        # ================================================================
        # Raw attention scores: (N, N, num_heads)
        # Using einsum for clarity: "i h d, j h d -> i j h"
        attention_scores = torch.einsum('ihd,jhd->ijh', Q, K)
        
        # Scale by sqrt(head_dim) for stable gradients
        attention_scores = attention_scores / math.sqrt(self.head_dim)
        
        # (A) Trigger-based temperature scaling
        # PROACTIVE → higher temperature → softer distribution (more exploration)
        # REACTIVE → lower temperature → sharper distribution (focus on critical)
        temperature = self.trigger_temperature(trigger_context)  # (num_heads,)
        # Clamp to prevent extreme values
        temperature = temperature.clamp(min=0.1, max=5.0)
        # Apply temperature: (N, N, num_heads) / (num_heads,)
        attention_scores = attention_scores / temperature.unsqueeze(0).unsqueeze(0)
        
        # ================================================================
        # Step 4: Physics-Aware Attention Masking and Biasing
        # ================================================================
        # (B) Apply topology mask: only attend to connected nodes
        # adj_matrix provides traffic-weighted connectivity
        # Expand adj_matrix for multi-head: (N, N) → (N, N, num_heads)
        adj_expanded = adj_matrix.unsqueeze(-1).expand(-1, -1, self.num_heads)
        
        # Mask disconnected pairs (zero traffic = no edge)
        # Use large negative value instead of -inf to avoid NaN in softmax
        mask = (adj_expanded == 0)
        attention_scores = attention_scores.masked_fill(mask, -1e9)
        
        # (C) Stateful-Trigger Interaction Bias
        # This is the CRITICAL component that captures asymmetric cost
        # 
        # Physical intuition:
        # - is_reactive = 1 - trigger_context[0] = 1 if REACTIVE, 0 if PROACTIVE
        # - When REACTIVE: stateful nodes get NEGATIVE bias (avoid attending to them
        #   because migration is expensive)
        # - When PROACTIVE: stateful nodes get POSITIVE bias (safe to migrate)
        is_reactive = 1.0 - trigger_context[0]  # Scalar: 1 if REACTIVE, 0 if PROACTIVE
        
        # Extract stateful flags: (N,)
        is_stateful = node_features[:, 2]
        
        # Create stateful interaction matrix: (N, N)
        # Outer product captures pairwise stateful relationships
        stateful_matrix = torch.outer(is_stateful, is_stateful)
        
        # Apply bias: stronger effect under REACTIVE trigger
        # The sign of stateful_trigger_bias is LEARNED, allowing the network
        # to discover that REACTIVE+stateful should be avoided
        stateful_bias = stateful_matrix.unsqueeze(-1) * self.stateful_trigger_bias * is_reactive
        attention_scores = attention_scores + stateful_bias
        
        # ================================================================
        # Step 5: Attention Weight Normalization and Gating
        # ================================================================
        # Softmax over source nodes (dim=1): (N, N, num_heads)
        attention_weights = F.softmax(attention_scores, dim=1)
        
        # (D) Trigger-based attention gating
        # Modulates the overall influence of graph attention
        gate = self.trigger_gate(trigger_context)  # (hidden_dim,)
        gate_value = gate.mean()  # Scalar gating factor
        
        # Scale attention weights: allows network to learn how much to trust
        # the attention mechanism under different triggers
        attention_weights = attention_weights * gate_value
        
        # ================================================================
        # Step 6: Message Passing (Weighted Aggregation)
        # ================================================================
        # Aggregate neighbor values weighted by attention: (N, num_heads, head_dim)
        # einsum: "ijk, jhd -> ihd" (sum over j, the source nodes)
        aggregated = torch.einsum('ijh,jhd->ihd', attention_weights, V)
        
        # Concatenate heads: (N, num_heads, head_dim) → (N, hidden_dim)
        aggregated = aggregated.reshape(n_nodes, -1)
        
        # ================================================================
        # Step 7: Output Projection with Residual Connection
        # ================================================================
        # Residual connection helps gradient flow and preserves node identity
        output = self.output_projection(aggregated) + self.residual_scale * h[:, :self.output_dim]
        
        # Store for downstream retrieval
        self._node_embeddings = output
        
        # v3.0: Store attention weights and trigger for debug inspection
        self._last_attention_weights = attention_weights.detach()
        self._last_trigger_context = trigger_context.detach()
        self._last_temperature = temperature.detach()
        self._last_stateful_bias = self.stateful_trigger_bias.detach()
        
        return output
    
    def debug_print_attention(self, prefix=""):
        """Print last computed attention weights for debugging (v3.7)."""
        if not hasattr(self, '_last_attention_weights'):
            print(f"{prefix}[GAT DEBUG] No attention weights computed yet.")
            return
        
        attn = self._last_attention_weights
        trigger = self._last_trigger_context
        temp = self._last_temperature
        bias = self._last_stateful_bias
        
        trigger_type = "PROACTIVE" if trigger[0] > 0.5 else "REACTIVE"
        # v3.7: Extract risk_ratio from trigger_context[2]
        risk_ratio = trigger[2].item() if len(trigger) > 2 else "N/A"
        
        print(f"\n{prefix}[GAT v3.7 Attention Debug]")
        print(f"  Trigger: {trigger_type} ({trigger.cpu().numpy()})")
        print(f"  Risk Ratio: {risk_ratio:.4f}" if isinstance(risk_ratio, float) else f"  Risk Ratio: {risk_ratio}")
        print(f"  Temperature per head: {temp.cpu().numpy()}")
        print(f"  Stateful-Trigger Bias: {bias.cpu().numpy()}")
        print(f"  Attention shape: {attn.shape}")
        print(f"  Attention mean: {attn.mean().item():.4f}, max: {attn.max().item():.4f}")
    
    def get_node_embedding(self, node_idx):
        """Retrieve embedding for a specific node."""
        if self._node_embeddings is None:
            raise RuntimeError("Must call forward() before get_node_embedding()")
        return self._node_embeddings[node_idx]


# =============================================================================
# Module 2: SAC Actor (Policy Network)
# =============================================================================

class SACDiscreteActor(nn.Module):
    """
    Soft Actor-Critic Policy Network for Discrete Action Space.
    
    This actor outputs a categorical probability distribution over 3 actions:
    - Action 0: STAY at current server
    - Action 1: FOLLOW SA proposal
    - Action 2: Move to NEAREST server
    
    The actor takes as input:
    - Node embedding from TriggerAwareGAT (captures graph context + trigger)
    - SA priors (captures SA's recommendation for this specific node)
    
    The network learns to output probabilities that maximize expected reward
    PLUS entropy bonus (soft actor-critic objective).
    
    Parameters
    ----------
    embedding_dim : int
        Dimension of graph embedding from GAT
    sa_prior_dim : int
        Dimension of SA prior features (default: 2)
    hidden_dim : int
        Hidden layer dimension
    action_dim : int
        Number of discrete actions (default: 3)
    """
    
    def __init__(self, embedding_dim=64, sa_prior_dim=2, hidden_dim=128, action_dim=3):
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.sa_prior_dim = sa_prior_dim
        self.action_dim = action_dim
        
        input_dim = embedding_dim + sa_prior_dim  # 64 + 2 = 66
        
        # Policy network: maps state to action logits
        self.policy_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim),
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize with small weights for stable early training."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=0.01)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, node_embedding, sa_prior):
        """
        Compute action probabilities for a single node.
        
        Parameters
        ----------
        node_embedding : torch.Tensor, shape (embedding_dim,) or (batch, embedding_dim)
            Graph-contextualized node embedding
        sa_prior : torch.Tensor, shape (sa_prior_dim,) or (batch, sa_prior_dim)
            SA recommendation features
        
        Returns
        -------
        action_probs : torch.Tensor, shape (action_dim,) or (batch, action_dim)
            Probability distribution over actions
        log_probs : torch.Tensor, same shape as action_probs
            Log probabilities (for entropy computation)
        """
        # Handle both single node and batch inputs
        if node_embedding.dim() == 1:
            node_embedding = node_embedding.unsqueeze(0)
            sa_prior = sa_prior.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False
        
        # Concatenate embedding with SA prior
        state = torch.cat([node_embedding, sa_prior], dim=-1)
        
        # Compute action logits
        logits = self.policy_net(state)
        
        # Convert to probabilities via softmax
        action_probs = F.softmax(logits, dim=-1)
        
        # Compute log probabilities with numerical stability
        # Add small epsilon to avoid log(0)
        log_probs = F.log_softmax(logits, dim=-1)
        
        if squeeze_output:
            action_probs = action_probs.squeeze(0)
            log_probs = log_probs.squeeze(0)
        
        return action_probs, log_probs
    
    def sample_action(self, node_embedding, sa_prior):
        """
        Sample action using categorical distribution (for exploration).
        
        Returns
        -------
        action : int
            Sampled action index
        log_prob : torch.Tensor
            Log probability of sampled action
        """
        action_probs, log_probs = self.forward(node_embedding, sa_prior)
        
        # Create categorical distribution and sample
        dist = Categorical(probs=action_probs)
        action = dist.sample()
        
        return action.item(), log_probs[action]
    
    def get_action_deterministic(self, node_embedding, sa_prior):
        """Get deterministic action (argmax) for evaluation."""
        action_probs, _ = self.forward(node_embedding, sa_prior)
        return action_probs.argmax().item()
    
    def sample_action_with_mask(self, node_embedding, sa_prior, action_mask=None):
        """
        Sample action with optional action masking for DAG coherence (v3.1).
        
        Parameters
        ----------
        node_embedding : torch.Tensor
        sa_prior : torch.Tensor
        action_mask : torch.Tensor, shape (action_dim,), optional
            Mask where 1.0 = valid action, 0.0 = invalid action.
            Invalid actions get large negative logits before softmax.
        
        Returns
        -------
        action : int
            Sampled action index
        log_prob : torch.Tensor
            Log probability of sampled action
        """
        # Handle both single node and batch inputs
        if node_embedding.dim() == 1:
            node_embedding = node_embedding.unsqueeze(0)
            sa_prior = sa_prior.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False
        
        # Concatenate embedding with SA prior
        state = torch.cat([node_embedding, sa_prior], dim=-1)
        
        # Compute action logits
        logits = self.policy_net(state)
        
        # v3.1: Apply action masking BEFORE softmax
        # This prevents DAG tearing by penalizing actions that
        # would place nodes far from their DAG neighbors
        if action_mask is not None:
            # action_mask: 1.0 = valid, 0.0 = invalid
            # We add -1e9 to invalid action logits
            invalid_mask = (1.0 - action_mask) * (-1e9)
            logits = logits + invalid_mask.unsqueeze(0) if logits.dim() == 2 else logits + invalid_mask
        
        # Convert to probabilities via softmax
        action_probs = F.softmax(logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)
        
        if squeeze_output:
            action_probs = action_probs.squeeze(0)
            log_probs = log_probs.squeeze(0)
        
        # Create categorical distribution and sample
        dist = Categorical(probs=action_probs)
        action = dist.sample()
        
        return action.item(), log_probs[action]


# =============================================================================
# Module 3: SAC Critic (Twin Q-Networks)
# =============================================================================

class SACDiscreteCritic(nn.Module):
    """
    Twin Q-Networks for Discrete SAC.
    
    Implements the "clipped double Q-learning" trick from TD3/SAC:
    - Maintains TWO independent Q-networks (Q1 and Q2)
    - Uses min(Q1, Q2) as the value estimate to prevent overestimation
    
    For discrete actions, each Q-network outputs Q-values for ALL actions,
    unlike continuous SAC which outputs a single Q-value.
    
    Parameters
    ----------
    embedding_dim : int
        Dimension of graph embedding
    sa_prior_dim : int
        Dimension of SA prior features
    hidden_dim : int
        Hidden layer dimension
    action_dim : int
        Number of discrete actions
    """
    
    def __init__(self, embedding_dim=64, sa_prior_dim=2, hidden_dim=128, action_dim=3):
        super().__init__()
        
        input_dim = embedding_dim + sa_prior_dim
        
        # Q-Network 1
        self.q1_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim),
        )
        
        # Q-Network 2 (independent initialization)
        self.q2_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim),
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize with different seeds for diversity."""
        for i, module in enumerate(self.q1_net.modules()):
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=1.0)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        
        for i, module in enumerate(self.q2_net.modules()):
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=1.0)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, node_embedding, sa_prior):
        """
        Compute Q-values from both networks.
        
        Parameters
        ----------
        node_embedding : torch.Tensor
            Graph-contextualized node embedding
        sa_prior : torch.Tensor
            SA recommendation features
        
        Returns
        -------
        q1_values : torch.Tensor, shape (action_dim,) or (batch, action_dim)
            Q-values from network 1
        q2_values : torch.Tensor, shape (action_dim,) or (batch, action_dim)
            Q-values from network 2
        """
        # Handle both single and batch inputs
        if node_embedding.dim() == 1:
            node_embedding = node_embedding.unsqueeze(0)
            sa_prior = sa_prior.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False
        
        state = torch.cat([node_embedding, sa_prior], dim=-1)
        
        q1_values = self.q1_net(state)
        q2_values = self.q2_net(state)
        
        if squeeze_output:
            q1_values = q1_values.squeeze(0)
            q2_values = q2_values.squeeze(0)
        
        return q1_values, q2_values
    
    def q1_forward(self, node_embedding, sa_prior):
        """Get Q-values from network 1 only (for target computation)."""
        if node_embedding.dim() == 1:
            node_embedding = node_embedding.unsqueeze(0)
            sa_prior = sa_prior.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False
        
        state = torch.cat([node_embedding, sa_prior], dim=-1)
        q1_values = self.q1_net(state)
        
        if squeeze_output:
            q1_values = q1_values.squeeze(0)
        
        return q1_values


# =============================================================================
# Module 4: Discrete SAC Training Logic (optimize_sac)
# =============================================================================

def optimize_sac(
    memory,
    gat_network,
    actor,
    critic,
    target_critic,
    gat_optimizer,
    actor_optimizer,
    critic_optimizer,
    device,
    alpha=0.2,
    gamma=0.95,
    batch_size=32,
    target_entropy=None,
    log_alpha=None,
    alpha_optimizer=None,
):
    """
    Discrete SAC Optimization Step.
    
    This function implements the discrete SAC training algorithm, which extends
    the continuous SAC to discrete action spaces using the formulation from
    Christodoulou (2019).
    
    Discrete Maximum Entropy Objective:
    ===================================
    
    The key insight of SAC is to maximize BOTH expected reward AND policy entropy:
    
        J(π) = Σ_t E[r_t + α H(π(·|s_t))]
    
    where H(π) = -Σ_a π(a|s) log π(a|s) is the entropy of the policy.
    
    For discrete actions, the soft value functions become:
    
        V(s) = Σ_a π(a|s) [Q(s,a) - α log π(a|s)]
        
        Q(s,a) = r + γ V(s')
    
    How Entropy Bonus Encourages SA Draft Exploration:
    ==================================================
    
    The entropy term α * H(π) in the objective has a critical effect on
    exploration of the SA draft (action 1 = "Follow SA"):
    
    1. DIVERSITY PRESSURE:
       - Entropy bonus penalizes policies that are too deterministic
       - If policy always chooses "Stay" (action 0), entropy is low → penalty
       - This FORCES the policy to also try "Follow SA" and "Nearest"
    
    2. SOFT PREFERENCE OVER SA:
       - SA draft provides a strong prior (encoded in sa_priors input)
       - Without entropy, network might ignore SA and learn simple heuristic
       - Entropy bonus ensures network EXPLORES the SA recommendation
       - If SA is consistently good, network learns to TRUST SA (action 1)
    
    3. EXTREME PENALTY AVOIDANCE:
       - The log probability term log π(a|s) penalizes extreme probabilities
       - If π(Follow SA) → 0, then log π → -∞, causing huge penalty
       - This prevents the network from completely ignoring ANY action
       - Ensures that SA draft always has non-zero probability
    
    4. ALPHA TEMPERATURE CONTROL:
       - α controls the exploration-exploitation trade-off
       - High α: More exploration, more trust in SA proposals
       - Low α: More exploitation, follows learned Q-values
       - Automatic α tuning adjusts based on actual entropy vs target
    
    Mathematical Details:
    ====================
    
    Critic Loss (for both Q1 and Q2):
        L_critic = E[(Q(s,a) - y)²]
        where y = r + γ * (Σ_a' π(a'|s') [min(Q1_target, Q2_target)(s',a') - α log π(a'|s')])
    
    Actor Loss:
        L_actor = E[Σ_a π(a|s) [α log π(a|s) - min(Q1, Q2)(s,a)]]
    
    Alpha Loss (optional automatic tuning):
        L_alpha = E[-α (log π(a|s) + H_target)]
    
    Parameters
    ----------
    memory : deque
        Replay buffer containing (graph_state, node_idx, action, reward, next_graph_state, done)
    gat_network : TriggerAwareGAT
        Graph attention network for embedding computation
    actor : SACDiscreteActor
        Policy network
    critic : SACDiscreteCritic
        Twin Q-networks
    target_critic : SACDiscreteCritic
        Target Q-networks (for stable training)
    *_optimizer : torch.optim.Optimizer
        Optimizers for each network
    device : torch.device
        Computation device
    alpha : float
        Temperature parameter for entropy regularization
    gamma : float
        Discount factor
    batch_size : int
        Mini-batch size
    target_entropy : float, optional
        Target entropy for automatic alpha tuning
    log_alpha : torch.Tensor, optional
        Learnable log(alpha) for automatic tuning
    alpha_optimizer : torch.optim.Optimizer, optional
        Optimizer for alpha
    
    Returns
    -------
    dict with keys:
        'critic_loss': float
        'actor_loss': float
        'alpha': float (current temperature)
        'entropy': float (policy entropy)
    """
    if len(memory) < batch_size:
        return None
    
    # Sample mini-batch from replay buffer
    batch = random.sample(memory, batch_size)
    
    critic_losses = []
    actor_losses = []
    entropies = []
    alpha_losses = []
    
    # Zero gradients
    gat_optimizer.zero_grad()
    actor_optimizer.zero_grad()
    critic_optimizer.zero_grad()
    
    total_critic_loss = torch.tensor(0.0, device=device)
    total_actor_loss = torch.tensor(0.0, device=device)
    total_alpha_loss = torch.tensor(0.0, device=device)
    
    for (graph_state, node_idx, action, reward, next_graph_state, done) in batch:
        # =================================================================
        # Step 1: Compute Current State Embeddings
        # =================================================================
        node_feat = torch.FloatTensor(graph_state['node_features']).to(device)
        adj = torch.FloatTensor(graph_state['adj_matrix']).to(device)
        trigger = torch.FloatTensor(graph_state['trigger_context']).to(device)
        sa_prior = torch.FloatTensor(graph_state['sa_priors']).to(device)
        
        # Get embeddings from GAT
        embeddings = gat_network(node_feat, adj, trigger)
        node_emb = embeddings[node_idx]
        node_sa_prior = sa_prior[node_idx]
        
        # =================================================================
        # Step 2: Compute Target Value (for Critic Update)
        # =================================================================
        with torch.no_grad():
            if done:
                target_value = torch.tensor(reward, dtype=torch.float32, device=device)
            else:
                # Get next state embeddings
                next_node_feat = torch.FloatTensor(next_graph_state['node_features']).to(device)
                next_adj = torch.FloatTensor(next_graph_state['adj_matrix']).to(device)
                next_trigger = torch.FloatTensor(next_graph_state['trigger_context']).to(device)
                next_sa_prior = torch.FloatTensor(next_graph_state['sa_priors']).to(device)
                
                next_embeddings = gat_network(next_node_feat, next_adj, next_trigger)
                next_node_emb = next_embeddings[node_idx]
                next_node_sa_prior = next_sa_prior[node_idx]
                
                # Get next action probabilities from current actor
                next_action_probs, next_log_probs = actor(next_node_emb, next_node_sa_prior)
                
                # Get target Q-values
                target_q1, target_q2 = target_critic(next_node_emb, next_node_sa_prior)
                target_q_min = torch.min(target_q1, target_q2)
                
                # Compute soft value: V(s') = Σ_a π(a|s') [Q(s',a) - α log π(a|s')]
                # This is the KEY discrete SAC formula
                next_value = (next_action_probs * (target_q_min - alpha * next_log_probs)).sum()
                
                # TD target: y = r + γ V(s')
                target_value = torch.tensor(reward, dtype=torch.float32, device=device) + gamma * next_value
        
        # =================================================================
        # Step 3: Critic Loss
        # =================================================================
        # Get current Q-values
        q1, q2 = critic(node_emb, node_sa_prior)
        
        # Select Q-value for taken action
        q1_action = q1[action]
        q2_action = q2[action]
        
        # v3.9: 使用 Huber Loss (Smooth L1) 抑制极端误差的梯度爆炸
        critic_loss = F.smooth_l1_loss(q1_action, target_value) + F.smooth_l1_loss(q2_action, target_value)
        total_critic_loss = total_critic_loss + critic_loss
        critic_losses.append(critic_loss.item())
        
        # =================================================================
        # Step 4: Actor Loss
        # =================================================================
        # Get action probabilities (need gradients here)
        # v3.1 FIX: Removed .detach() to allow gradients to flow back to GAT
        # This enables stateful_trigger_bias to be updated through actor loss
        action_probs, log_probs = actor(node_emb, node_sa_prior)
        
        # Get Q-values (detached to not affect actor gradient)
        with torch.no_grad():
            q1_detached, q2_detached = critic(node_emb, node_sa_prior)
            q_min = torch.min(q1_detached, q2_detached)
        
        # Actor loss: Σ_a π(a|s) [α log π(a|s) - Q(s,a)]
        # We want to MINIMIZE this, which MAXIMIZES Q - α log π (i.e., reward + entropy)
        actor_loss = (action_probs * (alpha * log_probs - q_min)).sum()
        total_actor_loss = total_actor_loss + actor_loss
        actor_losses.append(actor_loss.item())
        
        # Compute entropy for logging: H = -Σ_a π(a) log π(a)
        entropy = -(action_probs * log_probs).sum()
        entropies.append(entropy.item())
        
        # =================================================================
        # Step 5: Alpha Loss (Optional Automatic Tuning)
        # =================================================================
        if log_alpha is not None and target_entropy is not None:
            # Alpha loss: -α (log π(a) + H_target)
            # This increases α when entropy is below target (need more exploration)
            # and decreases α when entropy is above target (can exploit more)
            alpha_loss = -(log_alpha.exp() * (log_probs[action].detach() + target_entropy))
            total_alpha_loss = total_alpha_loss + alpha_loss
            alpha_losses.append(alpha_loss.item())
    
    # =================================================================
    # Step 6: Backpropagation
    # =================================================================
    # Average losses over batch
    avg_critic_loss = total_critic_loss / batch_size
    avg_actor_loss = total_actor_loss / batch_size
    
    # Critic update
    avg_critic_loss.backward(retain_graph=True)
    torch.nn.utils.clip_grad_norm_(critic.parameters(), max_norm=1.0)
    critic_optimizer.step()
    
    # Actor update (also updates GAT through backprop)
    gat_optimizer.zero_grad()
    actor_optimizer.zero_grad()
    avg_actor_loss.backward()
    torch.nn.utils.clip_grad_norm_(actor.parameters(), max_norm=1.0)
    torch.nn.utils.clip_grad_norm_(gat_network.parameters(), max_norm=1.0)
    actor_optimizer.step()
    gat_optimizer.step()
    
    # Alpha update (if enabled)
    current_alpha = alpha
    if log_alpha is not None and alpha_optimizer is not None and len(alpha_losses) > 0:
        alpha_optimizer.zero_grad()
        avg_alpha_loss = total_alpha_loss / batch_size
        avg_alpha_loss.backward()
        alpha_optimizer.step()
        current_alpha = log_alpha.exp().item()
    
    return {
        'critic_loss': np.mean(critic_losses),
        'actor_loss': np.mean(actor_losses),
        'alpha': current_alpha,
        'entropy': np.mean(entropies),
    }


def soft_update(target_net, source_net, tau):
    """
    Polyak averaging (soft update) for target network.
    
    θ_target = τ * θ_source + (1 - τ) * θ_target
    
    Parameters
    ----------
    target_net : nn.Module
        Target network to update
    source_net : nn.Module
        Source network to copy from
    tau : float
        Interpolation factor (typically 0.005)
    """
    for target_param, source_param in zip(target_net.parameters(), source_net.parameters()):
        target_param.data.copy_(tau * source_param.data + (1.0 - tau) * target_param.data)


# =============================================================================
# Module 5: Main Simulation Loop (run_hybrid_sac_microservice)
# =============================================================================

def run_hybrid_sac_microservice(
    df, servers_df, predictor=None, proactive=False, num_epochs=6,
    inference_mode=False,           # 新增：是否为推理模式
    checkpoint_path=None,           # 新增：权重文件路径（推理时加载）
    save_checkpoint_path=None,      # 新增：训练后保存路径
):
    """
    Hybrid SAC-Refined SA Microservice Migration with Trigger-Aware Graph Attention.
    
    This function implements the main simulation loop for the Discrete SAC algorithm
    with Trigger-Conditioned GAT for microservice migration decisions.
    
    Algorithm Overview:
    ==================
    
    For each taxi at each timestep:
    1. Check if migration is triggered (PROACTIVE or REACTIVE)
    2. If triggered:
       a. Run Simulated Annealing to get global draft proposal
       b. Build graph state with trigger context
       c. Compute graph embeddings via TriggerAwareGAT
       d. For each microservice (in topological order):
          - Sample action from Actor's categorical distribution
          - Execute action (Stay / Follow SA / Nearest)
       e. Compute reward and store transitions in replay buffer
       f. Optimize SAC networks
       g. Soft-update target critic
    
    Key Improvements (v2.0 - Ultimate Tuning):
    ==========================================
    
    1. DENSE REWARD: All nodes in a DAG share the same total reward (credit assignment fix)
    2. BEHAVIOR CLONING WARMUP: First N steps force FOLLOW SA to learn baseline
    3. MULTI-EPOCH TRAINING: Dataset is iterated multiple times for better convergence
    
    Parameters
    ----------
    df : pd.DataFrame
        Taxi trajectory data with columns [taxi_id, date_time, latitude, longitude]
    servers_df : pd.DataFrame
        Edge server data with columns [server_id, latitude, longitude]
    predictor : SimpleTrajectoryPredictor, optional
        Trajectory prediction model for proactive migration
    proactive : bool
        Whether to enable proactive (trajectory-based) migration triggers
    num_epochs : int
        Total passes over the timeline. The **last** epoch is a pure evaluation
        phase (no BC, deterministic actor, no replay / no optimize_sac / no soft
        update). Earlier epochs are training. Default 6 = 5 train + 1 eval.
    inference_mode : bool
        True: 跳过训练，只运行 1 个推理 epoch
        False: 正常训练 + 评估
    checkpoint_path : str
        推理模式下，从此路径加载权重
    save_checkpoint_path : str
        训练模式下，训练完成后保存权重到此路径
    
    Returns
    -------
    dict with keys:
        total_migrations : int
        total_violations : int
        proactive_decisions : int
        decision_count : int
        total_reward : float
        total_access_latency : float
        total_communication_cost : float
        total_migration_cost : float
        loss_history : list
        reward_history : list
        entropy_history : list
        total_decision_time : float (推理/评估时延总和)
        decision_count_for_latency : int (时延计数)
        avg_decision_time_ms : float (平均决策时延，毫秒)
    """
    servers_info = build_servers_info(servers_df)
    use_proactive = proactive and predictor is not None
    
    # Initialize tracking variables
    taxi_dag_type = {}
    taxi_dag_assignments = {}
    total_migrations = 0
    total_violations = 0
    proactive_decisions = 0
    total_reward_sum = 0.0
    
    total_access_latency = 0.0
    total_communication_cost = 0.0
    total_migration_cost = 0.0
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}  |  Proactive: {use_proactive}  |  Model: TriggerAwareGAT + Discrete SAC")
    print(f"  Multi-Epoch: {num_epochs} epochs (= {num_epochs - 1} train + 1 eval)")
    
    # =========================================================================
    # v3.9: Mode-Specific Curriculum Learning
    # =========================================================================
    # v3.8 Problem: Reactive 模式在 15x 死亡惩罚下完全瘫痪（0 迁移）
    # 
    # v3.9 Solution: 为 Reactive 模式设置更慢的 BC 衰减
    # - Proactive: 原有 schedule，快速过渡到自主决策
    # - Reactive: 更慢衰减，给模型更多时间学习 SA 的"带伤迁移"策略
    #
    # Proactive Schedule: [0.95, 0.85, 0.65, 0.35, 0.10]
    # Reactive Schedule:  [0.98, 0.90, 0.75, 0.55, 0.30] (更保守)
    # =========================================================================
    if proactive:
        bc_prob_schedule = [0.95, 0.85, 0.65, 0.35, 0.10]  # Proactive: 原有 schedule
    else:
        bc_prob_schedule = [0.98, 0.90, 0.75, 0.55, 0.30]  # Reactive: 更慢衰减
    
    print(f"  [v3.7] JIT Migration + 3D Trigger Context (risk_ratio):")
    print(f"    - Train epochs: {num_epochs - 1}  |  Eval epoch: 1 (last)")
    print(f"    - BC probability schedule (training): {bc_prob_schedule}")
    print(f"    - Eval: BC=0, argmax policy, no memory / no optimize_sac / no soft-update")
    print(f"    - v3.7 NEW: trigger_context now 3D [proactive, reactive, risk_ratio]")
    print(f"    - v3.7 NEW: Dynamic migration discount based on risk_ratio")
    
    bc_actions_taken = 0       # Counter for BC (SA imitation) actions
    explore_actions_taken = 0  # Counter for exploration actions
    
    # Debug counters for proactive trigger analysis
    _debug_reactive_triggers = 0
    _debug_proactive_triggers = 0
    _debug_no_triggers = 0
    _debug_predictions_made = 0
    
    # =========================================================================
    # Network Initialization
    # =========================================================================
    
    # Hyperparameters
    hidden_dim = 64
    embedding_dim = 64
    action_dim = 3
    learning_rate = 3e-4
    # v3.3: Moderate exploration temperature for Soft BC
    # With gradual probability decay, we can use a balanced alpha
    # that allows entropy-driven exploration when BC probability is low
    alpha_init = 0.05
    gamma = 0.95
    tau = 0.005
    batch_size = 32
    memory_size = 10000
    
    # Target entropy for automatic alpha tuning
    # For discrete actions: -log(1/|A|) * ratio = log(|A|) * ratio
    # v3.4: Reduced from 0.98 to 0.90 for more deterministic policy
    # Lower target = less exploration = fewer unnecessary migrations
    target_entropy = -np.log(1.0 / action_dim) * 0.90  # ~0.90 * log(3) ≈ 0.99
    
    # Initialize networks
    # v3.7: trigger_dim upgraded from 2 to 3 for continuous risk feature
    # trigger_context now contains [proactive_flag, reactive_flag, risk_ratio]
    gat_network = TriggerAwareGAT(
        node_feat_dim=3,
        trigger_dim=3,  # v3.7: 3D trigger context with risk_ratio
        hidden_dim=hidden_dim,
        num_heads=2,
        output_dim=embedding_dim,
        dropout=0.1,
    ).to(device)
    
    actor = SACDiscreteActor(
        embedding_dim=embedding_dim,
        sa_prior_dim=2,
        hidden_dim=128,
        action_dim=action_dim,
    ).to(device)
    
    critic = SACDiscreteCritic(
        embedding_dim=embedding_dim,
        sa_prior_dim=2,
        hidden_dim=128,
        action_dim=action_dim,
    ).to(device)
    
    target_critic = SACDiscreteCritic(
        embedding_dim=embedding_dim,
        sa_prior_dim=2,
        hidden_dim=128,
        action_dim=action_dim,
    ).to(device)
    
    # Copy weights to target network
    target_critic.load_state_dict(critic.state_dict())
    
    # Learnable alpha (temperature)
    log_alpha = torch.tensor(np.log(alpha_init), dtype=torch.float32, device=device, requires_grad=True)
    
    # Optimizers
    gat_optimizer = optim.Adam(gat_network.parameters(), lr=learning_rate)
    actor_optimizer = optim.Adam(actor.parameters(), lr=learning_rate)
    critic_optimizer = optim.Adam(critic.parameters(), lr=learning_rate)
    alpha_optimizer = optim.Adam([log_alpha], lr=learning_rate)
    
    # Replay buffer
    memory = deque(maxlen=memory_size)
    
    # Logging
    loss_history = []
    reward_history = []
    entropy_history = []
    alpha_history = []
    
    # 时延探针初始化
    total_decision_time = 0.0
    decision_count_for_latency = 0
    
    # === 推理模式：加载权重，跳过训练 ===
    if inference_mode:
        if checkpoint_path is None:
            raise ValueError("inference_mode=True 但未提供 checkpoint_path")
        log_alpha = load_sac_weights(checkpoint_path, gat_network, actor, critic, target_critic, device)
        num_epochs = 1  # 只运行 1 个推理 epoch
        print(f"  [INFERENCE MODE] Loaded weights, running 1 eval epoch only")
        
        # ⚠️ 关键：推理模式下强制关闭 BC（行为克隆）
        # 覆盖 bc_prob_schedule，确保不会执行 SA 模仿
        bc_prob_schedule = [0.0]  # 只有 1 个 epoch，BC 概率为 0
    
    # =========================================================================
    # Main Simulation Loop (Multi-Epoch)
    # =========================================================================
    
    timestamps = sorted(df['date_time'].unique())
    df_grouped = df.groupby('date_time')
    
    decision_count = 0
    global_step = 0  # Tracks total steps across all epochs
    
    for epoch in range(num_epochs):
        print(f"\n  === Epoch {epoch + 1}/{num_epochs} ===")
        
        is_eval_epoch = (epoch == num_epochs - 1)
        
        if is_eval_epoch:
            print("  [EVAL] Pure evaluation epoch — reset reported metrics; "
                  "BC off; argmax actions; no replay / no optimize / no soft-update")
            total_violations = 0
            total_migrations = 0
            decision_count = 0
            proactive_decisions = 0
            total_reward_sum = 0.0
            total_access_latency = 0.0
            total_communication_cost = 0.0
            total_migration_cost = 0.0
            gat_network.eval()
            actor.eval()
            critic.eval()
            target_critic.eval()
        else:
            gat_network.train()
            actor.train()
            critic.train()
            target_critic.train()
        
        epoch_start_decisions = decision_count
        epoch_start_migrations = total_migrations
        epoch_start_violations = total_violations
        epoch_start_reward = total_reward_sum
        
        # Reset per-epoch tracking (but keep networks and replay buffer!)
        epoch_migrations = 0
        epoch_violations = 0
        epoch_reward = 0.0
        
        pbar = tqdm(
            total=len(timestamps),
            desc=f"Epoch {epoch+1}/{num_epochs}{' [EVAL]' if is_eval_epoch else ''}",
        )
        
        for timestamp in timestamps:
            current_rows = df_grouped.get_group(timestamp)
            
            for _, row in current_rows.iterrows():
                taxi_id = row['taxi_id']
                current_lat = row['latitude']
                current_lon = row['longitude']
                
                # -------------------------------------------------------------
                # Initialize new taxi with nearest server assignment
                # -------------------------------------------------------------
                if taxi_id not in taxi_dag_assignments:
                    nearest = find_k_nearest_servers(current_lat, current_lon, servers_df, k=1)[0]
                    chosen_dag = assign_dag_type()
                    taxi_dag_type[taxi_id] = chosen_dag
                    taxi_dag_assignments[taxi_id] = initialize_dag_assignment(chosen_dag, nearest[0])
                    continue
                
                dag_type = taxi_dag_type[taxi_id]
                dag_info = MICROSERVICE_DAGS[dag_type]
                entry_nodes = get_entry_nodes(dag_info)
                gateway_node = entry_nodes[0]
                
                # -------------------------------------------------------------
                # Check current gateway distance and violation
                # -------------------------------------------------------------
                gateway_server_id = taxi_dag_assignments[taxi_id][gateway_node]
                gw_lat, gw_lon = servers_info[gateway_server_id]
                gateway_dist = haversine_distance(current_lat, current_lon, gw_lat, gw_lon)
                
                if gateway_dist > 15.0:
                    total_violations += 1
                
                # Compute current DAG reward
                current_dag_reward, _ = calculate_microservice_reward(
                    taxi_id, dag_info,
                    taxi_dag_assignments[taxi_id], taxi_dag_assignments[taxi_id],
                    (current_lat, current_lon), servers_info,
                )
                
                # -------------------------------------------------------------
                # Trajectory prediction (if proactive mode)
                # -------------------------------------------------------------
                predicted_locations = None
                if use_proactive:
                    raw = predictor.predict_future(
                        current_lon, current_lat, taxi_id, steps=FORECAST_HORIZON
                    )
                    predicted_locations = [(lat, lon) for lon, lat in raw]
                    _debug_predictions_made += 1
                
                # -------------------------------------------------------------
                # Get trigger type
                # -------------------------------------------------------------
                trigger_type = get_trigger_type(
                    current_lat, current_lon, gw_lat, gw_lon,
                    current_dag_reward, predicted_locations,
                    proactive_enabled=use_proactive,
                )
                
                if trigger_type is None:
                    _debug_no_triggers += 1
                    continue
                
                decision_count += 1
                if trigger_type == TRIGGER_PROACTIVE:
                    proactive_decisions += 1
                    _debug_proactive_triggers += 1
                elif trigger_type == TRIGGER_REACTIVE:
                    _debug_reactive_triggers += 1
                
                # -------------------------------------------------------------
                # Phase A: Simulated Annealing Global Draft
                # -------------------------------------------------------------
                candidates = find_k_nearest_servers(current_lat, current_lon, servers_df, k=3)
                old_assignments = copy.copy(taxi_dag_assignments[taxi_id])
                
                sa_proposal, _sa_cost = microservice_simulated_annealing(
                    taxi_id, dag_info,
                    taxi_dag_assignments[taxi_id],
                    candidates,
                    user_location=(current_lat, current_lon),
                    servers_info=servers_info,
                    previous_assignments=old_assignments,
                    predicted_locations=predicted_locations,
                    trigger_type=trigger_type,
                )
                
                # -------------------------------------------------------------
                # Phase B: Build Graph State and Compute Embeddings
                # -------------------------------------------------------------
                graph_state = build_graph_state(
                    taxi_id, dag_info,
                    taxi_dag_assignments[taxi_id],
                    servers_info, trigger_type,
                    sa_proposal, candidates,
                    current_lat, current_lon,
                )
                
                # Convert to tensors
                node_feat_t = torch.FloatTensor(graph_state['node_features']).to(device)
                adj_t = torch.FloatTensor(graph_state['adj_matrix']).to(device)
                trigger_t = torch.FloatTensor(graph_state['trigger_context']).to(device)
                sa_prior_t = torch.FloatTensor(graph_state['sa_priors']).to(device)
                
                # Compute graph embeddings
                with torch.no_grad():
                    embeddings = gat_network(node_feat_t, adj_t, trigger_t)
                
                # -------------------------------------------------------------
                # Phase C: Per-Node Decisions (Topological Order)
                # v3.3: Soft BC with probability decay (curriculum learning)
                # -------------------------------------------------------------
                sorted_nodes = topological_sort(dag_info)
                node_to_idx = {name: i for i, name in enumerate(graph_state['node_names'])}
                node_transitions = []
                nearest_server_id = candidates[0][0]
                
                # BC probability: training uses schedule; eval/inference forces 0 (no imitation)
                if is_eval_epoch or inference_mode:
                    current_bc_prob = 0.0
                else:
                    current_bc_prob = (
                        bc_prob_schedule[epoch] if epoch < len(bc_prob_schedule) else 0.0
                    )
                
                # ⚠️ 评估/推理模式：使用时延探针包裹整个决策过程
                if is_eval_epoch or inference_mode:
                    t_start = time.perf_counter()
                    
                    # === 纯决策时间开始（包含 GAT + 所有节点的 Actor 推理）===
                    with torch.no_grad():
                        # 1. GAT 前向传播（已在上面完成，这里重新计算以确保计时准确）
                        embeddings = gat_network(node_feat_t, adj_t, trigger_t)
                        
                        # 2. 所有节点的 Actor 决策
                        for ms_node in sorted_nodes:
                            node_idx = node_to_idx[ms_node]
                            node_emb = embeddings[node_idx]
                            node_sa_prior = sa_prior_t[node_idx]
                            sa_proposed_server = sa_proposal[ms_node]
                            
                            # 使用 deterministic 动作（argmax）
                            action = actor.get_action_deterministic(node_emb, node_sa_prior)
                            
                            # 执行动作（更新 taxi_dag_assignments）
                            current_node_server = taxi_dag_assignments[taxi_id][ms_node]
                            if action == 0:  # STAY
                                target_server = current_node_server
                            elif action == 1:  # FOLLOW SA
                                target_server = sa_proposed_server
                            else:  # NEAREST
                                target_server = nearest_server_id
                            
                            taxi_dag_assignments[taxi_id][ms_node] = target_server
                            node_transitions.append((node_idx, action))
                    # === 纯决策时间结束 ===
                    
                    t_end = time.perf_counter()
                    total_decision_time += (t_end - t_start)
                    decision_count_for_latency += 1
                else:
                    # 训练模式：原有逻辑
                    for ms_node in sorted_nodes:
                        node_idx = node_to_idx[ms_node]
                        sa_proposed_server = sa_proposal[ms_node]
                        
                        # Get node embedding and SA prior
                        node_emb = embeddings[node_idx]
                        node_sa_prior = sa_prior_t[node_idx]
                        
                        # ---------------------------------------------------------
                        # v3.3: Soft BC with Probability Decay
                        # ---------------------------------------------------------
                        with torch.no_grad():
                            if random.random() < current_bc_prob:
                                # Imitation: Follow SA recommendation
                                action = 1
                                bc_actions_taken += 1
                            else:
                                # Exploration: Sample from learned policy
                                action, _ = actor.sample_action(node_emb, node_sa_prior)
                                explore_actions_taken += 1
                        
                        # Execute action
                        current_node_server = taxi_dag_assignments[taxi_id][ms_node]
                        if action == 0:  # STAY
                            target_server = current_node_server
                        elif action == 1:  # FOLLOW SA
                            target_server = sa_proposed_server
                        else:  # NEAREST
                            target_server = nearest_server_id
                        
                        taxi_dag_assignments[taxi_id][ms_node] = target_server
                        node_transitions.append((node_idx, action))
                        global_step += 1
                
                # -------------------------------------------------------------
                # Phase D: Compute Reward
                # -------------------------------------------------------------
                reward, details = calculate_microservice_reward(
                    taxi_id, dag_info,
                    taxi_dag_assignments[taxi_id], old_assignments,
                    (current_lat, current_lon), servers_info,
                    predicted_locations=predicted_locations,
                    trigger_type=trigger_type,
                )
                total_reward_sum += reward
                reward_history.append(reward)
                
                total_access_latency += details['access_latency']
                total_communication_cost += details['communication_cost']
                total_migration_cost += details['migration_cost']
                
                nodes_migrated = sum(
                    1 for n in sorted_nodes
                    if old_assignments[n] != taxi_dag_assignments[taxi_id][n]
                )
                total_migrations += nodes_migrated
                
                # -------------------------------------------------------------
                # Phase E: Build Next State and Store Transitions
                # -------------------------------------------------------------
                next_graph_state = build_graph_state(
                    taxi_id, dag_info,
                    taxi_dag_assignments[taxi_id],
                    servers_info, trigger_type,
                    sa_proposal, candidates,
                    current_lat, current_lon,
                )
                
                # Store transitions in replay buffer
                # =====================================================
                # DENSE REWARD FIX (v2.0):
                # All nodes in the DAG share the same total reward.
                # This solves the credit assignment problem where only
                # the last node received reward while others got 0.
                # "一荣俱荣，一损俱损" - shared fate for all nodes
                # =====================================================
                num_nodes = len(node_transitions)
                per_node_reward = reward / num_nodes  # Fair share of total reward
                
                if not is_eval_epoch:
                    for i, (node_idx, action) in enumerate(node_transitions):
                        is_last = (i == len(node_transitions) - 1)
                        
                        memory.append((
                            graph_state,
                            node_idx,
                            action,
                            per_node_reward,  # Dense reward: each node gets fair share
                            next_graph_state,
                            is_last,  # Only mark last node as done
                        ))
                    
                    # -------------------------------------------------------------
                    # Phase F: Optimize SAC Networks
                    # -------------------------------------------------------------
                    train_info = optimize_sac(
                        memory,
                        gat_network,
                        actor,
                        critic,
                        target_critic,
                        gat_optimizer,
                        actor_optimizer,
                        critic_optimizer,
                        device,
                        alpha=log_alpha.exp().item(),
                        gamma=gamma,
                        batch_size=batch_size,
                        target_entropy=target_entropy,
                        log_alpha=log_alpha,
                        alpha_optimizer=alpha_optimizer,
                    )
                    
                    if train_info is not None:
                        loss_history.append(train_info['critic_loss'] + train_info['actor_loss'])
                        entropy_history.append(train_info['entropy'])
                        alpha_history.append(train_info['alpha'])
                    
                    # -------------------------------------------------------------
                    # Phase G: Soft Update Target Critic (Every Step)
                    # -------------------------------------------------------------
                    soft_update(target_critic, critic, tau)
            
            pbar.update(1)
        pbar.close()
        
        # Epoch summary (deltas; eval epoch totals == reported final metrics)
        ed = decision_count - epoch_start_decisions
        em = total_migrations - epoch_start_migrations
        ev = total_violations - epoch_start_violations
        er = total_reward_sum - epoch_start_reward
        tag = " [EVAL — reported to leaderboard]" if is_eval_epoch else ""
        print(
            f"  Epoch {epoch+1} completed: decisions={ed}, migrations={em}, "
            f"violations={ev}, reward={er:.2f}{tag}"
        )
    
    # Debug output for proactive trigger analysis (after all epochs)
    print(f"\n  [DEBUG] Summary (after {num_epochs} epochs = {num_epochs - 1} train + 1 eval):")
    print(f"    - Total epochs: {num_epochs} (last epoch = deterministic eval only)")
    print(f"    - BC prob schedule (training only): {bc_prob_schedule}")
    print(f"    - BC actions (SA imitation): {bc_actions_taken}")
    print(f"    - Explore actions (Actor sampling): {explore_actions_taken}")
    print(f"    - Total global steps: {global_step}")
    print(f"    - Predictions made: {_debug_predictions_made}")
    print(f"    - REACTIVE triggers: {_debug_reactive_triggers}")
    print(f"    - PROACTIVE triggers: {_debug_proactive_triggers}")
    print(f"    - No triggers (skipped): {_debug_no_triggers}")
    print(f"    - Total decisions: {decision_count}")
    
    # v3.0: Print final GAT attention weights to verify physics-aware learning
    gat_network.debug_print_attention(prefix="  ")
    
    # === 训练模式：保存权重 ===
    if not inference_mode and save_checkpoint_path:
        save_sac_weights(save_checkpoint_path, gat_network, actor, critic, target_critic, log_alpha)
    
    # =========================================================================
    # Return Results
    # =========================================================================
    return {
        # v3.6: These headline metrics reflect **only** the final eval epoch
        # (deterministic argmax, no BC, no training updates).
        'total_migrations': total_migrations,
        'total_violations': total_violations,
        'proactive_decisions': proactive_decisions,
        'decision_count': decision_count,
        'train_epochs': num_epochs - 1 if not inference_mode else 0,
        'eval_epochs': 1,
        'eval_deterministic_argmax': True,
        'total_reward': total_reward_sum,
        'total_access_latency': total_access_latency,
        'total_communication_cost': total_communication_cost,
        'total_migration_cost': total_migration_cost,
        'loss_history': loss_history,
        'reward_history': reward_history,
        'entropy_history': entropy_history,
        'alpha_history': alpha_history,
        # 时延信息
        'total_decision_time': total_decision_time,
        'decision_count_for_latency': decision_count_for_latency,
        'avg_decision_time_ms': (total_decision_time / decision_count_for_latency * 1000) if decision_count_for_latency > 0 else 0,
    }


# =============================================================================
# Evaluation Function (Optional)
# =============================================================================

def save_sac_weights(filepath, gat_network, actor, critic, target_critic, log_alpha):
    """
    保存 SAC 所有网络权重到单个 .pth 文件。
    
    Parameters
    ----------
    filepath : str
        保存路径，如 "checkpoints/sac_proactive.pth"
    gat_network : TriggerAwareGAT
    actor : SACDiscreteActor
    critic : SACDiscreteCritic
    target_critic : SACDiscreteCritic
    log_alpha : torch.Tensor
    """
    import os
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    checkpoint = {
        'gat_network': gat_network.state_dict(),
        'actor': actor.state_dict(),
        'critic': critic.state_dict(),
        'target_critic': target_critic.state_dict(),
        'log_alpha': log_alpha.detach().cpu(),
    }
    torch.save(checkpoint, filepath)
    print(f"  [SAVE] Weights saved to {filepath}")


def load_sac_weights(filepath, gat_network, actor, critic, target_critic, device):
    """
    从 .pth 文件加载 SAC 网络权重。
    
    Returns
    -------
    log_alpha : torch.Tensor
        加载的 alpha 参数
    """
    checkpoint = torch.load(filepath, map_location=device)
    
    gat_network.load_state_dict(checkpoint['gat_network'])
    actor.load_state_dict(checkpoint['actor'])
    critic.load_state_dict(checkpoint['critic'])
    target_critic.load_state_dict(checkpoint['target_critic'])
    log_alpha = checkpoint['log_alpha'].to(device)
    
    print(f"  [LOAD] Weights loaded from {filepath}")
    return log_alpha


def evaluate_sac_policy(df, servers_df, gat_network, actor, predictor=None, proactive=False):
    """
    Evaluate trained SAC policy without exploration (deterministic actions).
    
    This function runs the same simulation loop but uses argmax actions
    instead of sampling, providing a deterministic evaluation.
    """
    gat_network.eval()
    actor.eval()
    
    servers_info = build_servers_info(servers_df)
    use_proactive = proactive and predictor is not None
    device = next(gat_network.parameters()).device
    
    taxi_dag_type = {}
    taxi_dag_assignments = {}
    total_migrations = 0
    total_violations = 0
    total_reward_sum = 0.0
    
    timestamps = sorted(df['date_time'].unique())
    df_grouped = df.groupby('date_time')
    
    for timestamp in timestamps:
        current_rows = df_grouped.get_group(timestamp)
        
        for _, row in current_rows.iterrows():
            taxi_id = row['taxi_id']
            current_lat = row['latitude']
            current_lon = row['longitude']
            
            if taxi_id not in taxi_dag_assignments:
                nearest = find_k_nearest_servers(current_lat, current_lon, servers_df, k=1)[0]
                chosen_dag = assign_dag_type()
                taxi_dag_type[taxi_id] = chosen_dag
                taxi_dag_assignments[taxi_id] = initialize_dag_assignment(chosen_dag, nearest[0])
                continue
            
            dag_type = taxi_dag_type[taxi_id]
            dag_info = MICROSERVICE_DAGS[dag_type]
            entry_nodes = get_entry_nodes(dag_info)
            gateway_node = entry_nodes[0]
            
            gateway_server_id = taxi_dag_assignments[taxi_id][gateway_node]
            gw_lat, gw_lon = servers_info[gateway_server_id]
            gateway_dist = haversine_distance(current_lat, current_lon, gw_lat, gw_lon)
            
            if gateway_dist > 15.0:
                total_violations += 1
            
            current_dag_reward, _ = calculate_microservice_reward(
                taxi_id, dag_info,
                taxi_dag_assignments[taxi_id], taxi_dag_assignments[taxi_id],
                (current_lat, current_lon), servers_info,
            )
            
            predicted_locations = None
            if use_proactive:
                raw = predictor.predict_future(
                    current_lon, current_lat, taxi_id, steps=FORECAST_HORIZON
                )
                predicted_locations = [(lat, lon) for lon, lat in raw]
            
            trigger_type = get_trigger_type(
                current_lat, current_lon, gw_lat, gw_lon,
                current_dag_reward, predicted_locations,
                proactive_enabled=use_proactive,
            )
            
            if trigger_type is None:
                continue
            
            candidates = find_k_nearest_servers(current_lat, current_lon, servers_df, k=3)
            old_assignments = copy.copy(taxi_dag_assignments[taxi_id])
            
            sa_proposal, _ = microservice_simulated_annealing(
                taxi_id, dag_info,
                taxi_dag_assignments[taxi_id],
                candidates,
                user_location=(current_lat, current_lon),
                servers_info=servers_info,
                previous_assignments=old_assignments,
                predicted_locations=predicted_locations,
                trigger_type=trigger_type,
            )
            
            graph_state = build_graph_state(
                taxi_id, dag_info,
                taxi_dag_assignments[taxi_id],
                servers_info, trigger_type,
                sa_proposal, candidates,
                current_lat, current_lon,
            )
            
            node_feat_t = torch.FloatTensor(graph_state['node_features']).to(device)
            adj_t = torch.FloatTensor(graph_state['adj_matrix']).to(device)
            trigger_t = torch.FloatTensor(graph_state['trigger_context']).to(device)
            sa_prior_t = torch.FloatTensor(graph_state['sa_priors']).to(device)
            
            with torch.no_grad():
                embeddings = gat_network(node_feat_t, adj_t, trigger_t)
            
            sorted_nodes = topological_sort(dag_info)
            node_to_idx = {name: i for i, name in enumerate(graph_state['node_names'])}
            nearest_server_id = candidates[0][0]
            
            for ms_node in sorted_nodes:
                node_idx = node_to_idx[ms_node]
                sa_proposed_server = sa_proposal[ms_node]
                
                node_emb = embeddings[node_idx]
                node_sa_prior = sa_prior_t[node_idx]
                
                # Deterministic action selection
                with torch.no_grad():
                    action = actor.get_action_deterministic(node_emb, node_sa_prior)
                
                current_node_server = taxi_dag_assignments[taxi_id][ms_node]
                if action == 0:
                    target_server = current_node_server
                elif action == 1:
                    target_server = sa_proposed_server
                else:
                    target_server = nearest_server_id
                
                taxi_dag_assignments[taxi_id][ms_node] = target_server
            
            reward, _ = calculate_microservice_reward(
                taxi_id, dag_info,
                taxi_dag_assignments[taxi_id], old_assignments,
                (current_lat, current_lon), servers_info,
                predicted_locations=predicted_locations,
                trigger_type=trigger_type,
            )
            total_reward_sum += reward
            
            nodes_migrated = sum(
                1 for n in sorted_nodes
                if old_assignments[n] != taxi_dag_assignments[taxi_id][n]
            )
            total_migrations += nodes_migrated
    
    gat_network.train()
    actor.train()
    
    return {
        'total_migrations': total_migrations,
        'total_violations': total_violations,
        'total_reward': total_reward_sum,
    }
