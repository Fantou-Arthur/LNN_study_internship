import torch
import torch.nn as nn
from dgl.nn import GATConv

class GATRegressionModel(nn.Module):
    """
    GAT Regression Model matching the architecture used in the research notebook.
    Uses multi-head Graph Attention Networks (GAT) to learn node-level embeddings
    and perform regression.
    """
    def __init__(self, in_feats, hidden_feats, num_heads):
        super(GATRegressionModel, self).__init__()
        # First GAT layer with multi-head attention and ReLU activation
        self.conv1 = GATConv(in_feats, hidden_feats, num_heads)
        # Second GAT layer predicting a single feature per head
        self.conv2 = GATConv(hidden_feats * num_heads, 1, num_heads)
    
    def forward(self, g, features):
        """
        Forward pass.
        
        Args:
            g (dgl.DGLGraph): The graph structure.
            features (torch.Tensor): Node feature tensors.
            
        Returns:
            torch.Tensor: Node regression predictions.
        """
        # Multi-head output is flattened for conv2
        x = torch.relu(self.conv1(g, features).flatten(1))
        # Final output averages multi-head predictions
        x = self.conv2(g, x).mean(1)
        return x
