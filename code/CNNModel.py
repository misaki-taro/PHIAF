import torch

class CNNBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, pool_size):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, out_channels, kernel_size,padding='same'),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.MaxPool2d(pool_size),
            torch.nn.Dropout(0.5)
        )

    # X: [bs, c, h, w]        
    def forward(self, X):
        out = self.net(X)
        return out

class AttentionBlock(torch.nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels
        self.att = torch.nn.Linear(in_channels, in_channels)
        self.softmax = torch.nn.Softmax(dim=-1)
        
    # X: [bs, c, h, w]
    def forward(self, X):
        bs = X.shape[0]
        out = X.permute(0, 2, 3, 1) # out: [bs, h, w, c]
        out = out.reshape(bs, -1, self.in_channels) # out: [bs, h*w, c]
        
        # Attention
        alpha = self.att(out) # out: [bs, h*w, c]
        out = alpha * out
        
        return out.reshape(bs, -1)
    
class PHIAFModel(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dna_pool_size, pro_pool_size):
        super().__init__()
        lin_shape = 36 * out_channels
        
        self.dna_net = torch.nn.Sequential(
            CNNBlock(in_channels, out_channels, kernel_size, dna_pool_size),
            AttentionBlock(out_channels)
        )
        
        self.pro_net = torch.nn.Sequential(
            CNNBlock(in_channels, out_channels, kernel_size, pro_pool_size),
            AttentionBlock(out_channels)
        )
        
        self.out_layer = torch.nn.Sequential(
            torch.nn.Linear(lin_shape, 64),
            torch.nn.BatchNorm1d(64),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(64, 1),
            torch.nn.Sigmoid()
        )
    
    def forward(self, X_dna, X_pro):
        # CNN + Attention
        X_dna1 = self.dna_net(X_dna)
        X_pro1 = self.pro_net(X_pro)
        
        # Merge
        X_out = X_dna1 + X_pro1
        
        # Out
        out = self.out_layer(X_out)
        
        return out.reshape(-1)
        