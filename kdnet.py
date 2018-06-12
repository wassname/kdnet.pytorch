import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class ConvBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, dropout=0):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm1d(out_channels)
        self.act = nn.LeakyReLU()
#         self.drp = nn.Dropout(dropout)
        
        # init
        gain = nn.init.calculate_gain('leaky_relu')
        nn.init.xavier_uniform_(self.conv.weight, gain=gain)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
#         x = self.drp(x)
        return x

class KDNet_Batch(nn.Module):
    def __init__(self, depth=11, k = 16, input_levels=11, channels2=4, features=8, max_pool=False):
        """
        Uses a for loop, for simpler and slower logic.
        
        depth: Desired models depth, should be <=input_levesl
        input_levels: levels in the input kdtree
        k: output dimensions
        channels:  input channels
        features: base number of features for first convnet
        """
        super().__init__()
        self.channels2 = channels2
        self.depth = depth
        self.input_levels = input_levels
        self.max_pool = max_pool
        self.initial_leaf_dim = 2**(input_levels - depth)
        
        current_channels = self.channels2//2*self.initial_leaf_dim
        self.mult = 1 if self.max_pool else 2
        
        channels = (2**(np.arange(1,input_levels+1)//2)*features).tolist()
        print(channels)
        
        self.convs = torch.nn.ModuleList()
        for i in range(depth):
            out_channels = channels[i]
            self.convs.append(ConvBlock(current_channels * self.mult, out_channels * self.channels2,1,1))
            current_channels = out_channels

        hidden_dims = current_channels * self.mult
        print(hidden_dims)
        self.fc = nn.Linear(hidden_dims, k)
        
    def forward(self, x, cutdims):
        def kdconv(x, cutdim, block):
            # This version is just does each sample seperate, then joins the batch
            batchsize = x.size(0)
            channels = self.channels2
            batch, feat_dims, old_leaves = x.size()
            old_leaf_dims = feat_dims // channels
            
            # featdim = channels * points_in_leaf (since they are all unorder we group them as the non spatial dim)
            x = x.view(batchsize, channels * old_leaf_dims, old_leaves)
            x = block(x)
            leaf_dims = x.size(1)//channels
            leaves = x.size(-1)
            # It comes out as (-1, leaf_dims* channels, leaves) we will group the channels with the leaves then select the ones we want
            x = x.view(-1, leaf_dims, channels, leaves)
            x = x.view(batchsize, leaf_dims, channels * leaves)

            # Do each batch separately for now to avoid errors
            xs = []
            for i in range(batchsize):
                sel = Variable(cutdim[i] + (torch.arange(0, leaves) * channels).long())
                if x.is_cuda:
                    sel = sel.cuda()
                xi = torch.index_select(x[i], dim=-1, index=sel)
                
                # Reshape back to real dimensions
                xi = xi.view(leaf_dims, leaves)

                # Reduce amount of leaves for next level
                if self.max_pool:
                    xi = xi.view(leaf_dims, leaves // 2, 2)
                    xi = torch.squeeze(torch.max(xi, dim=-1, keepdim=True)[0], 3)
                else:
                    xi = xi.view(leaf_dims*2, leaves // 2)
                xs.append(xi)
            x = torch.stack(xs, 0)
            return x
        
        if len(x.shape)==4:
            # From (batch, channels, leaf_dim, leaves) to  (batch, channels * leaf_dim, leaves) 
            # We treat the channels and leaf_dim as one since they are both non spatial/non-ordered dimensions
            x = x.view(batch_size, -1, x.size(-1))
        
        
        # input shape should be (batch, channels, leaf_points, leaves)
        for i in range(self.depth):
            dim = 2**(self.depth-i)
            x = kdconv(x, cutdims[self.depth-i-1],  self.convs[i])
        
        x = x.view(-1, self.fc.in_features)
        out = self.fc(x)
        return out

    
class KDNet_Batch2(nn.Module):
    def __init__(self, depth=11, k = 16, input_levels=11, channels=4, features=None):
        """
        A slightly more complex version.
        
        depth: Desired models depth, should be <=input_levesl
        input_levels: levels in the input kdtree
        k: output dimensions
        channels:  input channels
        features: base number of features for first convnet
        """
        super().__init__()
        self.channels = channels
        self.depth = depth
        self.input_levels = input_levels
        if features is None:
            features = 2**(input_levels-depth)
        
        current_channels = self.channels2//2
        self.convs = torch.nn.ModuleList()
        channels = (2**(np.arange(1,input_levels+1)//2)*features).tolist()
        print(channels)
        for i in range(depth):
            out_channels = channels[i]
            self.convs.append(ConvBlock(current_channels*2, out_channels * self.channels,1,1))
            current_channels = out_channels

        self.fc = nn.Linear(current_channels*2**(self.input_levels-self.depth+1), k)
        nn.init.constant_(self.fc.weight, 0.001)
        nn.init.constant_(self.fc.bias, 0)
        
    def forward(self, x, cutdims):
        def kdconv(x, dim, featdim, cutdim, block):
            x = block(x)
            batchsize = x.size(0)  
            
            # Reshape to (featuredim, -1) so we can select
            x = x.view(batchsize, featdim, self.channels2 * dim)
            x = x.transpose(1,0).contiguous()
            x = x.view(featdim, self.channels2 * dim * batchsize)

            # We want to select the cut dimension, but index_select can only take in a 1d array
            # so we have some reshaping to do.
            
            # Offset cutdim so we can use it to select on a flattened array
            cutdim_offset = (torch.arange(0, dim) * self.channels2).repeat(batchsize, 1).long()
            sel = Variable(cutdim + cutdim_offset) 
            sel = sel.view(-1, 1)
            # Work out offsets for cutdims
            offset = Variable((torch.arange(0, batchsize) * dim * self.channels2))
            offset = offset.repeat(dim, 1).long().transpose(1, 0).contiguous().view(-1,1)
            
            sel2 = sel+offset
            sel2 = sel2.squeeze()
            if x.is_cuda:
                sel2 = sel2.cuda()     

            x = torch.index_select(x, dim = 1, index = sel2)
            # Reshape back
            x = x.view(featdim, batchsize, dim)
            x = x.transpose(1,0)    # (batchsize, featdim, dim)
            x = x.transpose(2,1).contiguous()    # (batchsize, dim, featdim)
            
            # move some half of the dimensions to the features
            x = x.view(-1, dim//2, featdim * 2)  # (-1, dim//2, featdim*2)   
            x = x.transpose(2,1).contiguous()    # (-1, featdim*2, dim//2)   
            return x
        
        for i in range(self.depth):
            outdims = self.convs[i].conv.out_channels//self.channels2
            dim = 2**(self.input_levels-i)
            x = kdconv(x, dim, outdims, cutdims[-i-1],  self.convs[i])
        
        x = x.view(-1, outdims*2**(self.input_levels-self.depth+1))
        out = self.fc(x)
        return out
