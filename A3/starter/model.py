import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential as Seq, Linear as Lin, LeakyReLU, GroupNorm

# the "MLP" block that you will use the in the PointNet and CorrNet modules you will implement
# This block is made of a linear transformation (FC layer), 
# followed by a Leaky RelU, a Group Normalization (optional, depending on enable_group_norm)
# the Group Normalization (see Wu and He, "Group Normalization", ECCV 2018) creates groups of 32 channels
def MLP(channels, enable_group_norm=True):
    if enable_group_norm:
        num_groups = [0]
        for i in range(1, len(channels)):
            if channels[i] >= 32:
                num_groups.append(channels[i]//32)
            else:
                num_groups.append(1)    
        return Seq(*[Seq(Lin(channels[i - 1], channels[i]), LeakyReLU(negative_slope=0.2), GroupNorm(num_groups[i], channels[i]))
                     for i in range(1, len(channels))])
    else:
        return Seq(*[Seq(Lin(channels[i - 1], channels[i]), LeakyReLU(negative_slope=0.2))
                     for i in range(1, len(channels))])


# PointNet module for extracting point descriptors
# num_input_features: number of input raw per-point or per-vertex features 
# 		 			  (should be 3, since we have 3D point positions in this assignment)
# num_output_features: number of output per-point descriptors (should be 32 for this assignment)
# this module should include
# - a MLP that processes each point i into a 128-dimensional vector f_i
# - another MLP that further processes these 128-dimensional vectors into h_i (same number of dimensions)
# - a max-pooling layer that collapses all point features h_i into a global shape representaton g
# - a concat operation that concatenates (f_i, g) to create a new per-point descriptor that stores local+global information
# - a MLP followed by a linear transformation layer that transform this concatenated descriptor into the output 32-dimensional descriptor x_i
# **** YOU SHOULD CHANGE THIS MODULE, CURRENTLY IT IS INCORRECT ****
class PointNet(torch.nn.Module):
    '''def __init__(self, num_input_features, num_output_features):
        super(PointNet, self).__init__()
        self.mlp = MLP([num_input_features, num_output_features])

    def forward(self, x):
        x = self.mlp(x)
        return x'''
    def __init__(self, num_input_features=3, num_output_features=32):
        super(PointNet, self).__init__()
        self.mlp_fi = MLP([num_input_features, 32, 64, 128])
        self.mlp_hi = MLP([128, 128])
        self.mlp_yi = MLP([256, 128, 64])
        self.mlp_y = Lin(64, num_output_features)

    def forward(self, x):
        f = self.mlp_fi(x)
        h = self.mlp_hi(f)
        g = torch.max(h, dim=0)[0]
        g_exp = g.expand(len(f), -1)
        f_concat_g = torch.cat((f, g_exp), dim=1)
        yi = self.mlp_yi(f_concat_g)
        y = self.mlp_y(yi)
        
        return y

# CorrNet module that serves 2 purposes:  
# (a) uses the PointNet module to extract the per-point descriptors of the point cloud (out_pts)
#     and the same PointNet module to extract the per-vertex descriptors of the mesh (out_vtx)
# (b) if self.train_corrmask=1, it outputs a correspondence mask
# The CorrNet module should
# - include a (shared) PointNet to extract the per-point and per-vertex descriptors 
# - normalize these descriptors to have length one
# - when train_corrmask=1, it should include a MLP that outputs a confidence 
#   that represents whether the mesh vertex i has a correspondence or not
#   Specifically, you should use the cosine similarity to compute a similarity matrix NxM where
#   N is the number of mesh vertices, M is the number of points in the point cloud
#   Each entry encodes the similarity of vertex i with point j
#   Use the similarity matrix to find for each mesh vertex i, its most similar point n[i] in the point cloud 
#   Form a descriptor matrix X = NxF whose each row stores the point descriptor of n[i] (from the point cloud descriptors)
#   Form a vector S = Nx1 whose each entry stores the similarity of the pair (i, n[i])
#   From the PointNet, you also have the descriptor matrix Y = NxF storing the per-vertex descriptors
#   Concatenate [X Y S] into a N x (2F + 1) matrix
#   Transform this matrix into the correspondence mask Nx1 through a MLP followed by a linear transformation
# **** YOU SHOULD CHANGE THIS MODULE, CURRENTLY IT IS INCORRECT ****
class CorrNet(torch.nn.Module):
    def __init__(self, num_output_features, train_corrmask):        
        super(CorrNet, self).__init__()
        self.train_corrmask = train_corrmask
        self.pointnet_share = PointNet(3, num_output_features)

        if self.train_corrmask:
            self.mlp_fin = nn.Sequential(nn.Linear(num_output_features * 2 + 1, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, vtx, pts):
        out_vtx = self.pointnet_share(vtx)
        out_pts = self.pointnet_share(pts)

        vtx_feature = F.normalize(out_vtx, p=2, dim=1)
        pts_feature = F.normalize(out_pts, p=2, dim=1)
        
        if self.train_corrmask:     
            x_matrix = vtx_feature
            similarity_matrix = torch.matmul(vtx_feature, pts_feature.T)
            n = similarity_matrix.argmax(dim=1)
            y_matrix = pts_feature[n, :]
            sim_vector = torch.max(similarity_matrix, dim=1)[0].unsqueeze(1)
            concat_matrix = torch.cat((x_matrix, y_matrix, sim_vector), dim=1)
            out_corrmask = self.mlp_fin(concat_matrix)
            
        else:
            out_corrmask=None
        
        return vtx_feature, pts_feature, out_corrmask
        

        
