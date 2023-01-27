import torch
import torch.nn as nn

import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, hidden_dims=768, ratio=4) -> None:
        super(MLP, self).__init__()
        self.hidden_dims = hidden_dims
        self.ratio = ratio
        self.act = nn.ReLU()
        self.drop = nn.Dropout()
        self.linear1 = nn.Linear(self.hidden_dims, self.ratio * self.hidden_dims, bias=True)
        self.linear2 = nn.Linear(ratio * self.hidden_dims, self.hidden_dims, bias=True)
        self.process = nn.Sequential(self.linear1, self.act, self.drop, self.linear2, self.act)
    
    def forward(self, x):
        x = self.process(x)
        return x 
    
class MultiHeadAttention(nn.Module):
    def __init__(self, input_dims=None, num_heads=12, hidden_dims=768, scale=None) -> None:
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.hidden_dims = hidden_dims
        self.input_dims = input_dims or hidden_dims
        self.head_dims = hidden_dims // self.num_heads
        self.scale = scale or self.head_dims ** -0.5
        
        # 这个3是为了后面分成q,k,v三份
        self.qkv = nn.Linear(self.input_dims, self.hidden_dims * 3, bias=True)
        
        # self.proj是多头注意力拼接后跟的那个线性层
        self.proj = nn.Linear(self.hidden_dims, self.hidden_dims, bias=True)
        
    def forward(self, x):
        B, N, D = x.shape
        # shape of qkv : [3, B, 6, N, head_dims]
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dims).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        attn = F.softmax(k @ q.transpose(-1, -2), dim=-1) * self.scale
        out = attn @ v
        
        # 这里的reshape操作就已经完成了concate的操作，在reshape之前要把相邻的维度通过
        # tranpose放在一起
        out = out.transpose(1, 2).reshape(B, N, self.hidden_dims)
        out = self.proj(out)
        
        return out

class TransformerEncoderLayer(nn.Module):
    def __init__(self, hidden_dims=768, img_shape=224, patch_size=16) -> None:
        super(TransformerEncoderLayer, self).__init__()
        self.hidden_dims = hidden_dims
        self.num_patch = (img_shape // patch_size) ** 2
        self.mlp = MLP()
        self.msa = MultiHeadAttention()
        self.act = nn.LayerNorm((1+self.num_patch, self.hidden_dims))
        
    def forward(self, x):
        x_norm = self.act(x)
        out_msa = self.msa(x_norm)
        
        out_msa = out_msa + x
        out_msa_norm = self.act(out_msa)
        
        out_mlp = self.mlp(out_msa_norm)
        out = out_mlp + out_msa
        
        return out

class VisionTransformer(nn.Module):
    def __init__(self, hidden_dims=768, num_layers=12, img_shape=224, patch_size=16, num_class=1000) -> None:
        super(VisionTransformer, self).__init__()
        self.img_shape = img_shape
        self.hidden_dims = hidden_dims
        self.num_layers = num_layers
        self.ps = patch_size
        self.num_class = num_class
        self.num_patch = (img_shape // patch_size) ** 2
        self.pe = nn.Linear(self.ps ** 2 * 3, self.hidden_dims)
        self.encoder_layer = TransformerEncoderLayer()
        self.fc = nn.Linear(self.hidden_dims, self.num_class)
        
        # 位置编码
        self.postion_embeding = nn.Parameter(torch.rand(1+self.num_patch, self.hidden_dims))
        
        # 类别token
        self.cls = nn.Parameter(torch.rand(1, self.hidden_dims))
        
        self.vit = nn.Sequential()
        for _ in range(self.num_layers):
            self.vit.append(self.encoder_layer)
        
    def forward(self, x):
        B, C, H, W = x.shape
        x = x.reshape(B, self.num_patch, -1)

        x = self.pe(x)
        
        cls = nn.Parameter(self.cls.repeat(B, 1, 1))
        pos = nn.Parameter(self.postion_embeding.repeat(B, 1, 1))
        
        x = torch.concat((x, cls), dim=1)
        x = x + pos
        
        x = self.vit(x)
        
        x = self.fc(x[:, 0])
        
        return x
    