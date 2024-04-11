# import torch._dynamo
# torch._dynamo.config.suppress_errors = True

from typing import *
import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import Parameter, init
from einops import rearrange
from einops.layers.torch import Rearrange
import math 
import difflib
from torch.nn import *
import math
import numpy as np
import torch.nn.functional as F




def get_layer(l_name, library=torch.nn):
    """Return layer object handler from library e.g. from torch.nn

    E.g. if l_name=="elu", returns torch.nn.ELU.

    Args:
        l_name (string): Case insensitive name for layer in library (e.g. .'elu').
        library (module): Name of library/module where to search for object handler
        with l_name e.g. "torch.nn".

    Returns:
        layer_handler (object): handler for the requested layer e.g. (torch.nn.ELU)

    """

    all_torch_layers = [x for x in dir(torch.nn)]
    match = [x for x in all_torch_layers if l_name.lower() == x.lower()]
    if len(match) == 0:
        close_matches = difflib.get_close_matches(
            l_name, [x.lower() for x in all_torch_layers]
        )
        raise NotImplementedError(
            "Layer with name {} not found in {}.\n Closest matches: {}".format(
                l_name, str(library), close_matches
            )
        )
    elif len(match) > 1:
        close_matches = difflib.get_close_matches(
            l_name, [x.lower() for x in all_torch_layers]
        )
        raise NotImplementedError(
            "Multiple matchs for layer with name {} not found in {}.\n "
            "All matches: {}".format(l_name, str(library), close_matches)
        )
    else:
        # valid
        layer_handler = getattr(library, match[0])
        return layer_handler

class LayerNormalization4D(nn.Module):
    def __init__(self, input_dimension, eps=1e-5):
        super().__init__()
        param_size = [1, input_dimension, 1, 1]
        self.gamma = Parameter(torch.Tensor(*param_size).to(torch.float32))
        self.beta = Parameter(torch.Tensor(*param_size).to(torch.float32))
        init.ones_(self.gamma)
        init.zeros_(self.beta)
        self.eps = eps

    def forward(self, x):
        with torch.autocast(device_type = "cuda", enabled = False):
            if x.ndim == 4:
                _, C, _, _ = x.shape
                stat_dim = (1,)
            else:
                raise ValueError("Expect x to have 4 dimensions, but got {}".format(x.ndim))
            mu_ = x.mean(dim=stat_dim, keepdim=True)  # [B,1,T,F]
            std_ = torch.sqrt(torch.clamp(x.var(dim=stat_dim, unbiased=False, keepdim=True), self.eps))  # [B,1,T,F]
            x_hat = (x - mu_) / (std_ )
                
            x_hat = x_hat * self.gamma + self.beta

            return x_hat


class IntraFrameFulBandModule(nn.Module):
    def __init__(self,
                 emb_dim,
                 emb_ks,
                 emb_hs,
                 hidden_channels,
                 eps=1e-5,
                 **kwargs
                 ) -> None:

        super().__init__()
        
        self.emb_dim = emb_dim
        self.emb_ks = emb_ks
        self.emb_hs = emb_hs
        
        in_channels = emb_dim * emb_ks
        self.intra_norm = LayerNormalization4D(emb_dim, eps=eps)
        self.intra_rnn = nn.LSTM(
            in_channels, hidden_channels, 1, batch_first=True, bidirectional=True
        )
        self.intra_linear = nn.ConvTranspose1d(
            hidden_channels * 2, emb_dim, emb_ks, stride=emb_hs
        )

    def forward(self, Rb):
        B, C, old_T, old_Q = Rb.shape
        T = math.ceil((old_T - self.emb_ks) / self.emb_hs) * \
            self.emb_hs + self.emb_ks
        Q = math.ceil((old_Q - self.emb_ks) / self.emb_hs) * \
            self.emb_hs + self.emb_ks
        Rb = F.pad(Rb, (0, Q - old_Q, 0, T - old_T))
        
        self.intra_rnn.flatten_parameters()  # Vahid

        # intra RNN
        input_ = Rb
        B, C, old_T, old_Q = Rb.shape
        T = math.ceil((old_T - self.emb_ks) / self.emb_hs) * \
            self.emb_hs + self.emb_ks
        Q = math.ceil((old_Q - self.emb_ks) / self.emb_hs) * \
            self.emb_hs + self.emb_ks
        x = F.pad(Rb, (0, Q - old_Q, 0, T - old_T))

        intra_rnn = self.intra_norm(input_)  # [B, C, T, Q]
        intra_rnn = (
            intra_rnn.transpose(1, 2).contiguous().view(B * T, C, Q)
        )  # [BT, C, Q]
        intra_rnn = F.unfold(
            intra_rnn[..., None], (self.emb_ks, 1), stride=(self.emb_hs, 1)
        )  # [BT, C*emb_ks, -1]
        intra_rnn = intra_rnn.transpose(1, 2)  # [BT, -1, C*emb_ks]
        intra_rnn, _ = self.intra_rnn(intra_rnn)  # [BT, -1, H]
        intra_rnn = intra_rnn.transpose(1, 2)  # [BT, H, -1]
        intra_rnn = self.intra_linear(intra_rnn)  # [BT, C, Q]
        intra_rnn = intra_rnn.view([B, T, C, Q])
        intra_rnn = intra_rnn.transpose(1, 2).contiguous()  # [B, C, T, Q]
        intra_rnn = intra_rnn + input_  # [B, C, T, Q]

        return [B, C, old_T, T, old_Q, Q ], intra_rnn


class LayerNormalization4DCF(nn.Module):
    def __init__(self, input_dimension, eps=1e-5):
        super().__init__()
        assert len(input_dimension) == 2
        param_size = [1, input_dimension[0], 1, input_dimension[1]]
        self.gamma = Parameter(torch.Tensor(*param_size).to(torch.float32))
        self.beta = Parameter(torch.Tensor(*param_size).to(torch.float32))
        init.ones_(self.gamma)
        init.zeros_(self.beta)
        self.eps = eps

    def forward(self, x):
        with torch.autocast(device_type = "cuda", enabled = False):
            
            if x.ndim == 4:
                stat_dim = (1, 3)
            else:
                raise ValueError("Expect x to have 4 dimensions, but got {}".format(x.ndim))
            mu_ = x.mean(dim=stat_dim, keepdim=True)  # [B,1,T,1]
            std_ = torch.sqrt(torch.clamp(x.var(dim=stat_dim, unbiased=False, keepdim=True), self.eps))  # [B,1,T,F]
            x_hat = (x - mu_) / (std_)
            
            x_hat = x_hat * self.gamma + self.beta
            
            
            return x_hat



class SubBandTemporalModule(nn.Module):
    def __init__(self,
                 emb_dim,
                 emb_ks,
                 emb_hs,
                 hidden_channels,
                 eps=1e-5,
                 **kwargs
                 ) -> None:

        super().__init__()
        
        self.emb_dim = emb_dim
        self.emb_ks = emb_ks
        self.emb_hs = emb_hs
        
        
        in_channels = emb_dim * emb_ks

        self.inter_norm = LayerNormalization4D(emb_dim, eps=eps)
        self.inter_rnn = nn.LSTM(
            in_channels, hidden_channels, 1, batch_first=True, bidirectional=True
        )
        self.inter_linear = nn.ConvTranspose1d(
            hidden_channels * 2, emb_dim, emb_ks, stride=emb_hs
        )

    def forward(self, Ub, dim=None):
        self.inter_rnn.flatten_parameters()  # Vahid
        B, C, old_T, T, old_Q, Q = dim
        input_ = Ub
        inter_rnn = self.inter_norm(input_)  # [B, C, T, F]
        inter_rnn = (
            inter_rnn.permute(0, 3, 1, 2).contiguous().view(B * Q, C, T)
        )  # [BF, C, T]
        inter_rnn = F.unfold(
            inter_rnn[..., None], (self.emb_ks, 1), stride=(self.emb_hs, 1)
        )  # [BF, C*emb_ks, -1]

        inter_rnn = inter_rnn.transpose(1, 2)  # [BF, -1, C*emb_ks]
        inter_rnn, _ = self.inter_rnn(inter_rnn)  # [BF, -1, H]
        inter_rnn = inter_rnn.transpose(1, 2)  # [BF, H, -1]
        inter_rnn = self.inter_linear(inter_rnn)  # [BF, C, T]
        inter_rnn = inter_rnn.view([B, Q, C, T])
        inter_rnn = inter_rnn.permute(0, 2, 3, 1).contiguous()  # [B, C, T, Q]
        inter_rnn = inter_rnn + input_  # [B, C, T, Q]
        inter_rnn = inter_rnn[..., :old_T, :old_Q]
        return inter_rnn



class LSTMPositionalEncdoing(nn.Module):
    def __getitem__(self, key):
        return getattr(self, key)

    def __init__(
        self,
        **kwargs
    ):
        super().__init__()
        
        self.intra_frame_fulband_module_audio = IntraFrameFulBandModule(**kwargs)
        self.subband_temporal_module_audio = SubBandTemporalModule(**kwargs)
         

    def forward(self, x):
        """GridNetBlock Forward.

        Args:
            x: [B, C, T, F]
            out: [B, C, T, F]
        """
        

        dim, U = self.intra_frame_fulband_module_audio(x)
        Z  = self.subband_temporal_module_audio(U, dim= dim)
        
        return Z

class RandomizedPositionalEncoding(nn.Module):
    def __init__(self, F, C, Tmax = 2000):
        """
        :param channels: The last dimension of the tensor you want to apply pos emb to.
        """
        super(RandomizedPositionalEncoding, self).__init__()
        channels = F * C 
        
        self.Tmax = Tmax
        self.org_channels = channels
        self.channels = int(np.ceil(channels / 2) * 2)
    
    def get_emb_from_pos(self, pos_x, dtype, device):
        inv_freq = 1.0 / (10000 ** (torch.arange(0, self.channels, 2, device=device, dtype=dtype) / self.channels))
        sin_inp_x = torch.einsum("bi,j->bij", pos_x, inv_freq)
        emb_x = self.get_emb(sin_inp_x)
        emb = torch.zeros((sin_inp_x.shape[0], pos_x.shape[-1], self.channels), device=device, dtype=dtype)
        emb[..., : self.channels] = emb_x
        emb = emb[..., :self.org_channels].permute(0, 2, 1) # (B, ch, Tmax)
        return emb
        
    
    def get_emb(self, sin_inp):
        emb = torch.stack((sin_inp.sin(), sin_inp.cos()), dim=-1)
        return torch.flatten(emb, -2, -1)

    def forward(self, x):
        B, F, T, H = x.shape
        assert self.Tmax >= T, f"time dimention should be smaller than Tmax = {self.Tmax} currently it is {T}"
        x = rearrange(x, 'b f t h -> b (f h) t')
        
        # if not in training mode, take first T embeddings 
        if not self.training:
            pos_x = torch.arange(T, device=x.device, dtype=x.dtype).reshape(1, T)
            emb = self.get_emb_from_pos(pos_x, device=x.device, dtype=x.dtype).repeat(B, 1, 1)
        # else sample T random embeddings
        else:
            pos_x = torch.stack([torch.sort(torch.randperm( self.Tmax, device=x.device, dtype=x.dtype )[:T]).values for _ in range(B)])
            emb = self.get_emb_from_pos(pos_x, device=x.device, dtype=x.dtype)  

                
        emb = rearrange(emb, 'b (f h) t -> b f t h', h = H, f = F)

        return emb + x


class RandomChunkPositionalEncoding(nn.Module):
    def __init__(self, F, C, Tmax = 5000):
        """
        :param channels: The last dimension of the tensor you want to apply pos emb to.
        """
        super(RandomChunkPositionalEncoding, self).__init__()
        channels = F * C 
        
        self.Tmax = Tmax
        self.org_channels = channels
        self.channels = int(np.ceil(channels / 2) * 2)
    
    def get_emb_from_pos(self, pos_x, dtype, device):
        inv_freq = 1.0 / (10000 ** (torch.arange(0, self.channels, 2, device=device, dtype=dtype) / self.channels))
        sin_inp_x = torch.einsum("bi,j->bij", pos_x, inv_freq)
        emb_x = self.get_emb(sin_inp_x)
        emb = torch.zeros((sin_inp_x.shape[0], pos_x.shape[-1], self.channels), device=device, dtype=dtype)
        emb[..., : self.channels] = emb_x
        emb = emb[..., :self.org_channels].permute(0, 2, 1) # (B, ch, Tmax)
        return emb
        
    
    def get_emb(self, sin_inp):
        emb = torch.stack((sin_inp.sin(), sin_inp.cos()), dim=-1)
        return torch.flatten(emb, -2, -1)

    def forward(self, x):
        B, F, T, H = x.shape
        assert self.Tmax >= T, f"time dimention should be smaller than Tmax = {self.Tmax} currently it is {T}"
        # x = rearrange(x, 'b f t h -> b (f h) t')
        
        # if not in training mode, take first T embeddings 
        if not self.training:
            pos_x = torch.arange(T, device=x.device, dtype=x.dtype).reshape(1, T)
            emb = self.get_emb_from_pos(pos_x, device=x.device, dtype=x.dtype).repeat(B, 1, 1)
        # else sample T random embeddings
        else:
            indicies = []
            for i in range(B):
                start = torch.randint(0, max(self.Tmax - T, 0), (1,)).item()
                end = start + T
                indicies.append( torch.arange(start, end, device=x.device, dtype=x.dtype) )
            
            pos_x = torch.stack(indicies)
            emb = self.get_emb_from_pos(pos_x, device=x.device, dtype=x.dtype)  

                
        emb = rearrange(emb, 'b (f h) t -> b f t h', h = H, f = F)
        return emb + x

class LinearGroup(nn.Module):

    def __init__(self, in_features: int, out_features: int, num_groups: int, bias: bool = True) -> None:
        super(LinearGroup, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_groups = num_groups 
        print(in_features, out_features, num_groups)
        self.weight = Parameter(torch.empty((num_groups, out_features, in_features)))
        if bias:
            self.bias = Parameter(torch.empty(num_groups, out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # same as linear
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)

    def forward(self, x: Tensor) -> Tensor:
        """shape [..., group, feature]"""
        x = torch.einsum("...gh,gkh->...gk", x, self.weight)
        if self.bias is not None:
            x = x + self.bias
        return x

    def extra_repr(self) -> str:
        return f"{self.in_features}, {self.out_features}, num_groups={self.num_groups}, bias={True if self.bias is not None else False}"


class Conv1dGroup(nn.Module):

    def __init__(self, in_features: int, out_features: int, num_groups: int, kernel_size: int, bias: bool = True) -> None:
        super(Conv1dGroup, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_groups = num_groups
        self.kernel_size = kernel_size

        self.weight = Parameter(torch.empty((num_groups, out_features, in_features, kernel_size)))
        if bias:
            self.bias = Parameter(torch.empty(num_groups, out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # same as linear
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)

    def forward(self, x: Tensor) -> Tensor:
        """shape [batch, time, group, feature]"""
        (B, T, G, F), K = x.shape, self.kernel_size
        x = x.permute(0, 2, 3, 1).reshape(B * G * F, 1, 1, T)  # [B*G*F,1,1,T]
        x = torch.nn.functional.unfold(x, kernel_size=(1, K), padding=(0, K // 2))  # [B*G*F,K,T]
        x = x.reshape(B, G, F, K, T)
        x = torch.einsum("bgfkt,gofk->btgo", x, self.weight)
        if self.bias is not None:
            x = x + self.bias
        return x

    def extra_repr(self) -> str:
        return f"{self.in_features}, {self.out_features}, num_groups={self.num_groups}, kernel_size={self.kernel_size}, bias={True if self.bias is not None else False}"


class PReLU(nn.PReLU):

    def __init__(self, num_parameters: int = 1, init: float = 0.25, dim: int = 1, device=None, dtype=None) -> None:
        super().__init__(num_parameters, init, device, dtype)
        self.dim = dim

    def forward(self, input: Tensor) -> Tensor:
        if self.dim == 1:
            # [B, Chn, Feature]
            return super().forward(input)
        else:
            return super().forward(input.transpose(self.dim, 1)).transpose(self.dim, 1)


def new_non_linear(non_linear_type: str, dim_hidden: int, seq_last: bool) -> nn.Module:
    if non_linear_type.lower() == 'prelu':
        return PReLU(num_parameters=dim_hidden, dim=1 if seq_last == True else -1)
    elif non_linear_type.lower() == 'silu':
        return nn.SiLU()
    elif non_linear_type.lower() == 'sigmoid':
        return nn.Sigmoid()
    elif non_linear_type.lower() == 'relu':
        return nn.ReLU()
    else:
        raise Exception(non_linear_type)




class LayerNorm(nn.LayerNorm):

    def __init__(self, seq_last: bool, **kwargs) -> None:
        # """
        # Arg s:
        #     seq_last (bool): whether the sequence dim is the last dim
        # """
        super().__init__(**kwargs)
        self.seq_last = seq_last

    def forward(self, input: Tensor) -> Tensor:
        if self.seq_last:
            input = input.transpose(-1, 1)  # [B, H, Seq] -> [B, Seq, H], or [B,H,w,h] -> [B,h,w,H]
        o = super().forward(input)
        if self.seq_last:
            o = o.transpose(-1, 1)
        return o


class GlobalLayerNorm(nn.Module):

    def __init__(self, dim_hidden: int, seq_last: bool, eps: float = 1e-5) -> None:
        super().__init__()
        self.dim_hidden = dim_hidden
        self.seq_last = seq_last
        self.eps = eps

        if seq_last:
            self.weight = Parameter(torch.empty([dim_hidden, 1]))
            self.bias = Parameter(torch.empty([dim_hidden, 1]))
        else:
            self.weight = Parameter(torch.empty([dim_hidden]))
            self.bias = Parameter(torch.empty([dim_hidden]))
        init.ones_(self.weight)
        init.zeros_(self.bias)

    def forward(self, input: Tensor) -> Tensor:
        # """
        # Args:
        #     input (Tensor): shape [B, Seq, H] or [B, H, Seq]
        # """
        var, mean = torch.var_mean(input, dim=(1, 2), unbiased=False, keepdim=True)

        output = (input - mean) / torch.sqrt(var + self.eps)
        output = output * self.weight + self.bias
        return output

    def extra_repr(self) -> str:
        return '{dim_hidden}, seq_last={seq_last}, eps={eps}'.format(**self.__dict__)


class BatchNorm1d(nn.Module):

    def __init__(self, seq_last: bool, **kwargs) -> None:
        super().__init__()
        self.seq_last = seq_last
        self.bn = nn.BatchNorm1d(**kwargs)

    def forward(self, input: Tensor) -> Tensor:
        if not self.seq_last:
            input = input.transpose(-1, -2)  # [B, Seq, H] -> [B, H, Seq]
        o = self.bn.forward(input)  # accepts [B, H, Seq]
        if not self.seq_last:
            o = o.transpose(-1, -2)
        return o


class GroupNorm(nn.GroupNorm):

    def __init__(self, seq_last: bool, **kwargs) -> None:
        super().__init__(**kwargs)
        self.seq_last = seq_last

    def forward(self, input: Tensor) -> Tensor:
        if self.seq_last == False:
            input = input.transpose(-1, -2)  # [B, Seq, H] -> [B, H, Seq]
        o = super().forward(input)  # accepts [B, H, Seq]
        if self.seq_last == False:
            o = o.transpose(-1, -2)
        return o


class GroupBatchNorm(nn.Module):
    # """Applies Group Batch Normalization over a group of inputs

    # see: `Changsheng Quan, Xiaofei Li. NBC2: Multichannel Speech Separation with Revised Narrow-band Conformer. arXiv:2212.02076.`

    # """

    dim_hidden: int
    group_size: int
    eps: float
    affine: bool
    seq_last: bool
    share_along_sequence_dim: bool

    def __init__(
        self,
        dim_hidden: int,
        group_size: Optional[int],
        share_along_sequence_dim: bool = False,
        seq_last: bool = False,
        affine: bool = True,
        eps: float = 1e-5,
    ) -> None:
        # """
        # Args:
        #     dim_hidden (int): hidden dimension
        #     group_size (int): the size of group, optional
        #     share_along_sequence_dim (bool): share statistics along the sequence dimension. Defaults to False.
        #     seq_last (bool): whether the shape of input is [B, Seq, H] or [B, H, Seq]. Defaults to False, i.e. [B, Seq, H].
        #     affine (bool): affine transformation. Defaults to True.
        #     eps (float): Defaults to 1e-5.
        # """
        super(GroupBatchNorm, self).__init__()

        self.dim_hidden = dim_hidden
        self.group_size = group_size
        self.eps = eps
        self.affine = affine
        self.seq_last = seq_last
        self.share_along_sequence_dim = share_along_sequence_dim
        if self.affine:
            if seq_last:
                self.weight = Parameter(torch.empty([dim_hidden, 1]))
                self.bias = Parameter(torch.empty([dim_hidden, 1]))
            else:
                self.weight = Parameter(torch.empty([dim_hidden]))
                self.bias = Parameter(torch.empty([dim_hidden]))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.affine:
            init.ones_(self.weight)
            init.zeros_(self.bias)

    def forward(self, x: Tensor, group_size: int = None) -> Tensor:
        """
        Args:
            x: shape [B, Seq, H] if seq_last=False, else shape [B, H, Seq] , where B = num of groups * group size.
            group_size: the size of one group. if not given anywhere, the input must be 4-dim tensor with shape [B, group_size, Seq, H] or [B, group_size, H, Seq]
        """
        if self.group_size != None:
            assert group_size == None or group_size == self.group_size, (group_size, self.group_size)
            group_size = self.group_size

        if group_size is not None:
            assert (x.shape[0] // group_size) * group_size, f'batch size {x.shape[0]} is not divisible by group size {group_size}'

        original_shape = x.shape
        if self.seq_last == False:
            if x.ndim == 4:
                assert group_size is None or group_size == x.shape[1], (group_size, x.shape)
                B, group_size, Seq, H = x.shape
            else:
                B, Seq, H = x.shape
                x = x.reshape(B // group_size, group_size, Seq, H)

            if self.share_along_sequence_dim:
                var, mean = torch.var_mean(x, dim=(1, 2, 3), unbiased=False, keepdim=True)
            else:
                var, mean = torch.var_mean(x, dim=(1, 3), unbiased=False, keepdim=True)

            output = (x - mean) / torch.sqrt(var + self.eps)
            if self.affine:
                output = output * self.weight + self.bias

            output = output.reshape(original_shape)
        else:
            if x.ndim == 4:
                assert group_size is None or group_size == x.shape[1], (group_size, x.shape)
                B, group_size, H, Seq = x.shape
            else:
                B, H, Seq = x.shape
                x = x.reshape(B // group_size, group_size, H, Seq)

            if self.share_along_sequence_dim:
                var, mean = torch.var_mean(x, dim=(1, 2, 3), unbiased=False, keepdim=True)
            else:
                var, mean = torch.var_mean(x, dim=(1, 2), unbiased=False, keepdim=True)

            output = (x - mean) / torch.sqrt(var + self.eps)
            if self.affine:
                output = output * self.weight + self.bias

            output = output.reshape(original_shape)

        return output

    def extra_repr(self) -> str:
        return '{dim_hidden}, {group_size}, share_along_sequence_dim={share_along_sequence_dim}, seq_last={seq_last}, eps={eps}, ' \
            'affine={affine}'.format(**self.__dict__)


def new_norm(norm_type: str, dim_hidden: int, seq_last: bool, group_size: int = None, num_groups: int = None) -> nn.Module:
    if norm_type.upper() == 'LN':
        norm = LayerNorm(normalized_shape=dim_hidden, seq_last=seq_last)
    elif norm_type.upper() == 'GBN':
        norm = GroupBatchNorm(dim_hidden=dim_hidden, seq_last=seq_last, group_size=group_size, share_along_sequence_dim=False)
    elif norm_type == 'GBNShare':
        norm = GroupBatchNorm(dim_hidden=dim_hidden, seq_last=seq_last, group_size=group_size, share_along_sequence_dim=True)
    elif norm_type.upper() == 'BN':
        norm = BatchNorm1d(num_features=dim_hidden, seq_last=seq_last)
    elif norm_type.upper() == 'GN':
        norm = GroupNorm(num_groups=num_groups, num_channels=dim_hidden, seq_last=seq_last)
    elif norm == 'gLN':
        norm = GlobalLayerNorm(dim_hidden, seq_last=seq_last)
    else:
        raise Exception(norm_type)
    return norm


def get_layer(l_name, library=torch.nn):
    # """Return layer object handler from library e.g. from torch.nn

    # E.g. if l_name=="elu", returns torch.nn.ELU.

    # Args:
    #     l_name (string): Case insensitive name for layer in library (e.g. .'elu').
    #     library (module): Name of library/module where to search for object handler
    #     with l_name e.g. "torch.nn".

    # Returns:
    #     layer_handler (object): handler for the requested layer e.g. (torch.nn.ELU)

    # """

    all_torch_layers = [x for x in dir(torch.nn)]
    match = [x for x in all_torch_layers if l_name.lower() == x.lower()]
    if len(match) == 0:
        close_matches = difflib.get_close_matches(
            l_name, [x.lower() for x in all_torch_layers]
        )
        raise NotImplementedError(
            "Layer with name {} not found in {}.\n Closest matches: {}".format(
                l_name, str(library), close_matches
            )
        )
    elif len(match) > 1:
        close_matches = difflib.get_close_matches(
            l_name, [x.lower() for x in all_torch_layers]
        )
        raise NotImplementedError(
            "Multiple matchs for layer with name {} not found in {}.\n "
            "All matches: {}".format(l_name, str(library), close_matches)
        )
    else:
        # valid
        layer_handler = getattr(library, match[0])
        return layer_handler



class LayerNormalization4DCF(nn.Module):
    def __init__(self, input_dimension, eps=1e-5):
        super().__init__()
        assert len(input_dimension) == 2
        param_size = [1, input_dimension[0], 1, input_dimension[1]]
        self.gamma = Parameter(torch.Tensor(*param_size).to(torch.float32))
        self.beta = Parameter(torch.Tensor(*param_size).to(torch.float32))
        init.ones_(self.gamma)
        init.zeros_(self.beta)
        self.eps = eps

    def forward(self, x):
        x = x.to(torch.float32)
        
        if x.ndim == 4:
            stat_dim = (1, 3)
        else:
            raise ValueError("Expect x to have 4 dimensions, but got {}".format(x.ndim))
        x_hat = (x - x.mean(dim=stat_dim, keepdim=True)) / (torch.sqrt(x.var(dim=stat_dim, unbiased=False, keepdim=True)) + self.eps)

        x_hat = x_hat * self.gamma + self.beta
        
        
        return x_hat


class CrossFrameSelfAttentionModule(nn.Module):
    def __init__(self,
                 embed_dim: int = 64,
                 n_freqs: int = 129,
                 num_heads: int = 4,
                 approx_qk_dim: int = 512,
                 activation: int = "prelu",
                 eps=1e-5,
                 **kwargs

                 ) -> None:

        super().__init__()
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        
        
        E = math.ceil(
            approx_qk_dim * 1.0 / n_freqs
        )  # approx_qk_dim is only approximate
        assert embed_dim % num_heads == 0
        
                
        for ii in range(num_heads):
            
            self.add_module(
                "attn_conv_Q_%d" % ii,
                nn.Sequential(
                    nn.Conv2d(embed_dim , E, 1),
                    get_layer(activation)(),
                    LayerNormalization4DCF((E, n_freqs), eps=eps),
                ),
            )
            self.add_module(
                "attn_conv_K_%d" % ii,
                nn.Sequential(
                    nn.Conv2d(embed_dim, E, 1),
                    get_layer(activation)(),
                    LayerNormalization4DCF((E, n_freqs), eps=eps),
                ),
            )
            self.add_module(
                "attn_conv_V_%d" % ii,
                nn.Sequential(
                    nn.Conv2d(embed_dim, embed_dim // num_heads, 1),
                    get_layer(activation)(),
                    LayerNormalization4DCF(
                        (embed_dim // num_heads, n_freqs), eps=eps),
                ),
            )
        self.add_module(
            "attn_concat_proj",
            nn.Sequential(
                nn.Conv2d(embed_dim, embed_dim, 1),
                get_layer(activation)(),
                LayerNormalization4DCF((embed_dim, n_freqs), eps=eps),
            ),
        )
    def __getitem__(self, key):
        return getattr(self, key)

    def forward(self, Zb_audio, average_attn_weights=False, attn_mask=None):
        # B, C, old_T, T, old_Q, Q = dim
        B, F, old_T, H = Zb_audio.shape
        # B = Zb_audio.shape[0]

        batch_a = rearrange(Zb_audio, "B F T C -> B C T F")
        
        
        

        all_Q, all_K, all_V = [], [], []
        for ii in range(self.num_heads):
            
            all_Q.append(self["attn_conv_Q_%d" % ii](batch_a))  # [B, C, T, F]
            all_K.append(self["attn_conv_K_%d" % ii](batch_a))  # [B, C, T, F]
            all_V.append(self["attn_conv_V_%d" % ii](batch_a))  # [B, C, T, F]

        Q = torch.cat(all_Q, dim=0)  # [Bxh, C, T, F]
        K = torch.cat(all_K, dim=0)  # [Bxh, C, T, F]
        V = torch.cat(all_V, dim=0)  # [Bxh, C, T, F]
        
        

        Q = Q.transpose(1, 2).flatten(start_dim=2)  # [Bxh, T, C*F]
        K = K.transpose(1, 2).flatten(start_dim=2)  # [Bxh, T, C*F]
        V = V.transpose(1, 2)  # [Bxh, T, C, F]
        old_shape = V.shape
        V = V.flatten(start_dim=2)  # [Bxh, T, C*F]
        embed_dim = Q.shape[-1]
        
        


        # with torch.autocast(device_type=Q.device.type, dtype=torch.float32):
        
        attn_mat = torch.matmul(Q / embed_dim**0.5, K.transpose(1, 2) / embed_dim**0.5) * embed_dim**0.5  # [Bxh, T, T]
        attn_mat_sotftmax = torch.nn.functional.softmax(attn_mat, dim=2)  # [Bxh, T, T]
        V = torch.matmul(attn_mat_sotftmax, V)  # [Bxh, T, C*F]

        V = V.reshape(old_shape)  # [Bxh, T, C, F]
        V = V.transpose(1, 2)  # [Bxh, C, T, F]
        embed_dim = V.shape[1]
        

        # [num_heads, B, C, T, F])
        batch = V.view([self.num_heads, B, embed_dim, old_T, -1])
        batch = batch.transpose(0, 1)  # [B, num_heads, C, T, F])
        batch = batch.contiguous().view(
            [B, self.num_heads * embed_dim, old_T, -1]
        )  # [B, C, T, F])
        batch = self["attn_concat_proj"](batch)  # [B, C, T, F])
        
        
        batch = rearrange(batch, "B C T F -> B F T C")

        out = batch + Zb_audio

        return out, attn_mat_sotftmax




class GlobalMultiheadSlefAttentionModule(nn.Module):
    def __init__(self,
                 embed_dim: int = 64,
                 n_freqs: int = 129,
                 num_heads: int = 4,
                 approx_qk_dim: int = 512,
                 activation: int = "prelu",
                 eps=1e-5,
                 **kwargs

                 ) -> None:

        super().__init__()
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        
        self.in_reshape = Rearrange("B F T C -> B C T F")
        
        
        E = math.ceil(
            approx_qk_dim * 1.0 / n_freqs
        )  # approx_qk_dim is only approximate
        assert embed_dim % num_heads == 0
        
        self.attn_conv_Q = nn.Sequential(
                    nn.Conv2d(embed_dim , num_heads*E, 1),
                    get_layer(activation)(),
                    Rearrange("B (num_heads E) T F -> (num_heads B) E T F", num_heads = num_heads),
                    LayerNormalization4DCF((E, n_freqs), eps=eps),
                    Rearrange("Bh C T F -> Bh T (C F)"),
                )
        
        self.attn_conv_K = nn.Sequential(
                    nn.Conv2d(embed_dim , num_heads*E, 1),
                    get_layer(activation)(),
                    Rearrange("B (num_heads E) T F -> (num_heads B) E T F", num_heads = num_heads),
                    LayerNormalization4DCF((E, n_freqs), eps=eps),
                    Rearrange("Bh C T F -> Bh T (C F)"),
                )
        
        
        self.attn_conv_V = nn.Sequential(
                    nn.Conv2d(embed_dim , num_heads*(embed_dim // num_heads), 1),
                    get_layer(activation)(),
                    Rearrange("B (num_heads E) T F -> (num_heads B) E T F", num_heads = num_heads),
                    LayerNormalization4DCF((embed_dim // num_heads, n_freqs), eps=eps),
                    Rearrange("hB C T F -> hB T (C F)"),
                )
        
        self.v_reshape = Rearrange("(h B) T (e F) -> B (h e) T F", F = n_freqs, h = self.num_heads)
        
        self.attn_concat_proj = nn.Sequential(
                nn.Conv2d(embed_dim, embed_dim, 1),
                get_layer(activation)(),
                LayerNormalization4DCF((embed_dim, n_freqs), eps=eps),
                Rearrange("B C T F -> B F T C")
            )
        
        
        
    def __getitem__(self, key):
        return getattr(self, key)

    def forward(self, x, average_attn_weights=False, attn_mask=None):

        batch_a = self.in_reshape(x)
       
        Q = self.attn_conv_Q(batch_a)  # [Bxh, C, T, F]
        K = self.attn_conv_K(batch_a)  # [Bxh, C, T, F]
        V = self.attn_conv_V(batch_a)  # [Bh, T, CxF]
        
        V = torch.nn.functional.scaled_dot_product_attention(Q,K,V) # [Bxh, T, C*F]
        
        V = self.v_reshape(V)

        batch = self.attn_concat_proj(V)  # [B, C, T, F])

        out = batch + x

        return out, None




class CrossNetLayer(nn.Module):

    def __init__(
            self,
            dim_hidden: int,
            dim_ffn: int,
            dim_squeeze: int,
            num_freqs: int,
            num_heads: int,
            dropout: Tuple[float, float, float] = (0, 0, 0),
            kernel_size: Tuple[int, int] = (5, 3),
            conv_groups: Tuple[int, int] = (8, 8),
            norms: List[str] = ("LN", "LN", "GN", "LN", "LN", "LN"),
            padding: str = 'zeros',
            full: nn.Module = None,
            gmhsa: bool = True,
    ) -> None:
        super().__init__()
        f_conv_groups = conv_groups[0]
        t_conv_groups = conv_groups[1]
        f_kernel_size = kernel_size[0]
        t_kernel_size = kernel_size[1]

        # cross-band block
        # frequency-convolutional module
        self.fconv1 = nn.ModuleList([
            new_norm(norms[3], dim_hidden, seq_last=True, group_size=None, num_groups=f_conv_groups),
            nn.Conv1d(in_channels=dim_hidden, out_channels=dim_hidden, kernel_size=f_kernel_size, groups=f_conv_groups, padding='same', padding_mode=padding),
            nn.PReLU(dim_hidden),
        ])
        # full-band linear module
        self.norm_full = new_norm(norms[5], dim_hidden, seq_last=False, group_size=None, num_groups=f_conv_groups)
        self.full_share = False if full == None else True
        self.squeeze = nn.Sequential(nn.Conv1d(in_channels=dim_hidden, out_channels=dim_squeeze, kernel_size=1), nn.SiLU())
        self.dropout_full = nn.Dropout2d(dropout[2]) if dropout[2] > 0 else None
        self.full = LinearGroup(num_freqs, num_freqs, num_groups=dim_squeeze) if full == None else full
        self.unsqueeze = nn.Sequential(nn.Conv1d(in_channels=dim_squeeze, out_channels=dim_hidden, kernel_size=1), nn.SiLU())
        
        # frequency-convolutional module
        self.fconv2 = nn.ModuleList([
            new_norm(norms[4], dim_hidden, seq_last=True, group_size=None, num_groups=f_conv_groups),
            nn.Conv1d(in_channels=dim_hidden, out_channels=dim_hidden, kernel_size=f_kernel_size, groups=f_conv_groups, padding='same', padding_mode=padding),
            nn.PReLU(dim_hidden),
        ])

        # # narrow-band block
        # # MHSA module
        # self.norm_mhsa = new_norm(norms[0], dim_hidden, seq_last=False, group_size=None, num_groups=t_conv_groups)
        # self.mhsa = MultiheadAttention(embed_dim=dim_hidden, num_heads=num_heads, batch_first=True)
        # self.dropout_mhsa = nn.Dropout(dropout[0])
        
        # cross multihead self attention
        self.gmhsa = gmhsa 
        if self.gmhsa:
            self.norm_crossmhsa = new_norm(norms[0], dim_hidden, seq_last=False, group_size=None, num_groups=t_conv_groups)
            self.global_mhsa = GlobalMultiheadSlefAttentionModule(embed_dim=dim_hidden, num_heads=num_heads, n_freqs= num_freqs, batch_first=True)
            self.dropout_crossmhsa = nn.Dropout(dropout[0])
        
        
        # T-ConvFFN module
        self.tconvffn = nn.ModuleList([
            new_norm(norms[1], dim_hidden, seq_last=True, group_size=None, num_groups=t_conv_groups),
            nn.Conv1d(in_channels=dim_hidden, out_channels=dim_ffn, kernel_size=1),
            nn.SiLU(),
            nn.Conv1d(in_channels=dim_ffn, out_channels=dim_ffn, kernel_size=t_kernel_size, padding='same', groups=t_conv_groups),
            nn.SiLU(),
            nn.Conv1d(in_channels=dim_ffn, out_channels=dim_ffn, kernel_size=t_kernel_size, padding='same', groups=t_conv_groups),
            new_norm(norms[2], dim_ffn, seq_last=True, group_size=None, num_groups=t_conv_groups),
            nn.SiLU(),
            nn.Conv1d(in_channels=dim_ffn, out_channels=dim_ffn, kernel_size=t_kernel_size, padding='same', groups=t_conv_groups),
            nn.SiLU(),
            nn.Conv1d(in_channels=dim_ffn, out_channels=dim_hidden, kernel_size=1),
        ])
        self.dropout_tconvffn = nn.Dropout(dropout[1])

    def forward(self, x: Tensor, att_mask: Optional[Tensor] = None) -> Tensor:
        r"""
        Args:
            x: shape [B, F, T, H]
            att_mask: the mask for attention along T. shape [B, T, T]

        Shape:
            out: shape [B, F, T, H]
        """
        x = x + self._fconv(self.fconv1, x)
        
        x = x + self._full(x)
        
        x = x + self._fconv(self.fconv2, x)
        
        if self.gmhsa:
            x, attn = self._csa(x, att_mask)
        else:
            attn = None

        x = x + self._tconvffn(x)
        
        return x, attn

    def _tsa(self, x: Tensor, attn_mask: Optional[Tensor]) -> Tuple[Tensor, Tensor]:
        B, F, T, H = x.shape
        x = self.norm_mhsa(x)
        x = x.reshape(B * F, T, H)
        need_weights = False
        x, attn = self.mhsa(x, x, x, average_attn_weights=False, attn_mask=attn_mask, need_weights = need_weights)
        x = x.reshape(B, F, T, H).contiguous()
        return self.dropout_mhsa(x)
    
    def _csa(self, x: Tensor, attn_mask: Optional[Tensor]) -> Tuple[Tensor, Tensor]:
        
        B, F, T, H = x.shape
        x = self.norm_crossmhsa(x)
        
        
        # x = rearrange(x, "B F T H -> (B H) T F")
        x, attn = self.global_mhsa(x, average_attn_weights=False, attn_mask=attn_mask)
        # x = rearrange(x, "(B H) T F -> B F T H", F = F)
        return self.dropout_crossmhsa(x), attn

    def _tconvffn(self, x: Tensor) -> Tensor:
        B, F, T, H0 = x.shape
        # T-Conv
        x = x.transpose(-1, -2).contiguous()  # [B,F,H,T]
        x = x.reshape(B * F, H0, T).contiguous()
        for m in self.tconvffn:
            if type(m) == GroupBatchNorm:
                x = m(x, group_size=F)
            else:
                x = m(x)
        x = x.reshape(B, F, H0, T).contiguous()
        x = x.transpose(-1, -2).contiguous()  # [B,F,T,H]
        return self.dropout_tconvffn(x)

    def _fconv(self, ml: nn.ModuleList, x: Tensor) -> Tensor:
        B, F, T, H = x.shape
        x = x.permute(0, 2, 3, 1).contiguous()  # [B,T,H,F]
        x = x.reshape(B * T, H, F).contiguous()
        for m in ml:
            if type(m) == GroupBatchNorm:
                x = m(x, group_size=T)
            else:
                x = m(x)
        x = x.reshape(B, T, H, F).contiguous()
        x = x.permute(0, 3, 1, 2).contiguous()  # [B,F,T,H]
        return x

    def _full(self, x: Tensor) -> Tensor:
        B, F, T, H = x.shape
        x = self.norm_full(x)
        x = x.permute(0, 2, 3, 1)  # [B,T,H,F]
        x = x.reshape(B * T, H, F)
        x = self.squeeze(x)  # [B*T,H',F]
        if self.dropout_full:
            x = x.reshape(B, T, -1, F).contiguous()
            x = x.transpose(1, 3).contiguous()  # [B,F,H',T]
            x = self.dropout_full(x)  # dropout some frequencies in one utterance
            x = x.transpose(1, 3).contiguous()  # [B,T,H',F]
            x = x.reshape(B * T, -1, F).contiguous()

        x = self.full(x)  # [B*T,H',F]
        x = self.unsqueeze(x)  # [B*T,H,F]
        x = x.reshape(B, T, H, F).contiguous()
        x = x.permute(0, 3, 1, 2).contiguous()  # [B,F,T,H]
        return x

    def extra_repr(self) -> str:
        return f"full_share={self.full_share}"


class CrossNet(nn.Module):

    def __init__(
            self,
            dim_input: int,  # the input dim for each time-frequency point
            dim_output: int,  # the output dim for each time-frequency point
            dim_squeeze: int,
            num_layers: int,
            num_freqs: int,
            encoder_kernel_size: int = 5,
            dim_hidden: int = 192,
            dim_ffn: int = 384,
            num_heads: int = 2,
            dropout: Tuple[float, float, float] = (0, 0, 0),
            kernel_size: Tuple[int, int] = (5, 3),
            conv_groups: Tuple[int, int] = (8, 8),
            norms: List[str] = ("LN", "LN", "GN", "LN", "LN", "LN"),
            padding: str = 'zeros',
            full_share: int = 0,  # share from layer 0,
            positional_encoding: bool = True,
            positional_encoding_type: str = "random_chunk", # random or lstm or random_chunk
            positional_encoding_hidden_channels: int = 64,
            gmhsa: bool = True,
    ):
        super().__init__()
        
        self.name = "CrossNet"
        
        if positional_encoding:
            assert positional_encoding_type in ["random", "lstm", "random_chunk"]

        # encoder
        self.encoder = nn.Conv1d(in_channels=dim_input, out_channels=dim_hidden, kernel_size=encoder_kernel_size, stride=1, padding="same")
        
        self.pe = None 
        self.pe_type = positional_encoding_type
        if positional_encoding:
            print(f"Using Positional Encoding {positional_encoding_type}")
            if positional_encoding_type == "random":
                self.pe = RandomizedPositionalEncoding(C = dim_hidden, F = num_freqs)
            elif positional_encoding_type == "random_chunk":
                self.pe = RandomChunkPositionalEncoding(C = dim_hidden, F = num_freqs)
            elif positional_encoding_type == "lstm":
                self.pe = LSTMPositionalEncdoing(
                        emb_dim = dim_hidden,
                        emb_ks = 4,
                        emb_hs = 1,
                        n_freqs = num_freqs, 
                        hidden_channels = positional_encoding_hidden_channels, # 192
                        approx_qk_dim=512,
                        activation="prelu",
                        eps=1e-5,
                )
            else: 
                raise NotImplementedError("Positional Encoding Type {} not implemented".format(positional_encoding_type))

        # spatialnet layers
        full = None
        layers = []
        for l in range(num_layers):
            layer = CrossNetLayer(
                dim_hidden=dim_hidden,
                dim_ffn=dim_ffn,
                dim_squeeze=dim_squeeze,
                num_freqs=num_freqs,
                num_heads=num_heads,
                dropout=dropout,
                kernel_size=kernel_size,
                conv_groups=conv_groups,
                norms=norms,
                padding=padding,
                full=full if l > full_share else None,
                gmhsa=gmhsa,
            )
            if hasattr(layer, 'full'):
                full = layer.full
            layers.append(layer)
        self.layers = nn.ModuleList(layers)

        # decoder
        self.decoder = nn.Linear(in_features=dim_hidden, out_features=dim_output)

    def forward(self, x: Tensor, return_attn_score: bool = False) -> Tensor:
        # x: [Batch, Freq, Time, Feature]
        B, F, T, H0 = x.shape
        x = self.encoder(x.reshape(B * F, T, H0).permute(0, 2, 1).contiguous()).permute(0, 2, 1).contiguous()
        H = x.shape[2]
        attns = [] if return_attn_score else None
        x = x.reshape(B, F, T, H).contiguous()
        
        pe = 0.0
        if self.pe is not None:
            if self.pe_type == "lstm":
                x = rearrange(x, "B F T C -> B C T F")
                x = self.pe(x) 
                x = rearrange(x, "B C T F -> B F T C")
            else: 
                x = self.pe(x)
                
        for m in self.layers:
            # x = x + pe
            x, attn = m(x)
            if return_attn_score:
                attns.append(attn)
        
        # x = rearrange(x, "B F T C -> B C T F")
        # x = self.out_lstm(x)
        # x = rearrange(x, "B C T F -> B F T C")
        

        y = self.decoder(x)
        if return_attn_score:
            return y.contiguous(), attns
        else:
            return y.contiguous()


if __name__ == '__main__':
    x = torch.randn((1, 129, 250, 2)) # .cuda()
    crossnet_small = CrossNet(
        dim_input=2,
        dim_output=4,
        num_layers=8,
        dim_hidden=96,
        dim_ffn=192,
        kernel_size=(5, 3),
        conv_groups=(8, 8),
        norms=("LN", "LN", "GN", "LN", "LN", "LN"),
        dim_squeeze=4,
        num_freqs=129,
        full_share=0,
        positional_encoding=True,
        positional_encoding_type="random_chunk",
    ) # .cuda()
    import time
    ts = time.time()
    y = crossnet_small(x)
    # torch.cuda.synchronize(7)
    te = time.time()
    print(crossnet_small)
    print(y.shape)
    print(te - ts)

    crossnet_small = crossnet_small.to('meta')
    x = x.to('meta')
    from torch.utils.flop_counter import FlopCounterMode # requires torch>=2.1.0
    with FlopCounterMode(crossnet_small, display=False) as fcm:
        y = crossnet_small(x)
        flops_forward_eval = fcm.get_total_flops()
        res = y.sum()
        res.backward()
        flops_backward_eval = fcm.get_total_flops() - flops_forward_eval

    params_eval = sum(param.numel() for param in crossnet_small.parameters())
    print(f"flops_forward={flops_forward_eval/1e9:.2f}G, flops_back={flops_backward_eval/1e9:.2f}G, params={params_eval/1e6:.2f} M")