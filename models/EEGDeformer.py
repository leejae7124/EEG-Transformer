# This is the script of EEG-Deformer
# This is the network script
import torch
from torch import nn

from einops import rearrange
from einops.layers.torch import Rearrange


def pair(t):
    return t if isinstance(t, tuple) else (t, t)


class FeedForward(nn.Module): #Transformer의 MLP 블록 역할.
    # nn.LayerNorm -> Linear -> GELU -> Dropout -> Linear -> Dropout
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module): #Multi-head self-attention을 구현한 모듈.
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class Transformer(nn.Module): #Deformer의 핵심 구조
    # CNN과 Transformer를 결합한 Hybrid Layer를 여러 층 쌓은 구조.
    # nn.Sequential(): Pytorch에서 여러 레이어를 순차적으로 묶어주는 컨테이너.
    def cnn_block(self, in_chan, kernel_size, dp): # fine-grained를 위해 병렬적으로 구성된 cnn 블록
        return nn.Sequential(
            nn.Dropout(p=dp),
            nn.Conv1d(in_channels=in_chan, out_channels=in_chan,
                      kernel_size=kernel_size, padding=self.get_padding_1D(kernel=kernel_size)),
            nn.BatchNorm1d(in_chan),
            nn.ELU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )

    def __init__(self, dim, depth, heads, dim_head, mlp_dim, in_chan, fine_grained_kernel=11, dropout=0.): #Atteitnion + CNN 통합 구조를 "정의!!"
        #파라미터 설명
        # dim: 입력 feature의 차원 (시간 feature 길이 등), depth: Transformer 블록 몇 층 쌓을지, heads: Multi=head attention의 head 수
        # dim_head: 각 head당 차원, mlp_dim: FeedForward 레이어의 은닉 크기, in_chan: CNN의 입력 채널 수(eeg 채널 수?)
        # fine_grained_kernel: CNN에 사용할 커널 크기, dropout: 드롭아웃 비율
        super().__init__()
        self.layers = nn.ModuleList([])
        for i in range(depth): #depth는 HCT 블록 갯수랑 똑같음!! 정의함.
            dim = int(dim * 0.5) #층이 깊어질수록 차원을 절반씩 줄인다. Transformer가 점점 압축된 feature를 학습
            self.layers.append(nn.ModuleList([
                Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout),
                FeedForward(dim, mlp_dim, dropout=dropout),
                self.cnn_block(in_chan=in_chan, kernel_size=fine_grained_kernel, dp=dropout)
            ])) #하나의 HCT 블록 같음. 포함된 모듈: Attention, FeedForward, cnn_block
            #다음과 같은 구조가 된다.
            #             [
            # [attn1, ff1, cnn1],
            # [attn2, ff2, cnn2],
            # ...
            # ]

        self.pool = nn.MaxPool1d(kernel_size=2, stride=2) #입력 feature를 다운샘플링. coarse feature 추출에 사용?

    def forward(self, x): #x의 shape는 (batch_size, in_chan, time_steps) 형태의 시계열 입력 -> EEG 데이터의 CNN 전처리 이후의 feature
        #forward는 데이터를 실행함. 앞에서 정의한 블록을 "하나씩 꺼내서 입력 데이터(x)"에 적용하는 부분. 즉, 실제 forward pass(순전파) 때 실행되는 "데이터 흐름 처리 로직"
        dense_feature = []
        for attn, ff, cnn in self.layers: #정의한 블록수만큼? 반복
            x_cg = self.pool(x)
            x_cg = attn(x_cg) + x_cg
            x_fg = cnn(x)
            x_info = self.get_info(x_fg)  # (b, in_chan)
            dense_feature.append(x_info)
            x = ff(x_cg) + x_fg
        x_dense = torch.cat(dense_feature, dim=-1)  # b, in_chan*depth
        x = x.view(x.size(0), -1)   # b, in_chan*d_hidden_last_layer
        emd = torch.cat((x, x_dense), dim=-1)  # b, in_chan*(depth + d_hidden_last_layer)
        return emd

    def get_info(self, x):
        # x: b, k, l
        x = torch.log(torch.mean(x.pow(2), dim=-1))
        return x

    def get_padding_1D(self, kernel):
        return int(0.5 * (kernel - 1))


class Conv2dWithConstraint(nn.Conv2d):
    def __init__(self, *args, doWeightNorm=True, max_norm=1, **kwargs):
        self.max_norm = max_norm
        self.doWeightNorm = doWeightNorm
        super(Conv2dWithConstraint, self).__init__(*args, **kwargs)

    def forward(self, x):
        if self.doWeightNorm:
            self.weight.data = torch.renorm(
                self.weight.data, p=2, dim=0, maxnorm=self.max_norm
            )
        return super(Conv2dWithConstraint, self).forward(x)


class Deformer(nn.Module):
    def cnn_block(self, out_chan, kernel_size, num_chan):
        return nn.Sequential(
            Conv2dWithConstraint(1, out_chan, kernel_size, padding=self.get_padding(kernel_size[-1]), max_norm=2),
            Conv2dWithConstraint(out_chan, out_chan, (num_chan, 1), padding=0, max_norm=2),
            nn.BatchNorm2d(out_chan),
            nn.ELU(),
            nn.MaxPool2d((1, 2), stride=(1, 2))
        )

    def __init__(self, *, num_chan, num_time, temporal_kernel, num_kernel=64,
                 num_classes, depth=4, heads=16,
                 mlp_dim=16, dim_head=16, dropout=0.):
        super().__init__()

        self.cnn_encoder = self.cnn_block(out_chan=num_kernel, kernel_size=(1, temporal_kernel), num_chan=num_chan)

        dim = int(0.5*num_time)  # embedding size after the first cnn encoder

        self.to_patch_embedding = Rearrange('b k c f -> b k (c f)')

        self.pos_embedding = nn.Parameter(torch.randn(1, num_kernel, dim))

        self.transformer = Transformer(
            dim=dim, depth=depth, heads=heads, dim_head=dim_head,
            mlp_dim=mlp_dim, dropout=dropout,
            in_chan=num_kernel, fine_grained_kernel=temporal_kernel,
        )

        L = self.get_hidden_size(input_size=dim, num_layer=depth)

        out_size = int(num_kernel * L[-1]) + int(num_kernel * depth)

        self.mlp_head = nn.Sequential(
            nn.Linear(out_size, num_classes)
        )

    def forward(self, eeg):
        # eeg: (b, chan, time)
        eeg = torch.unsqueeze(eeg, dim=1)  # (b, 1, chan, time)
        x = self.cnn_encoder(eeg)  # (b, num_kernel, 1, 0.5*num_time)

        x = self.to_patch_embedding(x)

        b, n, _ = x.shape
        x += self.pos_embedding
        x = self.transformer(x)
        return self.mlp_head(x)

    def get_padding(self, kernel):
        return (0, int(0.5 * (kernel - 1)))

    def get_hidden_size(self, input_size, num_layer):
        return [int(input_size * (0.5 ** i)) for i in range(num_layer + 1)]


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    data = torch.ones((16, 32, 1000))
    emt = Deformer(num_chan=32, num_time=1000, temporal_kernel=11, num_kernel=64,
                 num_classes=2, depth=4, heads=16,
                 mlp_dim=16, dim_head=16, dropout=0.5)
    print(emt)
    print(count_parameters(emt))

    out = emt(data)
