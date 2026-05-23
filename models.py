import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as torch_init
import numpy as np
from torch.nn.modules.module import Module
#self attention
class SelfAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., attn_head_dim=None):
        super().__init__()
        self.num_heads = num_heads
        head_dim = attn_head_dim if attn_head_dim is not None else dim // num_heads
        all_head_dim = head_dim * self.num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, all_head_dim * 3, bias=False)
        self.q_bias = nn.Parameter(torch.zeros(all_head_dim)) if qkv_bias else None
        self.v_bias = nn.Parameter(torch.zeros(all_head_dim)) if qkv_bias else None
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(all_head_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, T, N, C = x.shape
        x = x.view(-1, 1, N, C)
        B_new, T_new, N_new, C_new = x.shape
        qkv_bias = torch.cat((self.q_bias, torch.zeros_like(self.v_bias, requires_grad=False),
                              self.v_bias)) if self.q_bias is not None else None
        qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
        qkv = qkv.reshape(B_new, T_new, N_new, 3, self.num_heads, -1).permute(3, 0, 1, 4, 2, 5)#3*B*T*head*N*dim
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1)).softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(2, 3).reshape(B_new, T_new, N_new, -1)
        x = self.proj_drop(self.proj(x))
        x = x.view(B, T, N, C)
        return x

#cross attention
class CrossAttention2(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., attn_head_dim=None):
        super().__init__()
        self.num_heads = num_heads
        head_dim = attn_head_dim if attn_head_dim is not None else dim // num_heads
        all_head_dim = head_dim * self.num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, all_head_dim * 3, bias=False)
        self.q_bias = nn.Parameter(torch.zeros(all_head_dim)) if qkv_bias else None
        self.v_bias = nn.Parameter(torch.zeros(all_head_dim)) if qkv_bias else None
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(all_head_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, T, N, C = x.shape
        x = x.view(-1, 2, N, C)
        B_new, T_new, N_new, C_new = x.shape
        qkv_bias = torch.cat((self.q_bias, torch.zeros_like(self.v_bias, requires_grad=False),
                              self.v_bias)) if self.q_bias is not None else None
        qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
        qkv = qkv.reshape(B_new, T_new, N_new, 3, self.num_heads, -1).permute(3, 0, 1, 4, 2, 5)  # 3*B*T*head*N*dim
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)
        q = q * self.scale
        q = torch.split(q, 1, dim=1)
        k = torch.split(k, 1, dim=1)
        v = torch.split(v, 1, dim=1)
        #q0 k0 v0 && q0 k1 v1
        attn_map1 = ((q[0]) @ k[0].transpose(-2, -1)).softmax(dim=-1)  # attention map between query1 and key2
        attn_map2 = ((q[0]) @ k[1].transpose(-2, -1)).softmax(dim=-1)  # attention map between query2 and key1
        v1_cross = attn_map1 @ v[0]  # cross attention between value2 and attention map1
        v2_cross = attn_map2 @ v[1]  # cross attention between value1 and attention map2
        v_cross = torch.cat([v1_cross, v2_cross], dim=1)  # concatenate the two cross attentions
        x = v_cross.transpose(2, 3).reshape(B_new, T_new, N_new, -1)
        x = self.proj_drop(self.proj(x))
        x = x.view(B,T,N,C)
        return x

#cross attention
class CrossAttention4(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., attn_head_dim=None):
        super().__init__()
        self.num_heads = num_heads
        head_dim = attn_head_dim if attn_head_dim is not None else dim // num_heads
        all_head_dim = head_dim * self.num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, all_head_dim * 3, bias=False)
        self.q_bias = nn.Parameter(torch.zeros(all_head_dim)) if qkv_bias else None
        self.v_bias = nn.Parameter(torch.zeros(all_head_dim)) if qkv_bias else None
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(all_head_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, T, N, C = x.shape
        qkv_bias = torch.cat((self.q_bias, torch.zeros_like(self.v_bias, requires_grad=False),
                              self.v_bias)) if self.q_bias is not None else None
        qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
        qkv = qkv.reshape(B, T, N, 3, self.num_heads, -1).permute(3, 0, 1, 4, 2, 5)  # 3*B*T*head*N*dim
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)
        q = q * self.scale
        q = torch.split(q, 1, dim=1)
        k = torch.split(k, 1, dim=1)
        v = torch.split(v, 1, dim=1)
        # q0 k0 v0 && q0 k1 v1 && q1 k2 v2 && q2 k3 v3
        # attn_map0 = (q[0] @ k[0].transpose(-2, -1)).softmax(dim=-1)  # attention map between query1 and key2
        # attn_map1 = (q[0] @ k[0].transpose(-2, -1)).softmax(dim=-1)
        # attn_map2 = ((q[0]+q[1]) @ (k[0]+k[1]).transpose(-2, -1)).softmax(dim=-1)
        # attn_map3 = ((q[0]+q[1]+q[2]) @ (k[0]+k[1]+k[2]).transpose(-2, -1)).softmax(dim=-1)# attention map between query2 and key1
        attn_map0 = (q[0] @ k[0].transpose(-2, -1)).softmax(dim=-1)
        attn_map1 = (q[0] @ k[1].transpose(-2, -1)).softmax(dim=-1)
        attn_map2 = (q[1] @ k[2].transpose(-2, -1)).softmax(dim=-1)
        attn_map3 = (q[2] @ k[3].transpose(-2, -1)).softmax(dim=-1)
        array = torch.cat([attn_map0,attn_map1,attn_map2,attn_map3],dim=0).cpu().detach().numpy()
        v0_cross = attn_map0 @ v[0]
        v1_cross = attn_map1 @ v[1]
        v2_cross = attn_map2 @ v[2]  # cross attention between value2 and attention map1
        v3_cross = attn_map3 @ v[3]  # cross attention between value1 and attention map2
        v_cross = torch.cat([v0_cross, v1_cross, v2_cross, v3_cross], dim=1)  # concatenate the two cross attentions
        x = v_cross.transpose(2, 3).reshape(B, T, N, -1)
        x = self.proj_drop(self.proj(x))
        return x

class Block(nn.Module):
    def __init__(self, dim=2048, cross_clip=1, num_heads=16, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, attn_head_dim=None):
        super().__init__()
        self.norm1 = norm_layer(dim)
        if cross_clip == 1 or cross_clip == 4:
            self.attn = SelfAttention(
                dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                attn_drop=attn_drop, proj_drop=drop, attn_head_dim=attn_head_dim)
        elif cross_clip == 2:
            self.attn = CrossAttention2(
                dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                attn_drop=attn_drop, proj_drop=drop, attn_head_dim=attn_head_dim)
        else:
            self.attn = CrossAttention4(
                dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                attn_drop=attn_drop, proj_drop=drop, attn_head_dim=attn_head_dim)
        self.norm2 = norm_layer(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)), act_layer(), nn.Linear(int(dim * mlp_ratio), dim),
            nn.Dropout(attn_drop)
        )

    def forward(self, x):
        _, B, _, _ = x.shape
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class Block1(nn.Module):
    def __init__(self, dim=2048, cross_clip=1, num_heads=16, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, attn_head_dim=None):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = SelfAttention(
                dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                attn_drop=attn_drop, proj_drop=drop, attn_head_dim=attn_head_dim)
        self.norm2 = norm_layer(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)), act_layer(), nn.Linear(int(dim * mlp_ratio), dim),
            nn.Dropout(attn_drop)
        )

    def forward(self, x):
        _, B, _, _ = x.shape
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class Block2(nn.Module):
    def __init__(self, dim=2048, cross_clip=2, num_heads=16, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, attn_head_dim=None):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = CrossAttention2(
                dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                attn_drop=attn_drop, proj_drop=drop, attn_head_dim=attn_head_dim)
        self.norm2 = norm_layer(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)), act_layer(), nn.Linear(int(dim * mlp_ratio), dim),
            nn.Dropout(attn_drop)
        )

    def forward(self, x):
        _, B, _, _ = x.shape
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class Block4(nn.Module):
    def __init__(self, dim=2048, cross_clip=4, num_heads=16, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, attn_head_dim=None):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = CrossAttention4(
                dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                attn_drop=attn_drop, proj_drop=drop, attn_head_dim=attn_head_dim)
        self.norm2 = norm_layer(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)), act_layer(), nn.Linear(int(dim * mlp_ratio), dim),
            nn.Dropout(attn_drop)
        )

    def forward(self, x):
        _, B, _, _ = x.shape
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

def get_sinusoid_encoding_table(n_position, d_hid):
    ''' Sinusoid position encoding table '''
    # TODO: make it with torch instead of numpy
    def get_position_angle_vec(position):
        return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2]) # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2]) # dim 2i+1

    return torch.FloatTensor(sinusoid_table).unsqueeze(0)

class Temporal(Module):
    def __init__(self, input_size, out_size):
        super(Temporal, self).__init__()
        self.conv_1 = nn.Sequential(
            nn.Conv1d(in_channels=input_size, out_channels=out_size, kernel_size=3,
                    stride=1, padding=1),
            #nn.ReLU(),
        )
    def forward(self, x):
        B,C,H,W = x.size()
        x = x.view(B*C, H,W)
        #x = x.permute(0,2,1)
        x = self.conv_1(x)
        #x = x.permute(0,2,1)
        x = x.view(B, C, W)
        return x

class Decoder(nn.Module):
    def __init__(self, batchsize, embed_dim, Vitblock_num, cross_clip, split, norm_layer=nn.LayerNorm, norm_pix_loss=False):
        super().__init__()
        if split == 0:
            embed_dim = embed_dim // 16

        self.decoder_embed = nn.Linear(embed_dim, embed_dim, bias=True)
        self.pos_embed = get_sinusoid_encoding_table(16, embed_dim)
        self.dropout = nn.Dropout(p=0.5)
        if cross_clip == 1:
            self.decoder_blocks = nn.ModuleList([
                Block1(dim=embed_dim, cross_clip=cross_clip)
                for i in range(Vitblock_num)])
        elif cross_clip == 2:
            if Vitblock_num == 8:
                self.decoder_blocks = nn.ModuleList([
                    Block1(dim=embed_dim, cross_clip=cross_clip),Block1(dim=embed_dim, cross_clip=cross_clip),
                    Block1(dim=embed_dim, cross_clip=cross_clip), Block1(dim=embed_dim, cross_clip=cross_clip),
                    Block2(dim=embed_dim, cross_clip=cross_clip), Block2(dim=embed_dim, cross_clip=cross_clip),
                    Block2(dim=embed_dim, cross_clip=cross_clip), Block2(dim=embed_dim, cross_clip=cross_clip)
                    ])
            elif Vitblock_num == 6:
                self.decoder_blocks = nn.ModuleList([
                    Block1(dim=embed_dim, cross_clip=cross_clip),Block1(dim=embed_dim, cross_clip=cross_clip),
                    Block1(dim=embed_dim, cross_clip=cross_clip), Block2(dim=embed_dim, cross_clip=cross_clip),
                    Block2(dim=embed_dim, cross_clip=cross_clip), Block2(dim=embed_dim, cross_clip=cross_clip)
                    ])
            elif Vitblock_num == 2:
                self.decoder_blocks = nn.ModuleList([
                    Block1(dim=embed_dim, cross_clip=cross_clip),
                    Block2(dim=embed_dim, cross_clip=cross_clip)])
            elif Vitblock_num == 4:
                self.decoder_blocks = nn.ModuleList([
                    Block1(dim=embed_dim, cross_clip=cross_clip),Block1(dim=embed_dim, cross_clip=cross_clip),
                    Block2(dim=embed_dim, cross_clip=cross_clip),Block2(dim=embed_dim, cross_clip=cross_clip)])
        elif cross_clip == 4:
            if Vitblock_num == 8:
                self.decoder_blocks = nn.ModuleList([
                    Block1(dim=embed_dim, cross_clip=cross_clip),Block1(dim=embed_dim, cross_clip=cross_clip),
                    Block1(dim=embed_dim, cross_clip=cross_clip), Block1(dim=embed_dim, cross_clip=cross_clip),
                    Block4(dim=embed_dim, cross_clip=cross_clip), Block4(dim=embed_dim, cross_clip=cross_clip),
                    Block4(dim=embed_dim, cross_clip=cross_clip), Block4(dim=embed_dim, cross_clip=cross_clip)
                    ])
            elif Vitblock_num == 6:
                self.decoder_blocks = nn.ModuleList([
                    Block1(dim=embed_dim, cross_clip=cross_clip),Block1(dim=embed_dim, cross_clip=cross_clip),
                    Block1(dim=embed_dim, cross_clip=cross_clip), Block4(dim=embed_dim, cross_clip=cross_clip),
                    Block4(dim=embed_dim, cross_clip=cross_clip), Block4(dim=embed_dim, cross_clip=cross_clip)
                    ])
            elif Vitblock_num == 2:
                self.decoder_blocks = nn.ModuleList([
                    Block1(dim=embed_dim, cross_clip=cross_clip),
                    Block4(dim=embed_dim, cross_clip=cross_clip)])
            elif Vitblock_num == 4:
                self.decoder_blocks = nn.ModuleList([
                    Block1(dim=embed_dim, cross_clip=cross_clip),Block1(dim=embed_dim, cross_clip=cross_clip),
                    Block4(dim=embed_dim, cross_clip=cross_clip),Block4(dim=embed_dim, cross_clip=cross_clip)])
            elif Vitblock_num == 1:
                self.decoder_blocks = nn.ModuleList([
                Block1(dim=embed_dim, cross_clip=cross_clip)
                for i in range(Vitblock_num)])
        if not hasattr(self, 'decoder_blocks'):
            # Keep unusual ablations from crashing; use intra-clip blocks when no
            # explicit cross-clip block layout is defined for this setting.
            self.decoder_blocks = nn.ModuleList([
                Block1(dim=embed_dim, cross_clip=cross_clip)
                for i in range(max(1, Vitblock_num))
            ])
        self.norm = norm_layer(embed_dim)
        self.norm_pix_loss = norm_pix_loss
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)

            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    # feature:完整未被掩码的提的特征，target:被掩码掉待恢复的部分，feature_input:被掩码了75%的FV，mask_fei:掩码图中的灰块
    def forward(self, feature):
        if len(feature.shape) == 3:
            B, T, C = feature.size()
            feature = self.dropout(feature.view(B, T, 16, C//16))
        else:
            feature = self.dropout(feature)
        B, T, N, C = feature.size()
        feature_input = self.decoder_embed(feature)
        feature_input = feature_input + self.pos_embed.expand(B, T, -1, -1).type_as(feature_input).to(feature_input.device).clone().detach()
        for blk in self.decoder_blocks:
            feature_input = blk(feature_input)
        feature_output = self.norm(feature_input)
        feature_output = feature_output.view(B, T, -1)
        return feature_output


class Autoencoder_L(nn.Module):
    def __init__(self, input_dim):
        super(Autoencoder_L, self).__init__()
        self.fc1 = nn.Linear(input_dim, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.encoder = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 512)
        self.fc4 = nn.Linear(512, 1024)
        self.decoder = nn.Linear(1024, input_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.encoder(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.sigmoid(self.decoder(x))
        return x

class Autoencoder_B(nn.Module):
    def __init__(self, input_dim):
        super(Autoencoder_B, self).__init__()
        self.encoder = nn.Linear(input_dim, 512)
        self.decoder = nn.Linear(512, input_dim)
        self.dropout = nn.Dropout(0.5)
    def forward(self, x):
        x = F.relu(self.encoder(x))
        x = self.decoder(x)
        return x

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 or classname.find('Linear') != -1:
        torch_init.xavier_uniform_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0)

class ClipRouter(nn.Module):
    def __init__(self, n_feature):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_feature, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 2),
        )
        self.apply(weights_init)

    def forward(self, pooled_feature):
        return torch.softmax(self.net(pooled_feature), dim=-1)


class DynamicRetrievalMLP(nn.Module):
    def __init__(self, max_topk):
        super().__init__()
        self.max_topk = max_topk
        self.fc1 = nn.Linear(max_topk, max_topk)
        self.fc2 = nn.Linear(max_topk, max_topk)
        self.relu = nn.ReLU()
        self.apply(weights_init)

    def forward(self, topk_sim):
        # topk_sim: [B, K], K can be <= max_topk
        k = topk_sim.shape[1]
        if k < self.max_topk:
            pad = torch.zeros(topk_sim.shape[0], self.max_topk - k, device=topk_sim.device, dtype=topk_sim.dtype)
            topk_sim = torch.cat([topk_sim, pad], dim=1)
        elif k > self.max_topk:
            topk_sim = topk_sim[:, :self.max_topk]
            k = self.max_topk
        weights = self.fc2(self.relu(self.fc1(topk_sim)))
        weights = torch.softmax(weights, dim=-1)
        return weights[:, :k]


class model(torch.nn.Module):
    def __init__(
        self,
        max_seqlen,
        feature_size,
        Vitblock_num,
        cross_clip,
        split,
        beta,
        delta,
        score_rec_consistency_weight=0.0,
        input_feature_size=None,
        use_qavclab=True,
        short_clip=2,
        long_clip=4,
        retrieval_topk=16,
        retrieval_samples=32,
        bank_decay=0.9,
        bank_age_weight=0.05,
        difficulty_weight=0.6,
        uncertainty_weight=0.2,
        consistency_weight=0.2,
        teacher_momentum=0.995,
        teacher_threshold=0.5,
        min_pseudo_pos=0.03,
        max_pseudo_pos=1.0,
        max_pseudo_pos_start=-1.0,
        max_pseudo_pos_end=-1.0,
        max_pseudo_pos_anneal_iters=0,
        pseudo_loss_mode='bce',
        pseudo_pos_weight_max=6.0,
        pseudo_rank_weight=0.0,
        pseudo_rank_margin=0.10,
        pseudo_rank_max_samples=128,
        pseudo_rank_neg_ratio=1.0,
        rec_rank_weight=0.0,
        rec_rank_margin=0.08,
        rec_rank_top_ratio=0.10,
        rec_rank_bottom_ratio=0.25,
        rec_rank_max_samples=128,
        temporal_rank_weight=0.0,
        temporal_rank_margin=0.06,
        temporal_rank_top_ratio=0.12,
        temporal_rank_bottom_ratio=0.35,
        temporal_rank_max_samples=128,
        temporal_rank_context_mix=0.55,
        easy_neg_weight=0.0,
        easy_neg_ratio=0.35,
        easy_neg_margin=0.18,
        easy_neg_max_samples=192,
        easy_neg_focus_ratio=1.0,
        score_sparse_weight=0.0,
        score_sparse_target=0.22,
        score_sparse_flat_weight=0.0,
        score_sparse_flat_margin=0.18,
        score_sparse_top_ratio=0.25,
        score_sparse_rec_aware=0.0,
        score_sparse_min_weight=0.25,
        score_baseline_weight=0.0,
        score_baseline_margin=0.42,
        score_baseline_quantile=0.50,
        score_baseline_rec_ratio=0.35,
        score_baseline_top_ratio=0.50,
        score_baseline_tail_margin=0.58,
        enforce_min_pseudo_pos=False,
        hard_route_infer=False,
        infer_with_teacher=False,
        infer_teacher_alpha=0.7,
        infer_with_anchor=False,
        infer_anchor_alpha=0.8,
        infer_with_recon=False,
        infer_recon_alpha=0.8,
        infer_with_memory_novelty=False,
        infer_memory_alpha=0.8,
        infer_memory_frozen_only=False,
        full_evidence_enable=False,
        full_evidence_loss_weight=0.0,
        full_evidence_infer=False,
        full_evidence_alpha=0.65,
        full_evidence_rec_weight=1.0,
        full_evidence_temp_weight=0.35,
        full_evidence_center_weight=0.25,
        full_evidence_power=1.0,
        lab_start_iter=100,
        route_start_iter=100,
        enable_teacher_gate=False,
        teacher_gate_start_iter=200,
        route_batch_cap=256,
        freeze_c2fpl=False,
        scorer_dropout=0.7,
        input_norm='none',
        scorer_hidden1=0,
        scorer_hidden2=128,
        inc_enable=False,
        inc_scene_id=0,
        inc_memory_max=256,
        inc_memory_init=16,
        inc_match_temp=0.07,
        inc_match_threshold=0.55,
        inc_match_sim_threshold=0.55,
        inc_expand_warmup_iters=50,
        inc_proto_momentum=0.95,
        inc_mem_easy_ratio=1.0,
        inc_mem_score_quantile=0.70,
        inc_mem_score_weight=1.2,
        inc_replay_ratio=0.25,
        inc_replay_max_samples=128,
        inc_replay_scene_quota=12,
        inc_replay_topk=4,
        inc_replay_alpha=0.7,
        inc_replay_noise=0.02,
        inc_replay_min_sim=-1.0,
        inc_replay_hard_ratio=0.5,
        inc_replay_weight=0.3,
        inc_replay_warmup_iters=80,
        inc_replay_score_weight=1.0,
        inc_replay_loss_mode='full',
        inc_replay_putback_memory=True,
        inc_expected_scenes=13,
        inc_scene_proto_cap=0,
        inc_scene_min_slots=0,
        inc_overflow_policy='far',
        inc_overflow_expand=False,
        inc_anchor_replay_weight=0.0,
    ):
        super(model, self).__init__()
        self.batchsize = max_seqlen
        self.Vitblock_num = Vitblock_num
        self.cross_clip = cross_clip
        self.short_clip = max(1, min(short_clip, cross_clip))
        self.long_clip = max(self.short_clip, min(long_clip, cross_clip))
        self.feature_size = feature_size
        self.input_feature_size = int(input_feature_size) if input_feature_size is not None else int(feature_size)
        self.use_input_proj = int(self.input_feature_size) != int(self.feature_size)
        self.input_proj = nn.Linear(self.input_feature_size, self.feature_size) if self.use_input_proj else nn.Identity()
        self.input_norm_mode = str(input_norm).lower().strip()
        self.input_ln = nn.LayerNorm(self.feature_size) if self.input_norm_mode == 'layernorm' else None
        if scorer_hidden1 is None or int(scorer_hidden1) <= 0:
            scorer_hidden1 = 512 if int(feature_size) <= 768 else 1024
        scorer_hidden1 = int(max(64, scorer_hidden1))
        scorer_hidden2 = int(max(32, scorer_hidden2))
        self.use_qavclab = use_qavclab and cross_clip >= 2 and Vitblock_num > 0
        self.use_teacher = self.use_qavclab or enable_teacher_gate or infer_with_teacher
        self.hard_route_infer = hard_route_infer
        self.infer_with_teacher = bool(infer_with_teacher)
        self.infer_teacher_alpha = float(min(max(infer_teacher_alpha, 0.0), 1.0))
        self.infer_with_anchor = bool(infer_with_anchor)
        self.infer_anchor_alpha = float(min(max(infer_anchor_alpha, 0.0), 1.0))
        self.infer_with_recon = bool(infer_with_recon)
        self.infer_recon_alpha = float(min(max(infer_recon_alpha, 0.0), 1.0))
        self.infer_with_memory_novelty = bool(infer_with_memory_novelty)
        self.infer_memory_alpha = float(min(max(infer_memory_alpha, 0.0), 1.0))
        self.infer_memory_frozen_only = bool(infer_memory_frozen_only)
        self.full_evidence_enable = bool(full_evidence_enable)
        self.full_evidence_loss_weight = float(max(0.0, full_evidence_loss_weight))
        self.full_evidence_infer = bool(full_evidence_infer)
        self.full_evidence_alpha = float(min(max(full_evidence_alpha, 0.0), 1.0))
        self.full_evidence_rec_weight = float(max(0.0, full_evidence_rec_weight))
        self.full_evidence_temp_weight = float(max(0.0, full_evidence_temp_weight))
        self.full_evidence_center_weight = float(max(0.0, full_evidence_center_weight))
        self.full_evidence_power = float(max(0.05, full_evidence_power))
        self.decoder = Decoder(batchsize=self.batchsize,embed_dim=feature_size,Vitblock_num=Vitblock_num,cross_clip=cross_clip,split=split)
        if self.use_qavclab:
            self.decoder_short = Decoder(batchsize=self.batchsize, embed_dim=feature_size, Vitblock_num=Vitblock_num, cross_clip=self.short_clip, split=split)
            self.decoder_long = Decoder(batchsize=self.batchsize, embed_dim=feature_size, Vitblock_num=Vitblock_num, cross_clip=self.long_clip, split=split)
        else:
            self.decoder_short = None
            self.decoder_long = None
        self.AE = Autoencoder_B(feature_size)
        self.scorer = Scorer(
            n_feature=feature_size,
            hidden1=scorer_hidden1,
            hidden2=scorer_hidden2,
            dropout_p=scorer_dropout,
        )
        self.scorer_anchor = Scorer(
            n_feature=feature_size,
            hidden1=scorer_hidden1,
            hidden2=scorer_hidden2,
            dropout_p=scorer_dropout,
        ) if inc_enable else None
        self.scorer_teacher = Scorer(
            n_feature=feature_size,
            hidden1=scorer_hidden1,
            hidden2=scorer_hidden2,
            dropout_p=scorer_dropout,
        ) if self.use_teacher else None
        self.router = ClipRouter(feature_size) if self.use_qavclab else None
        self.retrieval_mlp = DynamicRetrievalMLP(retrieval_topk) if self.use_qavclab else None
        self.retrieval_topk = retrieval_topk
        self.retrieval_samples = retrieval_samples
        self.apply(weights_init)
        if self.scorer_teacher is not None:
            self.scorer_teacher.load_state_dict(self.scorer.state_dict())
            for p in self.scorer_teacher.parameters():
                p.requires_grad = False
        if self.scorer_anchor is not None:
            self.scorer_anchor.load_state_dict(self.scorer.state_dict())
            for p in self.scorer_anchor.parameters():
                p.requires_grad = False
            self.scorer_anchor.eval()
        self.numpylist = None
        self.lab = torch.zeros(0)
        self.lab_scores = torch.zeros(0)
        self.lab_age = torch.zeros(0)
        self.mb_size = int(self.batchsize*beta)
        self.delta = delta
        self.score_rec_consistency_weight = float(max(0.0, score_rec_consistency_weight))
        self.bank_decay = bank_decay
        self.bank_age_weight = bank_age_weight
        quality_sum = difficulty_weight + uncertainty_weight + consistency_weight + 1e-6
        self.difficulty_weight = difficulty_weight / quality_sum
        self.uncertainty_weight = uncertainty_weight / quality_sum
        self.consistency_weight = consistency_weight / quality_sum
        self.teacher_momentum = teacher_momentum
        self.teacher_threshold = teacher_threshold
        self.min_pseudo_pos = min_pseudo_pos
        self.max_pseudo_pos = min(max(max_pseudo_pos, 0.0), 1.0)
        self.max_pseudo_pos_start = float(max_pseudo_pos_start)
        self.max_pseudo_pos_end = float(max_pseudo_pos_end)
        self.max_pseudo_pos_anneal_iters = int(max(0, max_pseudo_pos_anneal_iters))
        self.pseudo_loss_mode = str(pseudo_loss_mode).lower().strip()
        if self.pseudo_loss_mode not in ('bce', 'balanced_bce'):
            self.pseudo_loss_mode = 'bce'
        self.pseudo_pos_weight_max = float(max(1.0, pseudo_pos_weight_max))
        self.pseudo_rank_weight = float(max(0.0, pseudo_rank_weight))
        self.pseudo_rank_margin = float(max(0.0, pseudo_rank_margin))
        self.pseudo_rank_max_samples = int(max(16, pseudo_rank_max_samples))
        self.pseudo_rank_neg_ratio = float(max(0.1, pseudo_rank_neg_ratio))
        self.rec_rank_weight = float(max(0.0, rec_rank_weight))
        self.rec_rank_margin = float(max(0.0, rec_rank_margin))
        self.rec_rank_top_ratio = float(min(max(rec_rank_top_ratio, 0.0), 1.0))
        self.rec_rank_bottom_ratio = float(min(max(rec_rank_bottom_ratio, 0.0), 1.0))
        self.rec_rank_max_samples = int(max(8, rec_rank_max_samples))
        self.temporal_rank_weight = float(max(0.0, temporal_rank_weight))
        self.temporal_rank_margin = float(max(0.0, temporal_rank_margin))
        self.temporal_rank_top_ratio = float(min(max(temporal_rank_top_ratio, 0.0), 1.0))
        self.temporal_rank_bottom_ratio = float(min(max(temporal_rank_bottom_ratio, 0.0), 1.0))
        self.temporal_rank_max_samples = int(max(8, temporal_rank_max_samples))
        self.temporal_rank_context_mix = float(min(max(temporal_rank_context_mix, 0.0), 1.0))
        self.easy_neg_weight = float(max(0.0, easy_neg_weight))
        self.easy_neg_ratio = float(min(max(easy_neg_ratio, 0.0), 1.0))
        self.easy_neg_margin = float(min(max(easy_neg_margin, 0.0), 1.0))
        self.easy_neg_max_samples = int(max(8, easy_neg_max_samples))
        self.easy_neg_focus_ratio = float(min(max(easy_neg_focus_ratio, 0.0), 1.0))
        self.score_sparse_weight = float(max(0.0, score_sparse_weight))
        self.score_sparse_target = float(min(max(score_sparse_target, 0.0), 1.0))
        self.score_sparse_flat_weight = float(max(0.0, score_sparse_flat_weight))
        self.score_sparse_flat_margin = float(max(1e-6, score_sparse_flat_margin))
        self.score_sparse_top_ratio = float(min(max(score_sparse_top_ratio, 0.01), 0.49))
        self.score_sparse_rec_aware = float(min(max(score_sparse_rec_aware, 0.0), 1.0))
        self.score_sparse_min_weight = float(min(max(score_sparse_min_weight, 0.0), 1.0))
        self.score_baseline_weight = float(max(0.0, score_baseline_weight))
        self.score_baseline_margin = float(min(max(score_baseline_margin, 0.0), 1.0))
        self.score_baseline_quantile = float(min(max(score_baseline_quantile, 0.0), 1.0))
        self.score_baseline_rec_ratio = float(min(max(score_baseline_rec_ratio, 0.0), 1.0))
        self.score_baseline_top_ratio = float(min(max(score_baseline_top_ratio, 0.01), 1.0))
        self.score_baseline_tail_margin = float(min(max(score_baseline_tail_margin, 0.0), 1.0))
        self.enforce_min_pseudo_pos = bool(enforce_min_pseudo_pos)
        self.lab_start_iter = lab_start_iter
        self.route_start_iter = route_start_iter
        self.enable_teacher_gate = enable_teacher_gate
        self.teacher_gate_start_iter = teacher_gate_start_iter
        self.route_batch_cap = route_batch_cap
        self.iter = 0
        self.debug_pseudo_pos = 0.0
        self.debug_aux_pos = 0.0
        self.debug_teacher_pos = 0.0
        self.debug_rank_loss = 0.0
        self.debug_full_evidence_loss = 0.0
        self.inc_enable = bool(inc_enable)
        self.inc_scene_id = int(inc_scene_id)
        self.inc_memory_max = int(max(1, inc_memory_max))
        self.inc_memory_init = int(max(1, inc_memory_init))
        self.inc_match_temp = float(max(1e-6, inc_match_temp))
        self.inc_match_threshold = float(min(max(inc_match_threshold, 0.0), 1.0))
        self.inc_match_sim_threshold = float(min(max(inc_match_sim_threshold, -1.0), 1.0))
        self.inc_expand_warmup_iters = int(max(0, inc_expand_warmup_iters))
        self.inc_proto_momentum = float(min(max(inc_proto_momentum, 0.0), 0.9999))
        self.inc_mem_easy_ratio = float(min(max(inc_mem_easy_ratio, 0.0), 1.0))
        self.inc_mem_score_quantile = float(min(max(inc_mem_score_quantile, 0.0), 1.0))
        self.inc_mem_score_weight = float(max(0.0, inc_mem_score_weight))
        self.inc_replay_ratio = float(max(0.0, inc_replay_ratio))
        self.inc_replay_max_samples = int(max(1, inc_replay_max_samples))
        self.inc_replay_scene_quota = int(max(1, inc_replay_scene_quota))
        self.inc_replay_topk = int(max(1, inc_replay_topk))
        self.inc_replay_alpha = float(min(max(inc_replay_alpha, 0.0), 1.0))
        self.inc_replay_noise = float(max(0.0, inc_replay_noise))
        self.inc_replay_min_sim = float(inc_replay_min_sim)
        self.inc_replay_hard_ratio = float(min(max(inc_replay_hard_ratio, 0.0), 1.0))
        self.inc_replay_weight = float(max(0.0, inc_replay_weight))
        self.inc_replay_warmup_iters = int(max(0, inc_replay_warmup_iters))
        self.inc_replay_score_weight = float(max(0.0, inc_replay_score_weight))
        self.inc_replay_loss_mode = str(inc_replay_loss_mode).lower().strip()
        if self.inc_replay_loss_mode not in ('full', 'rec', 'anchor_only', 'none'):
            self.inc_replay_loss_mode = 'full'
        self.inc_replay_putback_memory = bool(inc_replay_putback_memory)
        self.inc_expected_scenes = int(max(1, inc_expected_scenes))
        self.inc_scene_proto_cap = int(inc_scene_proto_cap)
        self.inc_scene_min_slots = int(max(0, inc_scene_min_slots))
        self.inc_overflow_policy = str(inc_overflow_policy).lower().strip()
        if self.inc_overflow_policy not in ('far', 'similar', 'redundant'):
            self.inc_overflow_policy = 'far'
        self.inc_overflow_expand = bool(inc_overflow_expand)
        self.inc_anchor_replay_weight = float(max(0.0, inc_anchor_replay_weight))
        self.inc_anchor_active = False
        self.debug_inc_memory = 0
        self.debug_inc_replay = 0
        self.debug_inc_replay_putback = 0
        self.register_buffer('inc_memory_bank', torch.zeros(self.inc_memory_max, self.feature_size))
        self.register_buffer('inc_memory_scene', torch.full((self.inc_memory_max,), -1, dtype=torch.long))
        self.register_buffer('inc_memory_frozen', torch.zeros(self.inc_memory_max, dtype=torch.bool))
        self.register_buffer('inc_memory_count', torch.zeros(1, dtype=torch.long))

    def __del__(self):
        for i in range(10):
            torch.cuda.empty_cache()

    def _project_inputs(self, x):
        orig_shape = x.shape
        x = x.reshape(-1, orig_shape[-1]).float()
        x = self.input_proj(x)
        if self.input_norm_mode == 'layernorm' and self.input_ln is not None:
            x = self.input_ln(x)
        elif self.input_norm_mode == 'l2':
            x = F.normalize(x, p=2, dim=-1)
        x = x.reshape(*orig_shape[:-1], self.feature_size)
        return x

    @torch.no_grad()
    def update_teacher(self):
        # Skip EMA update when teacher is unused to reduce overhead in baseline runs.
        if self.scorer_teacher is None:
            return
        m = self.teacher_momentum
        for t, s in zip(self.scorer_teacher.parameters(), self.scorer.parameters()):
            t.data.mul_(m).add_(s.data, alpha=1.0 - m)

    def _pad_clip_dim(self, features, target_clip):
        if features.shape[2] == target_clip:
            return features
        if features.shape[2] > target_clip:
            return features[:, :, :target_clip, :]
        pad_num = target_clip - features.shape[2]
        pad_token = features[:, :, -1:, :].repeat(1, 1, pad_num, 1)
        return torch.cat([features, pad_token], dim=2)

    def _pad_error_dim(self, err, target_clip):
        if err.shape[1] == target_clip:
            return err
        if err.shape[1] > target_clip:
            return err[:, :target_clip]
        pad_num = target_clip - err.shape[1]
        return torch.cat([err, err[:, -1:].repeat(1, pad_num)], dim=1)

    def _empty_like_scalar(self, ref_tensor):
        return ref_tensor.new_zeros(())

    def _inc_count(self):
        return int(self.inc_memory_count.item())

    def _inc_scene_count(self, scene_id=None):
        count = self._inc_count()
        if count <= 0:
            return 0
        sid = int(self.inc_scene_id if scene_id is None else scene_id)
        return int(torch.sum(self.inc_memory_scene[:count] == sid).item())

    def _inc_scene_cap(self):
        if self.inc_scene_proto_cap > 0:
            return int(min(self.inc_memory_max, max(1, self.inc_scene_proto_cap)))
        auto_cap = (self.inc_memory_max + self.inc_expected_scenes - 1) // self.inc_expected_scenes
        auto_cap = max(self.inc_memory_init, auto_cap)
        return int(min(self.inc_memory_max, max(1, auto_cap)))

    @torch.no_grad()
    def _inc_select_overflow_slot(self, feat):
        count = self._inc_count()
        if count <= 0:
            return None
        scene_ids = self.inc_memory_scene[:count]
        unique_scene_ids = torch.unique(scene_ids)
        donor_scene = None
        donor_count = 0
        for sid_t in unique_scene_ids:
            sid = int(sid_t.item())
            if sid >= int(self.inc_scene_id):
                continue
            c = int(torch.sum(scene_ids == sid).item())
            if c > 1 and c > donor_count:
                donor_scene = sid
                donor_count = c
        if donor_scene is None:
            return None
        donor_idx = torch.nonzero(scene_ids == donor_scene, as_tuple=False).squeeze(1)
        if donor_idx.numel() == 0:
            return None
        donor_bank = self.inc_memory_bank[donor_idx]
        if self.inc_overflow_policy == 'redundant' and donor_idx.numel() >= 2:
            # Remove the most redundant old-domain prototype, preserving rare
            # boundary prototypes that are dissimilar to the rest of the scene.
            sim_mat = torch.matmul(donor_bank, donor_bank.t())
            sim_mat.fill_diagonal_(-2.0)
            redundancy = torch.max(sim_mat, dim=1).values
            replace_idx = donor_idx[torch.argmax(redundancy)]
        else:
            donor_sim = torch.matmul(donor_bank, feat)
            if self.inc_overflow_policy == 'similar':
                replace_idx = donor_idx[torch.argmax(donor_sim)]
            else:
                replace_idx = donor_idx[torch.argmin(donor_sim)]
        return int(replace_idx.item())

    @torch.no_grad()
    def set_incremental_scene(self, scene_id):
        self.inc_scene_id = int(scene_id)
        self._inc_refresh_anchor()
        self._inc_freeze_old_scenes()

    @torch.no_grad()
    def _inc_refresh_anchor(self):
        if self.scorer_anchor is None:
            self.inc_anchor_active = False
            return
        self.scorer_anchor.load_state_dict(self.scorer.state_dict(), strict=False)
        self.scorer_anchor.eval()
        self.inc_anchor_active = bool(self.inc_scene_id > 0)

    @torch.no_grad()
    def _inc_freeze_old_scenes(self):
        if not self.inc_enable:
            return
        count = self._inc_count()
        if count <= 0:
            return
        scene_ids = self.inc_memory_scene[:count]
        self.inc_memory_frozen[:count] = (scene_ids < self.inc_scene_id)

    @torch.no_grad()
    def _inc_diverse_seed_indices(self, pooled, add_n):
        # Greedy farthest-point sampling in cosine space for diverse memory bootstrap.
        n = int(pooled.shape[0])
        k = int(max(1, min(add_n, n)))
        if k >= n:
            return torch.arange(n, device=pooled.device, dtype=torch.long)

        center = F.normalize(pooled.mean(dim=0, keepdim=True), p=2, dim=-1)
        sim_center = torch.matmul(pooled, center.t()).squeeze(1)
        first = int(torch.argmin(sim_center).item())
        selected = [first]

        dist = 1.0 - torch.matmul(pooled, pooled[first].unsqueeze(1)).squeeze(1)
        for _ in range(1, k):
            if len(selected) > 0:
                dist[torch.tensor(selected, device=pooled.device, dtype=torch.long)] = -1.0
            nxt = int(torch.argmax(dist).item())
            selected.append(nxt)
            new_dist = 1.0 - torch.matmul(pooled, pooled[nxt].unsqueeze(1)).squeeze(1)
            dist = torch.minimum(dist, new_dist)
        return torch.tensor(selected, device=pooled.device, dtype=torch.long)

    @torch.no_grad()
    def _inc_bootstrap_from_batch(self, pooled):
        if not self.inc_enable:
            return
        if pooled.numel() == 0:
            return
        count = self._inc_count()
        if count > 0:
            return
        add_n = min(self.inc_memory_max, self.inc_memory_init, pooled.shape[0])
        if add_n <= 0:
            return
        pick = self._inc_diverse_seed_indices(pooled, add_n)
        proto = F.normalize(pooled.index_select(0, pick).detach(), p=2, dim=-1)
        self.inc_memory_bank[:add_n] = proto
        self.inc_memory_scene[:add_n] = int(self.inc_scene_id)
        self.inc_memory_frozen[:add_n] = False
        self.inc_memory_count[0] = int(add_n)
        self._inc_freeze_old_scenes()

    @torch.no_grad()
    def _inc_update_memory(self, pooled):
        if not self.inc_enable:
            return
        if pooled.numel() == 0:
            return
        pooled = F.normalize(pooled.detach(), p=2, dim=-1)
        self._inc_bootstrap_from_batch(pooled)

        for feat in pooled:
            count = self._inc_count()
            if count <= 0:
                self._inc_bootstrap_from_batch(feat.unsqueeze(0))
                continue

            bank = self.inc_memory_bank[:count]
            sim = torch.matmul(bank, feat)  # [M]
            prob = torch.softmax(sim / self.inc_match_temp, dim=0)
            max_prob, best_idx = torch.max(prob, dim=0)
            best_idx = int(best_idx.item())
            max_prob_val = float(max_prob.item())
            max_sim_val = float(torch.max(sim).item())
            scene_count = self._inc_scene_count(self.inc_scene_id)
            scene_cap = self._inc_scene_cap()
            cur_mask_for_expand = (~self.inc_memory_frozen[:count]) & (self.inc_memory_scene[:count] == int(self.inc_scene_id))
            if cur_mask_for_expand.any():
                cur_idx_for_expand = torch.nonzero(cur_mask_for_expand, as_tuple=False).squeeze(1)
                cur_max_sim_val = float(torch.max(torch.matmul(self.inc_memory_bank[cur_idx_for_expand], feat)).item())
            else:
                cur_max_sim_val = -1.0

            if scene_count < self.inc_scene_min_slots:
                if count < self.inc_memory_max:
                    self.inc_memory_bank[count] = feat
                    self.inc_memory_scene[count] = int(self.inc_scene_id)
                    self.inc_memory_frozen[count] = False
                    self.inc_memory_count[0] = count + 1
                    continue
                overflow_idx = self._inc_select_overflow_slot(feat)
                if overflow_idx is not None:
                    self.inc_memory_bank[overflow_idx] = feat
                    self.inc_memory_scene[overflow_idx] = int(self.inc_scene_id)
                    self.inc_memory_frozen[overflow_idx] = False
                continue

            can_expand = (
                ((max_prob_val < self.inc_match_threshold) or (max_sim_val < self.inc_match_sim_threshold))
                and count < self.inc_memory_max
                and self.iter >= self.inc_expand_warmup_iters
                and scene_count < scene_cap
            )

            if can_expand:
                self.inc_memory_bank[count] = feat
                self.inc_memory_scene[count] = int(self.inc_scene_id)
                self.inc_memory_frozen[count] = False
                self.inc_memory_count[0] = count + 1
                continue

            can_overflow_expand = (
                self.inc_overflow_expand
                and count >= self.inc_memory_max
                and self.iter >= self.inc_expand_warmup_iters
                and scene_count < scene_cap
                and (
                    (max_prob_val < self.inc_match_threshold)
                    or (max_sim_val < self.inc_match_sim_threshold)
                    or (cur_max_sim_val < self.inc_match_sim_threshold)
                )
            )

            if can_overflow_expand:
                overflow_idx = self._inc_select_overflow_slot(feat)
                if overflow_idx is not None:
                    self.inc_memory_bank[overflow_idx] = feat
                    self.inc_memory_scene[overflow_idx] = int(self.inc_scene_id)
                    self.inc_memory_frozen[overflow_idx] = False
                    continue

            if bool(self.inc_memory_frozen[best_idx]):
                cur_mask = (~self.inc_memory_frozen[:count]) & (self.inc_memory_scene[:count] == int(self.inc_scene_id))
                if cur_mask.any():
                    cand = torch.nonzero(cur_mask, as_tuple=False).squeeze(1)
                    cand_sim = torch.matmul(self.inc_memory_bank[cand], feat)
                    best_idx = int(cand[torch.argmax(cand_sim)].item())
                elif count < self.inc_memory_max and scene_count < scene_cap:
                    self.inc_memory_bank[count] = feat
                    self.inc_memory_scene[count] = int(self.inc_scene_id)
                    self.inc_memory_frozen[count] = False
                    self.inc_memory_count[0] = count + 1
                    continue
                else:
                    overflow_idx = self._inc_select_overflow_slot(feat)
                    if overflow_idx is not None:
                        self.inc_memory_bank[overflow_idx] = feat
                        self.inc_memory_scene[overflow_idx] = int(self.inc_scene_id)
                        self.inc_memory_frozen[overflow_idx] = False
                    continue

            old = self.inc_memory_bank[best_idx]
            updated = self.inc_proto_momentum * old + (1.0 - self.inc_proto_momentum) * feat
            self.inc_memory_bank[best_idx] = F.normalize(updated, p=2, dim=-1)

    @torch.no_grad()
    def _inc_generate_replay(self, pooled, query_priority=None, return_scene=False):
        def empty_replay():
            replay = pooled.new_zeros((0, self.feature_size))
            if return_scene:
                scene = self.inc_memory_scene.new_full((0,), -1)
                return replay, scene
            return replay

        if not self.inc_enable or self.inc_replay_ratio <= 0:
            return empty_replay()
        count = self._inc_count()
        if count <= 0:
            return empty_replay()
        frozen_mask = self.inc_memory_frozen[:count]
        if not frozen_mask.any():
            return empty_replay()
        old_bank_full = self.inc_memory_bank[:count][frozen_mask]
        old_scene_full = self.inc_memory_scene[:count][frozen_mask]
        selected_idx = []
        unique_scene = torch.unique(old_scene_full)
        for sid_t in unique_scene:
            sid = int(sid_t.item())
            idx = torch.nonzero(old_scene_full == sid, as_tuple=False).squeeze(1)
            if idx.numel() == 0:
                continue
            if idx.numel() > self.inc_replay_scene_quota:
                perm = torch.randperm(idx.numel(), device=idx.device)[:self.inc_replay_scene_quota]
                idx = idx.index_select(0, perm)
            selected_idx.append(idx)
        if len(selected_idx) == 0:
            return empty_replay()
        selected_idx = torch.cat(selected_idx, dim=0)
        old_bank = old_bank_full.index_select(0, selected_idx)
        old_scene = old_scene_full.index_select(0, selected_idx)
        if old_bank.numel() == 0:
            return empty_replay()

        bsz = int(pooled.shape[0])
        if bsz <= 0:
            return empty_replay()
        replay_n = max(1, int(round(self.inc_replay_ratio * max(1, bsz))))
        replay_n = min(replay_n, bsz, self.inc_replay_max_samples)
        if replay_n >= bsz:
            q_idx = torch.arange(bsz, device=pooled.device, dtype=torch.long)
        else:
            hard_n = int(round(self.inc_replay_hard_ratio * float(replay_n)))
            hard_n = max(0, min(replay_n, hard_n))
            use_hard = (
                query_priority is not None
                and hard_n > 0
                and torch.is_tensor(query_priority)
                and query_priority.numel() == bsz
            )
            if use_hard:
                q_score = query_priority.detach().reshape(-1).float().to(pooled.device)
                hard_n = min(hard_n, bsz)
                hard_idx = torch.topk(q_score, k=hard_n, dim=0).indices
                if hard_n >= replay_n:
                    q_idx = hard_idx
                else:
                    remain = replay_n - hard_n
                    cand_mask = torch.ones(bsz, device=pooled.device, dtype=torch.bool)
                    cand_mask[hard_idx] = False
                    cand = torch.nonzero(cand_mask, as_tuple=False).squeeze(1)
                    if cand.numel() > remain:
                        perm = torch.randperm(cand.numel(), device=pooled.device)[:remain]
                        rand_idx = cand.index_select(0, perm)
                    else:
                        rand_idx = cand
                    q_idx = torch.cat([hard_idx, rand_idx], dim=0)
            else:
                q_idx = torch.randperm(bsz, device=pooled.device)[:replay_n]
        if q_idx.numel() == 0:
            return empty_replay()
        query = F.normalize(pooled[q_idx].detach(), p=2, dim=-1)

        sim = torch.matmul(query, old_bank.t())
        topk = min(self.inc_replay_topk, old_bank.shape[0])
        topv, topi = torch.topk(sim, k=topk, dim=1)
        if self.inc_replay_min_sim > -1.0:
            keep = topv[:, 0] >= float(self.inc_replay_min_sim)
            if not torch.any(keep):
                return empty_replay()
            topv = topv[keep]
            topi = topi[keep]
            query = query[keep]
            if query.shape[0] == 0:
                return empty_replay()
        weights = torch.softmax(topv / self.inc_match_temp, dim=1)
        old_mix = (old_bank[topi] * weights.unsqueeze(-1)).sum(dim=1)

        replay = self.inc_replay_alpha * old_mix + (1.0 - self.inc_replay_alpha) * query
        if self.inc_replay_noise > 0:
            replay = replay + torch.randn_like(replay) * self.inc_replay_noise
        replay = F.normalize(replay, p=2, dim=-1)
        if return_scene:
            replay_scene = old_scene[topi[:, 0]].detach()
            return replay, replay_scene
        return replay

    @torch.no_grad()
    def _inc_putback_replay_memory(self, replay_pooled, replay_scene):
        if not self.inc_enable or not self.inc_replay_putback_memory:
            return 0
        if replay_pooled.numel() == 0 or replay_scene is None:
            return 0
        if int(replay_pooled.shape[0]) != int(replay_scene.numel()):
            return 0

        replay_pooled = F.normalize(replay_pooled.detach(), p=2, dim=-1)
        replay_scene = replay_scene.detach().reshape(-1).to(replay_pooled.device, dtype=torch.long)
        putback_count = 0
        current_scene = int(self.inc_scene_id)

        for feat, sid_t in zip(replay_pooled, replay_scene):
            sid = int(sid_t.item())
            if sid < 0 or sid >= current_scene:
                continue

            count = self._inc_count()
            frozen_value = bool(sid < current_scene)
            if count <= 0:
                self.inc_memory_bank[0] = feat
                self.inc_memory_scene[0] = sid
                self.inc_memory_frozen[0] = frozen_value
                self.inc_memory_count[0] = 1
                putback_count += 1
                continue

            scene_mask = self.inc_memory_scene[:count] == sid
            if not scene_mask.any():
                if count < self.inc_memory_max:
                    self.inc_memory_bank[count] = feat
                    self.inc_memory_scene[count] = sid
                    self.inc_memory_frozen[count] = frozen_value
                    self.inc_memory_count[0] = count + 1
                    putback_count += 1
                continue

            scene_idx = torch.nonzero(scene_mask, as_tuple=False).squeeze(1)
            scene_bank = self.inc_memory_bank[scene_idx]
            sim = torch.matmul(scene_bank, feat)
            prob = torch.softmax(sim / self.inc_match_temp, dim=0)
            max_prob_val = float(torch.max(prob).item())
            max_sim_val = float(torch.max(sim).item())
            scene_count = int(scene_idx.numel())
            scene_cap = self._inc_scene_cap()

            can_expand = (
                ((max_prob_val < self.inc_match_threshold) or (max_sim_val < self.inc_match_sim_threshold))
                and count < self.inc_memory_max
                and self.iter >= self.inc_expand_warmup_iters
                and scene_count < scene_cap
            )
            if can_expand:
                self.inc_memory_bank[count] = feat
                self.inc_memory_scene[count] = sid
                self.inc_memory_frozen[count] = frozen_value
                self.inc_memory_count[0] = count + 1
                putback_count += 1
                continue

            can_overflow_expand = (
                self.inc_overflow_expand
                and count >= self.inc_memory_max
                and self.iter >= self.inc_expand_warmup_iters
                and scene_count < scene_cap
                and ((max_prob_val < self.inc_match_threshold) or (max_sim_val < self.inc_match_sim_threshold))
            )
            if can_overflow_expand:
                overflow_idx = self._inc_select_overflow_slot(feat)
                if overflow_idx is not None:
                    self.inc_memory_bank[overflow_idx] = feat
                    self.inc_memory_scene[overflow_idx] = sid
                    self.inc_memory_frozen[overflow_idx] = frozen_value
                    putback_count += 1
                    continue

            best_local = int(torch.argmax(sim).item())
            best_idx = int(scene_idx[best_local].item())
            old = self.inc_memory_bank[best_idx]
            updated = self.inc_proto_momentum * old + (1.0 - self.inc_proto_momentum) * feat
            self.inc_memory_bank[best_idx] = F.normalize(updated, p=2, dim=-1)
            self.inc_memory_scene[best_idx] = sid
            self.inc_memory_frozen[best_idx] = frozen_value
            putback_count += 1

        return int(putback_count)

    @torch.no_grad()
    def _inc_memory_replay_step(self, pooled, replay_priority, sample_score=None):
        mem_pooled = self._inc_select_memory_update_subset(
            pooled,
            replay_priority,
            sample_score=sample_score,
        )
        self._inc_update_memory(mem_pooled)
        if self.inc_replay_putback_memory:
            replay_pooled, replay_scene = self._inc_generate_replay(
                pooled,
                query_priority=replay_priority,
                return_scene=True,
            )
            putback_count = self._inc_putback_replay_memory(replay_pooled, replay_scene)
        else:
            replay_pooled = self._inc_generate_replay(pooled, query_priority=replay_priority)
            putback_count = 0
        self.debug_inc_memory = self._inc_count()
        self.debug_inc_replay = int(replay_pooled.shape[0])
        self.debug_inc_replay_putback = int(putback_count)
        return replay_pooled

    def _inc_current_replay_weight(self):
        if self.inc_replay_weight <= 0:
            return 0.0
        if self.inc_replay_warmup_iters <= 0:
            return self.inc_replay_weight
        ramp = min(1.0, float(max(0, self.iter)) / float(self.inc_replay_warmup_iters))
        return float(self.inc_replay_weight * ramp)

    def _inc_replay_loss(self, replay_pooled):
        if replay_pooled.numel() == 0:
            return replay_pooled.new_zeros(())
        if self.inc_replay_loss_mode == 'none':
            return replay_pooled.new_zeros(())
        replay_inputs = replay_pooled.unsqueeze(1).unsqueeze(1).repeat(1, 10, self.cross_clip, 1)
        rec = replay_pooled.new_zeros(())
        if self.inc_replay_loss_mode in ('full', 'rec'):
            if self.Vitblock_num == 0:
                replay_output = self.AE(replay_inputs)
            else:
                replay_output = self.decoder(replay_inputs.reshape(-1, self.cross_clip, self.feature_size))
                replay_output = replay_output.reshape(-1, 10, self.cross_clip, self.feature_size)
            rec = F.mse_loss(replay_output, replay_inputs)

        need_student = self.inc_replay_loss_mode in ('full', 'anchor_only')
        student_scores = self.scorer(replay_inputs, is_training=True) if need_student else None
        score = student_scores.mean() if self.inc_replay_loss_mode == 'full' else rec.new_zeros(())
        if self.inc_enable and self.inc_anchor_active and self.scorer_anchor is not None and self.inc_anchor_replay_weight > 0:
            with torch.no_grad():
                anchor_scores = self.scorer_anchor(replay_inputs, is_training=False)
            if student_scores is None:
                student_scores = self.scorer(replay_inputs, is_training=True)
            replay_distill = F.mse_loss(student_scores.float(), anchor_scores.detach().float())
        else:
            replay_distill = rec.new_zeros(())
        if self.inc_replay_loss_mode == 'anchor_only':
            return self.inc_anchor_replay_weight * replay_distill
        if self.inc_replay_loss_mode == 'rec':
            return rec + self.inc_anchor_replay_weight * replay_distill
        return rec + self.inc_replay_score_weight * score + self.inc_anchor_replay_weight * replay_distill

    @torch.no_grad()
    def _inc_select_memory_update_subset(self, pooled, sample_err, sample_score=None):
        if pooled.numel() == 0:
            return pooled
        if self.inc_mem_easy_ratio >= 0.999:
            return pooled
        n = int(pooled.shape[0])
        keep_n = max(1, int(round(float(n) * self.inc_mem_easy_ratio)))
        keep_n = min(keep_n, n)
        if keep_n >= n:
            return pooled
        err = sample_err.detach().reshape(-1).float().to(pooled.device)
        if err.numel() != n:
            return pooled
        rank = err
        candidate_idx = torch.arange(n, device=pooled.device, dtype=torch.long)
        if sample_score is not None and torch.is_tensor(sample_score):
            sc = sample_score.detach().reshape(-1).float().to(pooled.device)
            if sc.numel() == n:
                # Prefer memory candidates with both low reconstruction error and low anomaly score.
                err_n = (err - err.mean()) / (err.std() + 1e-6)
                sc_n = (sc - sc.mean()) / (sc.std() + 1e-6)
                rank = err_n + self.inc_mem_score_weight * sc_n
                # Additional anti-pollution guard: keep memory updates in low-score majority.
                if n >= 4 and self.inc_mem_score_quantile > 0:
                    q_th = torch.quantile(sc, self.inc_mem_score_quantile)
                    cand_mask = (sc <= q_th)
                    if int(cand_mask.sum().item()) >= keep_n:
                        candidate_idx = torch.nonzero(cand_mask, as_tuple=False).squeeze(1)
        cand_rank = rank.index_select(0, candidate_idx)
        local_idx = torch.topk(cand_rank, k=keep_n, largest=False, dim=0).indices
        idx = candidate_idx.index_select(0, local_idx)
        return pooled.index_select(0, idx)

    @torch.no_grad()
    def _rec_soft_target(self, rec_err):
        # rec_err: [B,C], larger means more abnormal
        m = rec_err.mean(dim=1, keepdim=True)
        s = rec_err.std(dim=1, keepdim=True) + 1e-6
        z = (rec_err - m) / s
        return torch.sigmoid(z).unsqueeze(-1)

    def _pseudo_scorer_loss(self, pred_scores, pseudo_labels):
        pred = pred_scores.float()
        tgt = pseudo_labels.float()
        if self.pseudo_loss_mode != 'balanced_bce':
            return F.binary_cross_entropy(pred, tgt)
        pos_ratio = float(tgt.mean().item())
        pos_ratio = min(max(pos_ratio, 1e-6), 1.0 - 1e-6)
        pos_w = (1.0 - pos_ratio) / pos_ratio
        pos_w = float(min(self.pseudo_pos_weight_max, max(1.0, pos_w)))
        weight = 1.0 + (pos_w - 1.0) * tgt
        return F.binary_cross_entropy(pred, tgt, weight=weight)

    def _pseudo_rank_loss(self, pred_scores, pseudo_labels):
        pred = pred_scores.float().squeeze(-1).reshape(-1)
        tgt = pseudo_labels.float().reshape(-1)
        pos_mask = (tgt > 0.5)
        neg_mask = ~pos_mask
        if int(pos_mask.sum().item()) == 0 or int(neg_mask.sum().item()) == 0:
            return self._empty_like_scalar(pred_scores)

        pos_scores = pred[pos_mask]
        neg_scores = pred[neg_mask]

        keep_pos = min(int(pos_scores.numel()), int(self.pseudo_rank_max_samples))
        keep_neg = int(max(1, round(self.pseudo_rank_neg_ratio * keep_pos)))
        keep_neg = min(keep_neg, int(neg_scores.numel()), int(self.pseudo_rank_max_samples))

        if keep_pos < int(pos_scores.numel()):
            pos_scores = torch.topk(pos_scores, k=keep_pos, largest=False, dim=0).values
        if keep_neg < int(neg_scores.numel()):
            neg_scores = torch.topk(neg_scores, k=keep_neg, largest=True, dim=0).values

        margin_gap = self.pseudo_rank_margin - (pos_scores.unsqueeze(1) - neg_scores.unsqueeze(0))
        return F.relu(margin_gap).mean()

    def _rec_contrast_rank_loss(self, pred_scores, rec_scores):
        if self.rec_rank_weight <= 0:
            return self._empty_like_scalar(pred_scores)

        pred = pred_scores.float().squeeze(-1).reshape(-1)
        rec = rec_scores.float().reshape(-1).detach()
        n = int(pred.numel())
        if n < 4:
            return self._empty_like_scalar(pred_scores)

        top_n = int(round(self.rec_rank_top_ratio * n))
        bottom_n = int(round(self.rec_rank_bottom_ratio * n))
        top_n = max(1, min(top_n, n // 2, self.rec_rank_max_samples))
        bottom_n = max(1, min(bottom_n, n - top_n, self.rec_rank_max_samples))
        if top_n <= 0 or bottom_n <= 0:
            return self._empty_like_scalar(pred_scores)

        top_idx = torch.topk(rec, k=top_n, largest=True, dim=0).indices
        bottom_idx = torch.topk(rec, k=bottom_n, largest=False, dim=0).indices
        pos_scores = pred[top_idx]
        neg_scores = pred[bottom_idx]

        keep_pos = min(int(pos_scores.numel()), int(self.rec_rank_max_samples))
        keep_neg = min(int(neg_scores.numel()), int(self.rec_rank_max_samples))
        if keep_pos < int(pos_scores.numel()):
            pos_scores = torch.topk(pos_scores, k=keep_pos, largest=False, dim=0).values
        if keep_neg < int(neg_scores.numel()):
            neg_scores = torch.topk(neg_scores, k=keep_neg, largest=True, dim=0).values

        margin_gap = self.rec_rank_margin - (pos_scores.unsqueeze(1) - neg_scores.unsqueeze(0))
        return F.relu(margin_gap).mean()

    def _temporal_saliency_rank_loss(self, pred_scores, inputs):
        if self.temporal_rank_weight <= 0:
            return self._empty_like_scalar(pred_scores)
        if inputs is None or inputs.dim() != 4:
            return self._empty_like_scalar(pred_scores)

        pred = pred_scores.float().squeeze(-1)
        if pred.dim() == 1:
            pred = pred.unsqueeze(0)
        pred = pred.reshape(pred.shape[0], -1)
        n = int(pred.numel())
        if n < 4:
            return self._empty_like_scalar(pred_scores)

        with torch.no_grad():
            # Inputs are [B, 10, C, F]. Average the 10 views and score temporal
            # clips by context deviation plus adjacent feature change.
            feat = inputs.float().detach().mean(dim=1)
            feat = F.normalize(feat, p=2, dim=-1)
            context = F.normalize(feat.mean(dim=1, keepdim=True), p=2, dim=-1)
            context_dist = 1.0 - torch.sum(feat * context, dim=-1)

            if feat.shape[1] > 1:
                adj_dist = 1.0 - torch.sum(feat[:, 1:, :] * feat[:, :-1, :], dim=-1)
                diff_score = torch.zeros_like(context_dist)
                diff_count = torch.zeros_like(context_dist)
                diff_score[:, 1:] = diff_score[:, 1:] + adj_dist
                diff_score[:, :-1] = diff_score[:, :-1] + adj_dist
                diff_count[:, 1:] = diff_count[:, 1:] + 1.0
                diff_count[:, :-1] = diff_count[:, :-1] + 1.0
                diff_score = diff_score / diff_count.clamp_min(1.0)
            else:
                diff_score = torch.zeros_like(context_dist)

            mix = self.temporal_rank_context_mix
            saliency = mix * context_dist + (1.0 - mix) * diff_score
            saliency = saliency.reshape(-1)

        pred = pred.reshape(-1)
        usable = min(int(pred.numel()), int(saliency.numel()))
        if usable < 4:
            return self._empty_like_scalar(pred_scores)
        pred = pred[:usable]
        saliency = saliency[:usable]

        top_n = int(round(self.temporal_rank_top_ratio * usable))
        bottom_n = int(round(self.temporal_rank_bottom_ratio * usable))
        top_n = max(1, min(top_n, usable // 2, self.temporal_rank_max_samples))
        bottom_n = max(1, min(bottom_n, usable - top_n, self.temporal_rank_max_samples))
        if top_n <= 0 or bottom_n <= 0:
            return self._empty_like_scalar(pred_scores)

        top_idx = torch.topk(saliency, k=top_n, largest=True, dim=0).indices
        bottom_idx = torch.topk(saliency, k=bottom_n, largest=False, dim=0).indices
        pos_scores = pred[top_idx]
        neg_scores = pred[bottom_idx]

        keep_pos = min(int(pos_scores.numel()), self.temporal_rank_max_samples)
        keep_neg = min(int(neg_scores.numel()), self.temporal_rank_max_samples)
        if keep_pos < int(pos_scores.numel()):
            pos_scores = torch.topk(pos_scores, k=keep_pos, largest=False, dim=0).values
        if keep_neg < int(neg_scores.numel()):
            neg_scores = torch.topk(neg_scores, k=keep_neg, largest=True, dim=0).values

        margin_gap = self.temporal_rank_margin - (pos_scores.unsqueeze(1) - neg_scores.unsqueeze(0))
        return F.relu(margin_gap).mean()

    def _easy_negative_loss(self, pred_scores, rec_scores):
        if self.easy_neg_weight <= 0:
            return self._empty_like_scalar(pred_scores)
        pred = pred_scores.float().squeeze(-1).reshape(-1)
        rec = rec_scores.float().reshape(-1).detach()
        n = int(pred.numel())
        if n < 4:
            return self._empty_like_scalar(pred_scores)
        neg_n = int(round(self.easy_neg_ratio * n))
        neg_n = max(1, min(neg_n, n - 1, self.easy_neg_max_samples))
        easy_idx = torch.topk(rec, k=neg_n, largest=False, dim=0).indices
        easy_scores = pred[easy_idx]
        if self.easy_neg_focus_ratio < 0.999 and easy_scores.numel() > 1:
            # Low reconstruction error should be normal; focus the suppression on
            # the high-score tail of those easy samples rather than every sample.
            focus_n = int(round(self.easy_neg_focus_ratio * int(easy_scores.numel())))
            focus_n = max(1, min(focus_n, int(easy_scores.numel())))
            easy_scores = torch.topk(easy_scores, k=focus_n, largest=True, dim=0).values
        return torch.mean(F.relu(easy_scores - self.easy_neg_margin) ** 2)

    def _score_sparsity_loss(self, pred_scores, rec_scores=None):
        if self.score_sparse_weight <= 0:
            return self._empty_like_scalar(pred_scores)
        pred = pred_scores.float().squeeze(-1)
        if pred.numel() == 0:
            return self._empty_like_scalar(pred_scores)
        if pred.dim() == 1:
            pred = pred.unsqueeze(0)
        pred = pred.reshape(pred.shape[0], -1)
        if pred.shape[1] <= 1:
            sample_loss = F.relu(pred.mean(dim=1) - self.score_sparse_target) ** 2
            return sample_loss.mean()

        mean_score = pred.mean(dim=1)
        mean_loss = F.relu(mean_score - self.score_sparse_target) ** 2

        sample_loss = mean_loss

        if self.score_sparse_flat_weight > 0:
            k = int(round(self.score_sparse_top_ratio * pred.shape[1]))
            k = max(1, min(k, max(1, pred.shape[1] // 2)))
            high = torch.topk(pred, k=k, largest=True, dim=1).values.mean(dim=1)
            low = torch.topk(pred, k=k, largest=False, dim=1).values.mean(dim=1)
            contrast = high - low
            flat_gate = F.relu(self.score_sparse_flat_margin - contrast) / self.score_sparse_flat_margin
            sample_loss = sample_loss * (1.0 + self.score_sparse_flat_weight * flat_gate.detach())

        if self.score_sparse_rec_aware > 0 and rec_scores is not None:
            rec = rec_scores.float().detach()
            if rec.dim() == 1:
                rec = rec.unsqueeze(0)
            rec = rec.reshape(rec.shape[0], -1).mean(dim=1)
            if rec.numel() == sample_loss.numel() and rec.numel() > 1:
                rec_z = (rec - rec.mean()) / (rec.std(unbiased=False) + 1e-6)
                easy_weight = torch.sigmoid(-rec_z)
                min_w = self.score_sparse_min_weight
                rec_weight = min_w + (1.0 - min_w) * easy_weight
                mixed_weight = (1.0 - self.score_sparse_rec_aware) + self.score_sparse_rec_aware * rec_weight
                sample_loss = sample_loss * mixed_weight

        return sample_loss.mean()

    def _score_baseline_loss(self, pred_scores, rec_scores=None):
        if self.score_baseline_weight <= 0:
            return self._empty_like_scalar(pred_scores)
        pred = pred_scores.float().squeeze(-1)
        if pred.numel() == 0:
            return self._empty_like_scalar(pred_scores)
        if pred.dim() == 1:
            pred = pred.unsqueeze(0)
        pred = pred.reshape(pred.shape[0], -1)
        if pred.shape[0] == 0:
            return self._empty_like_scalar(pred_scores)

        if rec_scores is not None and self.score_baseline_rec_ratio > 0:
            rec = rec_scores.float().detach()
            if rec.dim() == 1:
                rec = rec.unsqueeze(0)
            rec = rec.reshape(rec.shape[0], -1).mean(dim=1)
            if rec.numel() == pred.shape[0] and rec.numel() > 1:
                keep = int(round(self.score_baseline_rec_ratio * int(rec.numel())))
                keep = max(1, min(keep, int(rec.numel())))
                easy_idx = torch.topk(rec, k=keep, largest=False, dim=0).indices
                pred = pred.index_select(0, easy_idx)

        if pred.numel() == 0:
            return self._empty_like_scalar(pred_scores)

        if pred.shape[1] <= 1:
            baseline = pred.mean(dim=1)
        else:
            try:
                baseline = torch.quantile(pred, self.score_baseline_quantile, dim=1)
            except Exception:
                k = int(round(self.score_baseline_quantile * (pred.shape[1] - 1))) + 1
                k = max(1, min(k, pred.shape[1]))
                baseline = torch.topk(pred, k=k, largest=False, dim=1).values[:, -1]

        baseline_loss = F.relu(baseline - self.score_baseline_margin) ** 2

        k_tail = int(round(self.score_baseline_top_ratio * pred.shape[1]))
        k_tail = max(1, min(k_tail, pred.shape[1]))
        tail = torch.topk(pred, k=k_tail, largest=True, dim=1).values.mean(dim=1)
        tail_loss = F.relu(tail - self.score_baseline_tail_margin) ** 2
        return (baseline_loss + 0.5 * tail_loss).mean()

    def _full_evidence_norm(self, values):
        if values is None:
            return None
        v = values.float()
        if v.numel() == 0:
            return v
        original_shape = v.shape
        if v.dim() == 0:
            return torch.sigmoid(v).reshape(original_shape)
        if v.dim() == 1:
            flat = v.reshape(1, -1)
        else:
            flat = v.reshape(v.shape[0], -1)

        eps = 1e-6
        if flat.shape[1] <= 1 and flat.shape[0] > 1:
            ref = flat.detach().reshape(-1)
            try:
                lo = torch.quantile(ref, 0.05)
                hi = torch.quantile(ref, 0.95)
            except Exception:
                lo = torch.min(ref)
                hi = torch.max(ref)
            norm = ((flat - lo) / (hi - lo + eps)).clamp(0.0, 1.0)
        else:
            ref = flat.detach()
            try:
                lo = torch.quantile(ref, 0.05, dim=1, keepdim=True)
                hi = torch.quantile(ref, 0.95, dim=1, keepdim=True)
            except Exception:
                lo = torch.min(ref, dim=1, keepdim=True).values
                hi = torch.max(ref, dim=1, keepdim=True).values
            norm = ((flat - lo) / (hi - lo + eps)).clamp(0.0, 1.0)
        return norm.reshape(original_shape)

    def _full_temporal_center_scores(self, inputs):
        if inputs is None:
            return None, None
        if inputs.dim() not in (3, 4):
            return None, None

        with torch.no_grad():
            x = inputs.float().detach()
            mix = self.temporal_rank_context_mix
            if x.dim() == 4:
                feat = F.normalize(x.mean(dim=1), p=2, dim=-1)  # [B,C,F]
                context = F.normalize(feat.mean(dim=1, keepdim=True), p=2, dim=-1)
                context_dist = (1.0 - torch.sum(feat * context, dim=-1)).clamp_min(0.0)
                if feat.shape[1] > 1:
                    adj_dist = (1.0 - torch.sum(feat[:, 1:, :] * feat[:, :-1, :], dim=-1)).clamp_min(0.0)
                    diff_score = torch.zeros_like(context_dist)
                    diff_count = torch.zeros_like(context_dist)
                    diff_score[:, 1:] = diff_score[:, 1:] + adj_dist
                    diff_score[:, :-1] = diff_score[:, :-1] + adj_dist
                    diff_count[:, 1:] = diff_count[:, 1:] + 1.0
                    diff_count[:, :-1] = diff_count[:, :-1] + 1.0
                    diff_score = diff_score / diff_count.clamp_min(1.0)
                else:
                    diff_score = torch.zeros_like(context_dist)
                temporal = mix * context_dist + (1.0 - mix) * diff_score
                return temporal.unsqueeze(-1), context_dist.unsqueeze(-1)

            feat = F.normalize(x, p=2, dim=-1)  # [B,10,F] during concat/global inference
            context = F.normalize(feat.mean(dim=1, keepdim=True), p=2, dim=-1)
            context_dist = (1.0 - torch.sum(feat * context, dim=-1)).clamp_min(0.0)
            center_score = context_dist.mean(dim=1, keepdim=True)
            if feat.shape[1] > 1:
                adj_dist = (1.0 - torch.sum(feat[:, 1:, :] * feat[:, :-1, :], dim=-1)).clamp_min(0.0)
                diff_score = adj_dist.mean(dim=1, keepdim=True)
            else:
                diff_score = torch.zeros_like(center_score)
            temporal = mix * center_score + (1.0 - mix) * diff_score
            return temporal, center_score

    def _full_evidence_target(self, inputs, rec_scores=None):
        components = []
        weight_sum = 0.0

        if self.full_evidence_rec_weight > 0:
            rec_comp = rec_scores
            if rec_comp is None:
                rec_comp = self._infer_recon_scores(inputs)
            elif rec_comp.dim() == 2 and inputs is not None and inputs.dim() == 4:
                rec_comp = rec_comp.unsqueeze(-1)
            rec_comp = self._full_evidence_norm(rec_comp)
            if rec_comp is not None:
                components.append(self.full_evidence_rec_weight * rec_comp)
                weight_sum += self.full_evidence_rec_weight

        temporal_comp, center_comp = self._full_temporal_center_scores(inputs)
        if self.full_evidence_temp_weight > 0 and temporal_comp is not None:
            temporal_comp = self._full_evidence_norm(temporal_comp)
            components.append(self.full_evidence_temp_weight * temporal_comp)
            weight_sum += self.full_evidence_temp_weight
        if self.full_evidence_center_weight > 0 and center_comp is not None:
            center_comp = self._full_evidence_norm(center_comp)
            components.append(self.full_evidence_center_weight * center_comp)
            weight_sum += self.full_evidence_center_weight

        if not components or weight_sum <= 0:
            return None
        evidence = torch.stack(components, dim=0).sum(dim=0) / float(weight_sum)
        evidence = evidence.clamp(0.0, 1.0)
        if abs(self.full_evidence_power - 1.0) > 1e-6:
            evidence = evidence.pow(self.full_evidence_power)
        return evidence.detach()

    def _full_evidence_loss(self, pred_scores, inputs, rec_scores=None):
        # Full-evidence is intentionally disabled for incremental-memory runs so
        # cvad4 full experiments cannot silently change the incremental method.
        if self.inc_enable or (not self.full_evidence_enable) or self.full_evidence_loss_weight <= 0:
            return self._empty_like_scalar(pred_scores)
        target = self._full_evidence_target(inputs, rec_scores=rec_scores)
        if target is None or target.shape != pred_scores.shape:
            return self._empty_like_scalar(pred_scores)
        return F.mse_loss(pred_scores.float(), target.float())

    def _infer_recon_scores(self, inputs):
        single_global_input = (inputs.dim() == 3)
        work_inputs = inputs
        if single_global_input and self.cross_clip > 1:
            work_inputs = inputs.unsqueeze(2).expand(-1, -1, self.cross_clip, -1).contiguous()

        if self.Vitblock_num == 0:
            recon = self.AE(work_inputs)
        else:
            if work_inputs.dim() == 3:
                work_inputs = work_inputs.unsqueeze(2)
            recon = self.decoder(work_inputs.reshape(-1, self.cross_clip, self.feature_size))
            recon = recon.reshape(-1, 10, self.cross_clip, self.feature_size)

        rec_err = torch.mean((recon - work_inputs) ** 2, dim=-1)  # [B,10,C]
        rec = torch.max(rec_err, dim=1)[0]  # [B,C]
        if single_global_input:
            rec = torch.mean(rec, dim=1, keepdim=True)  # [B,1], matches scorer(global_test) output
        if rec.shape[1] <= 1:
            rec_mean = torch.mean(rec, dim=0, keepdim=True)
            rec_std = torch.std(rec, dim=0, keepdim=True, unbiased=False) + 1e-6
        else:
            rec_mean = torch.mean(rec, dim=1, keepdim=True)
            rec_std = torch.std(rec, dim=1, keepdim=True, unbiased=False) + 1e-6
        rec_z = (rec - rec_mean) / rec_std
        rec_score = torch.sigmoid(rec_z)
        if single_global_input:
            return rec_score
        return rec_score.unsqueeze(-1)

    def _infer_memory_novelty_scores(self, inputs):
        if not self.inc_enable:
            return None
        count = self._inc_count()
        if count <= 0:
            return None
        bank = self.inc_memory_bank[:count]
        if self.infer_memory_frozen_only:
            frozen = self.inc_memory_frozen[:count]
            if torch.any(frozen):
                bank = bank[frozen]
        if bank.numel() == 0:
            return None

        query = inputs.mean(dim=1)  # [B,C,F]
        single_global_input = (query.dim() == 2)
        if single_global_input:
            query = query.unsqueeze(1)
        query = F.normalize(query, p=2, dim=-1)
        bank = F.normalize(bank, p=2, dim=-1)
        sim = torch.einsum('bcf,mf->bcm', query, bank)
        max_sim = torch.max(sim, dim=2)[0]
        novelty = (1.0 - max_sim) * 0.5
        novelty = novelty.clamp(0.0, 1.0)
        if single_global_input:
            return novelty
        return novelty.unsqueeze(-1)

    def _current_max_pseudo_pos(self):
        if self.max_pseudo_pos_start < 0 or self.max_pseudo_pos_end < 0:
            return self.max_pseudo_pos
        if self.max_pseudo_pos_anneal_iters <= 0:
            return min(max(self.max_pseudo_pos_end, 0.0), 1.0)
        t = min(1.0, float(self.iter) / float(self.max_pseudo_pos_anneal_iters))
        cur = self.max_pseudo_pos_start + t * (self.max_pseudo_pos_end - self.max_pseudo_pos_start)
        return min(max(cur, 0.0), 1.0)

    def _cap_pseudo_ratio(self, pseudo_labels, rank_scores):
        # Keep only top reconstruction-difficulty pseudo positives when ratio is too high.
        cap = self._current_max_pseudo_pos()
        if cap >= 1.0 and not self.enforce_min_pseudo_pos:
            return pseudo_labels
        cur_ratio = pseudo_labels.float().mean().item()
        flat_label = pseudo_labels.reshape(-1)
        flat_rank = rank_scores.reshape(-1)

        if cur_ratio > cap:
            candidate = torch.nonzero(flat_label > 0, as_tuple=False).squeeze(1)
            if candidate.numel() != 0:
                max_keep = int(cap * flat_label.numel())
                max_keep = max(1, min(max_keep, candidate.numel()))
                top_rel = torch.topk(flat_rank[candidate], k=max_keep, dim=0).indices
                keep = candidate[top_rel]
                new_flat = torch.zeros_like(flat_label)
                new_flat[keep] = flat_label[keep]
                flat_label = new_flat

        if self.enforce_min_pseudo_pos and self.min_pseudo_pos > 0:
            cur_keep = int((flat_label > 0).sum().item())
            min_keep = int(self.min_pseudo_pos * flat_label.numel())
            min_keep = max(1, min(min_keep, flat_label.numel()))
            if cur_keep < min_keep:
                add_top = torch.topk(flat_rank, k=min_keep, dim=0).indices
                boosted = torch.zeros_like(flat_label)
                boosted[add_top] = 1.0
                flat_label = torch.maximum(flat_label, boosted)

        return flat_label.view_as(pseudo_labels)

    def _decay_bank_scores(self, device):
        if len(self.lab_scores) == 0:
            return
        if self.lab_scores.device != device:
            self.lab_scores = self.lab_scores.to(device)
            self.lab_age = self.lab_age.to(device)
            self.lab = self.lab.to(device)
        self.lab_age = self.lab_age + 1
        self.lab_scores = self.bank_decay * self.lab_scores - self.bank_age_weight * self.lab_age.float()

    def _update_quality_bank(self, new_features, new_quality_scores):
        if self.mb_size <= 0:
            self.lab = torch.zeros(0, device=new_features.device if len(new_features) else 'cpu')
            self.lab_scores = torch.zeros(0, device=new_features.device if len(new_features) else 'cpu')
            self.lab_age = torch.zeros(0, device=new_features.device if len(new_features) else 'cpu')
            return

        device = new_features.device if len(new_features) else (self.lab.device if len(self.lab) else torch.device('cpu'))
        self._decay_bank_scores(device)

        if len(new_features) != 0:
            new_features = new_features.detach()
            new_quality_scores = new_quality_scores.detach()
            new_age = torch.zeros(len(new_features), device=device)
            if len(self.lab) == 0:
                self.lab = new_features
                self.lab_scores = new_quality_scores
                self.lab_age = new_age
            else:
                self.lab = torch.cat([self.lab, new_features], dim=0)
                self.lab_scores = torch.cat([self.lab_scores, new_quality_scores], dim=0)
                self.lab_age = torch.cat([self.lab_age, new_age], dim=0)

        if len(self.lab) > self.mb_size:
            top_idx = torch.topk(self.lab_scores, k=self.mb_size).indices
            self.lab = self.lab[top_idx]
            self.lab_scores = self.lab_scores[top_idx]
            self.lab_age = self.lab_age[top_idx]

    def _retrieve_dynamic_features(self, query_features):
        # query_features: [B, 10, C, 2048]
        if not self.use_qavclab or len(self.lab) == 0 or self.retrieval_topk <= 0 or self.retrieval_samples <= 0:
            return torch.zeros(0, device=query_features.device)

        if self.lab.device != query_features.device:
            self.lab = self.lab.to(query_features.device)
            self.lab_scores = self.lab_scores.to(query_features.device)
            self.lab_age = self.lab_age.to(query_features.device)

        query_embed = query_features.mean(dim=(1, 2))  # [B, D]
        bank_embed = self.lab.mean(dim=(1, 2))  # [M, D]
        sim = F.cosine_similarity(query_embed.unsqueeze(1), bank_embed.unsqueeze(0), dim=-1)  # [B, M]

        query_num = min(self.retrieval_samples, sim.shape[0])
        if query_num <= 0:
            return torch.zeros(0, device=query_features.device)
        # Randomly sample queries for retrieval to keep memory bounded.
        query_idx = torch.randperm(sim.shape[0], device=sim.device)[:query_num]
        sim = sim[query_idx]

        k = min(self.retrieval_topk, sim.shape[1])
        topv, topi = torch.topk(sim, k=k, dim=1)
        weights = self.retrieval_mlp(topv)  # [B, K]
        # Memory-friendly fusion: avoid materializing [B, K, 10, C, D] at once.
        fused_list = []
        for qi in range(query_num):
            bank_slice = self.lab.index_select(0, topi[qi])  # [K,10,C,D]
            fused_i = (bank_slice * weights[qi].view(k, 1, 1, 1)).sum(dim=0)  # [10,C,D]
            fused_list.append(fused_i)
        fused = torch.stack(fused_list, dim=0) if len(fused_list) else torch.zeros(0, device=query_features.device)
        return fused.detach()

    def slide_score(self,scores):
        scores_new = scores.clone()
        if self.cross_clip >= 2:
            scores_new[:, 1, :] = (scores[:, 0, :] + scores[:, 1, :]) / 2
            if self.cross_clip >= 3:
                scores_new[:, 2, :] = (scores[:, 0, :] + scores[:, 1, :] + scores[:, 2, :]) / 3
                if self.cross_clip >= 4:
                    scores_new[:, 3, :] = (scores[:, 0, :] + scores[:, 1, :] + scores[:, 2, :] + scores[:, 3,
                                                                                                 :]) / 4
        return scores_new

    def forward(self, inputs, lab=None, lab_type=None, is_training=False, is_test=False):
        del lab
        del lab_type
        if is_training:
            raise RuntimeError("This package only supports evaluation.")

        inputs = self._project_inputs(inputs)
        if is_test:
            if self.Vitblock_num == 0:
                return self.AE(inputs)
            recon = self.decoder(inputs.reshape(-1, self.cross_clip, self.feature_size))
            return recon.reshape(-1, 10, self.cross_clip, self.feature_size)

        scores = self.scorer(inputs, is_training=False)
        if scores.dim() >= 3 and scores.shape[1] == self.cross_clip:
            scores = self.slide_score(scores)
        if self.infer_with_recon:
            recon_scores = self._infer_recon_scores(inputs)
            if recon_scores is not None and recon_scores.shape == scores.shape:
                scores = self.infer_recon_alpha * scores + (1.0 - self.infer_recon_alpha) * recon_scores
        if self.full_evidence_infer and not self.inc_enable:
            evidence_scores = self._full_evidence_target(inputs, rec_scores=None)
            if evidence_scores is not None and evidence_scores.shape == scores.shape:
                scores = self.full_evidence_alpha * scores + (1.0 - self.full_evidence_alpha) * evidence_scores
        if self.infer_with_memory_novelty:
            novelty_scores = self._infer_memory_novelty_scores(inputs)
            if novelty_scores is not None and novelty_scores.shape == scores.shape:
                scores = self.infer_memory_alpha * scores + (1.0 - self.infer_memory_alpha) * novelty_scores
        if self.infer_with_teacher:
            with torch.no_grad():
                teacher_scores = self.scorer_teacher(inputs, is_training=False)
                if teacher_scores.dim() >= 3 and teacher_scores.shape[1] == self.cross_clip:
                    teacher_scores = self.slide_score(teacher_scores)
            scores = self.infer_teacher_alpha * scores + (1.0 - self.infer_teacher_alpha) * teacher_scores
        if self.infer_with_anchor and self.inc_enable and self.inc_anchor_active and self.scorer_anchor is not None:
            with torch.no_grad():
                anchor_scores = self.scorer_anchor(inputs, is_training=False)
                if anchor_scores.dim() >= 3 and anchor_scores.shape[1] == self.cross_clip:
                    anchor_scores = self.slide_score(anchor_scores)
            scores = self.infer_anchor_alpha * scores + (1.0 - self.infer_anchor_alpha) * anchor_scores
        return scores

class Scorer_C2FPL(nn.Module):  # multiplication then Addition
    def __init__(self, n_features):
        super(Scorer_C2FPL, self).__init__()
        self.fc1 = nn.Linear(n_features, 512)

        self.fc_att1 = nn.Sequential(nn.Linear(n_features, 512), nn.Softmax(dim=1))

        self.fc2 = nn.Linear(512, 32)

        self.fc_att2 = nn.Sequential(nn.Linear(512, 32), nn.Softmax(dim=1))

        self.fc3 = nn.Linear(32, 1)
        self.dropout = nn.Dropout(0.6)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.apply(weights_init)

    def forward(self, inputs, is_training=True):
        x = self.fc1(inputs)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.sigmoid(self.fc3(x))
        x = x.mean(dim=1)
        return x

class Scorer(torch.nn.Module):
    def __init__(self, n_feature, hidden1=1024, hidden2=128, dropout_p=0.7):
        super(Scorer, self).__init__()
        self.fc1 = nn.Linear(n_feature, int(hidden1))
        self.fc2 = nn.Linear(int(hidden1), int(hidden2))
        self.classifier = nn.Linear(int(hidden2), 1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(float(dropout_p))
        self.apply(weights_init)

    def forward(self, inputs, is_training=True):
        if is_training:
            x = self.relu(self.fc1(inputs))  # 2048
            x = self.dropout(x)
            x = self.relu(self.fc2(x))  # 2048
            x = self.dropout(x)
            x = self.classifier(x)
            score = self.sigmoid(x)
            return torch.mean(score,dim=1)
        else:
            x = self.relu(self.fc1(inputs))  # 2048
            x = self.relu(self.fc2(x))  # 2048
            x = self.classifier(x)
            score = self.sigmoid(x)
            return torch.mean(score,dim=1)

class Scorer_FRD(torch.nn.Module):
    def __init__(self, n_feature):
        super(Scorer_FRD, self).__init__()
        self.fc1 = nn.Linear(n_feature, 1024)
        self.fc2 = nn.Linear(1024, 128)
        self.classifier = nn.Linear(128, 1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.7)
        self.apply(weights_init)

    def forward(self, inputs, is_training=True):
        if is_training:
            x = self.relu(self.fc1(inputs))  # 2048
            x = self.dropout(x)
            x = self.relu(self.fc2(x))  # 2048
            x = self.dropout(x)
            x = self.classifier(x)
            score = self.sigmoid(x)
            return torch.mean(score,dim=1)
        else:
            x = self.relu(self.fc1(inputs))  # 2048
            x = self.relu(self.fc2(x))  # 2048
            x = self.classifier(x)
            score = self.sigmoid(x)
            #return torch.mean(score,dim=1)
            return score
