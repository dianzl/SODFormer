import torch
from torch import nn
import torch.nn.functional as F
from models.ops.modules import MSDeformAttn
from util.misc import _get_activation_fn, _get_clones
from .deformable_transformer import (DeformableTransformerEncoderLayer, DeformableTransformerEncoder,
                                     DeformableTransformerDecoderLayer, DeformableTransformerDecoder)
from torch.nn.init import normal_
from util.misc import NestedTensor, nested_tensor_from_tensor_list
from .backbone import build_backbone


class TemporalDETR(nn.Module):
    def __init__(self, d_model=256, nhead=8,
                 num_encoder_layers=6, num_decoder_layers=6,
                 dim_feedforward=1024, dropout=0.1,
                 n_frames=4, activation="relu", return_intermediate_dec=False,
                 dec_n_points=4, enc_n_points=4, fuse=True):
        super().__init__()
        
        self.d_model = d_model
        self.fuse = fuse
        encoder_layer = TemporalEncoderLayer(d_model=d_model,
                                             d_ffn=dim_feedforward,
                                             dropout=dropout,
                                             activation=activation,
                                             n_frames=n_frames,
                                             n_heads=nhead,
                                             n_points=enc_n_points,
                                             fuse=self.fuse)
        self.encoder = TemporalEncoder(encoder_layer, num_encoder_layers)

        decoder_layer = DeformableTransformerDecoderLayer(d_model, dim_feedforward,
                                                          dropout, activation,
                                                          1, nhead, dec_n_points)
        self.decoder = DeformableTransformerDecoder(decoder_layer, num_decoder_layers, return_intermediate_dec)

        self.reference_points = nn.Linear(d_model, 2)
        self._reset_parameters()
        
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MSDeformAttn):
                m._reset_parameters()
        nn.init.xavier_uniform_(self.reference_points.weight.data, gain=1.0)
        nn.init.constant_(self.reference_points.bias.data, 0.)

    def forward(self, src, reference_src, mask, reference_mask, src_shape, reference_shapes, reference_start_index,
                query_embeds, query_pos, valid_ratios):
        '''
        :param src_shape: (1, 2) [(H_t, W_t)]
        :param valid_ratios: (bs, 2)
        '''
        temporal_memory = self.encoder(src, reference_src, src_shape, reference_shapes, reference_start_index, reference_mask, query_pos, valid_ratios)
        bs, _, c = temporal_memory.shape
        query_embed, tgt = torch.split(query_embeds, c, dim=1)
        query_embed = query_embed.unsqueeze(0).expand(bs, -1, -1)
        tgt = tgt.unsqueeze(0).expand(bs, -1, -1)
        reference_points = self.reference_points(query_embed).sigmoid()
        self_start_index = src_shape.new_zeros(1, )
        hs, references = self.decoder(tgt, reference_points, temporal_memory, src_shape, self_start_index, valid_ratios[:, None], query_embed, mask)
        return hs, reference_points, references


class TemporalEncoderLayer(nn.Module):
    def __init__(self,
                 d_model=256, d_ffn=1024,
                 dropout=0.1, activation="relu",
                 n_frames=8, n_heads=8,
                 n_points=4, fuse=True):
        super().__init__()
        # self attention
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # temporal deformable attention
        n_frames = 2 * n_frames if fuse else n_frames
        self.cross_attn = MSDeformAttn(d_model, n_frames, n_heads, n_points)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

        # feed forward
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation)
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)
    
    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, tgt):
        tgt2 = self.linear2(self.dropout3(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout4(tgt2)
        tgt = self.norm3(tgt)
        return tgt
    
    def forward(self, tgt, query_pos, reference_points, src, src_spatial_shapes, level_start_index, src_padding_mask):
        # self attention
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q.transpose(0, 1), k.transpose(0, 1), tgt.transpose(0, 1))[0].transpose(0, 1)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # cross attention
        tgt2 = self.cross_attn(self.with_pos_embed(tgt, query_pos),
                               reference_points,
                               src, src_spatial_shapes, level_start_index, src_padding_mask)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        # ffn
        tgt = self.forward_ffn(tgt)

        return tgt

class TemporalEncoder(nn.Module):
    def __init__(self, encoder_layer, num_encoder):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_encoder)
        self.num_encoder = num_encoder

    @staticmethod
    def get_reference_points(spatial_shape, valid_ratios, ref_valid_ratios, device):
        """
        :param spatial_shape: (1, 2) [(H_t, W_t)]
        :param valid_ratios: (bs, 2)
        :param ref_valid_ratios: (bs, nf, 2)

        :return (bs, H_t*W_t, nf, 2)
        """
        H_, W_ = spatial_shape[0]
        ref_y, ref_x = torch.meshgrid(torch.linspace(0.5, H_ - 0.5, H_, dtype=torch.float32, device=device),
                                      torch.linspace(0.5, W_ - 0.5, W_, dtype=torch.float32, device=device))
        ref_y = ref_y.reshape(-1)[None] / (valid_ratios[:, None, 1] * H_)
        ref_x = ref_x.reshape(-1)[None] / (valid_ratios[:, None, 0] * W_)
        ref = torch.stack((ref_x, ref_y), -1)
        reference_points = ref[:, :, None] * ref_valid_ratios[:, None]
        return reference_points

    def get_valid_ratio(self, mask):
        '''
        :param mask (bs, nf, H, W)

        :return valid_ratio (bs, nf, 2)
        '''
        _, _, H, W = mask.shape
        valid_H = torch.sum(~mask[:, :, :, 0], 2)
        valid_W = torch.sum(~mask[:, :, 0, :], 2)
        valid_ratio_h = valid_H.float() / H
        valid_ratio_w = valid_W.float() / W
        valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)
        return valid_ratio
    
    def forward(self, tgt, src, self_shape, spatial_shapes, level_start_index,
                padding_mask, query_pos, valid_ratios):
        """
        tgt: current frame encoder output: (bs, H_t*W_t, hidden_dim)
        src: reference frames encoder outputs: (bs, nf*H_p*W_p, hidden_dim)
        self_shape: (1, 2) [(H_t, W_t)]
        spatial_shapes: (l+1, 2) [(H_p, W_p) * nf]
        level_start_index: (l, ) [0, H_p*W_p, 2*H_p*W_p, ..., (l-1) * H_p*W_p]
        padding_mask: (bs, nf*H_p*W_p)
        self_mask: (bs, H_t*W_t)
        query_pos: position embedding (bs, H_t*W_t, hidden_dim)
        valid_ratios: (bs, 2)
        """
        output = tgt
        H_p, W_p = spatial_shapes[0]
        bs, _ = padding_mask.shape
        ref_valid_ratios = self.get_valid_ratio(padding_mask.view(bs, -1, H_p, W_p))
        reference_points = self.get_reference_points(self_shape, valid_ratios, ref_valid_ratios, device=src.device) # (bs, H_t*W_t, nf, 2)

        for _, layer in enumerate(self.layers):
            output = layer(output, query_pos, reference_points, src,
                           spatial_shapes, level_start_index, padding_mask)
        return output


class SpatialEncoderTransformer(nn.Module):
    def __init__(self, backbone, d_model=256, nhead=8,
                 num_encoder_layers=6, dim_feedforward=1024, dropout=0.1,
                 num_feature_levels=4, activation="relu", enc_n_points=4):
        super().__init__()
        self.backbone = backbone

        self.num_feature_levels = num_feature_levels
        hidden_dim = d_model
        if num_feature_levels > 1:
            num_backbone_outs = len(backbone.strides)
            input_proj_list = []
            for _ in range(num_backbone_outs):
                in_channels = backbone.num_channels[_]
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, hidden_dim),
                ))
            for _ in range(num_feature_levels - num_backbone_outs):
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, hidden_dim, kernel_size=3, stride=2, padding=1),
                    nn.GroupNorm(32, hidden_dim),
                ))
                in_channels = hidden_dim
            self.input_proj = nn.ModuleList(input_proj_list)
        else:
            self.input_proj = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(backbone.num_channels[0], hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, hidden_dim),
                )])
        for proj in self.input_proj:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0)

        self.d_model = d_model
        self.nhead = nhead
        encoder_layer = DeformableTransformerEncoderLayer(d_model, dim_feedforward,
                                                          dropout, activation,
                                                          num_feature_levels, nhead, enc_n_points)
        self.encoder = DeformableTransformerEncoder(encoder_layer, num_encoder_layers)

        self.level_embed = nn.Parameter(torch.Tensor(num_feature_levels, d_model))
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MSDeformAttn):
                m._reset_parameters()
        normal_(self.level_embed)

    def get_valid_ratio(self, mask):
        _, H, W = mask.shape
        valid_H = torch.sum(~mask[:, :, 0], 1)
        valid_W = torch.sum(~mask[:, 0, :], 1)
        valid_ratio_h = valid_H.float() / H
        valid_ratio_w = valid_W.float() / W
        valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)
        return valid_ratio

    def forward(self, samples: NestedTensor):
        if not isinstance(samples, NestedTensor):
            samples = nested_tensor_from_tensor_list(samples)
        features, pos = self.backbone(samples)

        srcs = []
        masks = []
        for l, feat in enumerate(features):
            src, mask = feat.decompose()
            srcs.append(self.input_proj[l](src))
            masks.append(mask)
            assert mask is not None
        if self.num_feature_levels > len(srcs):
            _len_srcs = len(srcs)
            for l in range(_len_srcs, self.num_feature_levels):
                if l == _len_srcs:
                    src = self.input_proj[l](features[-1].tensors)
                else:
                    src = self.input_proj[l](srcs[-1])
                m = samples.mask
                mask = F.interpolate(m[None].float(), size=src.shape[-2:]).to(torch.bool)[0]
                pos_l = self.backbone[1](NestedTensor(src, mask)).to(src.dtype)
                srcs.append(src)
                masks.append(mask)
                pos.append(pos_l)

        # prepare input for encoder
        src_flatten = []
        mask_flatten = []
        lvl_pos_embed_flatten = []
        spatial_shapes = []
        for lvl, (src, mask, pos_embed) in enumerate(zip(srcs, masks, pos)):
            _, _, h, w = src.shape
            spatial_shape = (h, w)
            spatial_shapes.append(spatial_shape)
            src = src.flatten(2).transpose(1, 2)
            mask = mask.flatten(1)
            pos_embed = pos_embed.flatten(2).transpose(1, 2)
            lvl_pos_embed = pos_embed + self.level_embed[lvl].view(1, 1, -1)
            lvl_pos_embed_flatten.append(lvl_pos_embed)
            src_flatten.append(src)
            mask_flatten.append(mask)
        src_flatten = torch.cat(src_flatten, 1)
        mask_flatten = torch.cat(mask_flatten, 1)
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=src_flatten.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros((1, )), spatial_shapes.prod(1).cumsum(0)[:-1]))
        valid_ratios = torch.stack([self.get_valid_ratio(m) for m in masks], 1)

        # encoder
        memory = self.encoder(src_flatten, spatial_shapes, level_start_index, valid_ratios, lvl_pos_embed_flatten, mask_flatten)

        e_h, e_w = spatial_shapes[-1]
        single_feature_memory = memory[:, -e_h*e_w:]
        single_feature_mask = mask_flatten[:, -e_h*e_w:]
        return single_feature_memory, single_feature_mask, spatial_shapes[-1:], valid_ratios[:, -1]


def build_temporal_transformer(args):
    fuse = False if args.no_event or args.no_frame else True
    return TemporalDETR(d_model=args.hidden_dim,
                        nhead=args.nheads,
                        num_encoder_layers=args.tem_enc_layers, ## for comparison
                        num_decoder_layers=args.tem_dec_layers, ## for comparison
                        dim_feedforward=args.dim_feedforward,
                        dropout=args.dropout,
                        n_frames=args.n_frames,
                        activation="relu",
                        return_intermediate_dec=True,
                        dec_n_points=args.dec_n_points,
                        enc_n_points=args.enc_n_points,
                        fuse=fuse)


def build_spatial_encoder(args):
    backbone = build_backbone(args)
    return SpatialEncoderTransformer(backbone,
                                     d_model=args.hidden_dim,
                                     nhead=args.nheads,
                                     num_encoder_layers=args.enc_layers,
                                     dim_feedforward=args.dim_feedforward,
                                     dropout=args.dropout,
                                     num_feature_levels=args.num_feature_levels,
                                     activation="relu",
                                     enc_n_points=args.enc_n_points)
