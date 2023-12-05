from __future__ import print_function
from cmath import sqrt
import torch
import torch.nn as nn
import torch.nn.functional as F
from .deformable_transformer import DeformableTransformerDecoderLayer, DeformableTransformerDecoder
from models.ops.modules import MSDeformAttn
from collections import deque
from util.misc import _max_by_axis
from .temporal_detr import build_spatial_encoder, TemporalEncoderLayer, TemporalEncoder


class Fusion_detr(nn.Module):
    def __init__(self, d_model=256, nhead=8,
                 num_decoder_layers=6,
                 dim_feedforward=1024, dropout=0.1,
                 activation="relu", return_intermediate_dec=False,
                 dec_n_points=4) -> None:
        super().__init__()
        # Fusion module
        self.d_model = d_model
        self.aps_linear1 = nn.Linear(d_model, d_model)
        self.aps_linear2 = nn.Linear(d_model, d_model)
        self.dvs_linear1 = nn.Linear(d_model, d_model)
        self.dvs_linear2 = nn.Linear(d_model, d_model)
        
        # Decoder module
        decoder_layer = DeformableTransformerDecoderLayer(d_model, dim_feedforward,
                                                          dropout, activation,
                                                          1, nhead, dec_n_points)
        self.decoder = DeformableTransformerDecoder(decoder_layer, num_decoder_layers, return_intermediate_dec)
        self.reference_points = nn.Linear(d_model, 2)

        # # concatenation
        # decoder_layer = DeformableTransformerDecoderLayer(d_model*2, dim_feedforward,
        #                                                   dropout, activation,
        #                                                   1, nhead, dec_n_points)
        # self.decoder = DeformableTransformerDecoder(decoder_layer, num_decoder_layers, return_intermediate_dec)

        # self.reference_points = nn.Linear(d_model*2, 2)

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

    def forward(self, aps, dvs, mask, spatial_shapes, query_embeds, valid_ratios):
        '''
        param aps: (bs, H_tW_t, hd)
        param dvs: (bs, H_tW_t, hd)
        param mask: (bs, H_tW_t)
        param spatial_shapes:
        param query_embeds: 
        param valid_ratios:
        '''
        aps_query = self.aps_linear1(aps) # (bs, H_tW_t, hd)
        aps_key = self.aps_linear2(aps) # (bs, H_tW_t, hd)
        aps_weight = torch.div(torch.matmul(aps_query, torch.transpose(aps_key, dim0=1, dim1=2)), sqrt(self.d_model)).float() # (bs, H_tW_t, H_tW_t)
        aps_value = torch.sum(aps_weight, dim=-1) # (bs, H_tW_t)

        dvs_query = self.dvs_linear1(dvs)
        dvs_key = self.dvs_linear2(dvs)
        dvs_weight = torch.div(torch.matmul(dvs_query, torch.transpose(dvs_key, dim0=1, dim1=2)), sqrt(self.d_model)).float() # (bs, H_tW_t, H_tW_t)
        dvs_value = torch.sum(dvs_weight, dim=-1) # (bs, H_tW_t)

        fusion_weight = torch.stack([aps_value, dvs_value], dim=-1) # (bs, H_tW_t, 2)
        fusion_value = F.softmax(fusion_weight, -1) # (bs, H_tW_t, 2)
        value_list = torch.split(fusion_value, 1, -1) # list, element of shape (bs, H_tW_t, 1), len = 2

        fused_src = value_list[0] * aps + value_list[1] * dvs

        # averaging
        # fused_src = 0.5 * aps + 0.5 * dvs

        # concatenation
        # fused_src = torch.cat([aps, dvs], dim=-1)

        bs, _, c = fused_src.shape
        query_embed, tgt = torch.split(query_embeds, c, dim=1)
        query_embed = query_embed.unsqueeze(0).expand(bs, -1, -1)
        tgt = tgt.unsqueeze(0).expand(bs, -1, -1)
        reference_points = self.reference_points(query_embed).sigmoid()
        self_start_index = spatial_shapes.new_zeros(1, )
        hs, references = self.decoder(tgt, reference_points, fused_src, spatial_shapes, self_start_index, valid_ratios[:, None], query_embed, mask)
        return hs, reference_points, references


class TemporalEncoderTransformer(nn.Module):
    def __init__(self, num_ref_frames,
                 spatial_transformer,
                 temporal_encoder) -> None:
        super().__init__()
        self.num_ref_frames = num_ref_frames
        self.spatial_transformer = spatial_transformer
        self.temporal_encoder = temporal_encoder
        self.video_id = 0

        self.src_all = deque(maxlen=num_ref_frames)
        self.mask_all = deque(maxlen=num_ref_frames)
        self.spatial_shapes_all = deque(maxlen=num_ref_frames)
        
        self.d_model = self.spatial_transformer.d_model
    
    def reset_history(self, video_id):
        self.src_all.clear()
        self.mask_all.clear()
        self.spatial_shapes_all.clear()
        self.video_id = video_id
        return

    def forward(self, samples, video_id):
        if not video_id == self.video_id:
            self.reset_history(video_id)
        bs, _, _, _ = samples.tensors.shape
        """
        memory: (bs, H_tW_t, hidden_dim)
        mask: (bs, H_tW_t)
        """
        memory, mask, spatial_shapes, valid_ratios = self.spatial_transformer(samples)
        H_t, W_t = spatial_shapes[0]
        assert H_t*W_t == memory.shape[1]

        memories = list(torch.split(memory, 1, dim=0))
        masks = list(torch.split(mask, 1, dim=0))

        H_l, W_l = _max_by_axis(list(self.spatial_shapes_all)) if len(self.spatial_shapes_all) else (H_t, W_t)
        H_p = max(H_t, H_l)
        W_p = max(W_t, W_l)
        for i in range(len(self.src_all)):
            self.src_all[i], self.mask_all[i] = self.pad_to_same_size(self.src_all[i], self.mask_all[i], self.spatial_shapes_all[i], (H_p, W_p))
            self.spatial_shapes_all[i] = [H_p, W_p]

        padded_ref_memories = []
        padded_ref_masks = []

        for i in range(bs):
            pad_num = max(self.num_ref_frames - len(self.src_all) - i, 0)
            prev_index = max(len(self.src_all) - self.num_ref_frames + i, 0)
            start_index = max(i - self.num_ref_frames, 0)

            pad_memories = torch.zeros([1, pad_num*H_p*W_p, self.d_model], device=memory.device)
            ref_memories1 = list(self.src_all)[prev_index:]
            ref_memories1 = torch.cat(ref_memories1, dim=1) if len(ref_memories1) else torch.tensor([], device=memory.device).view(1, 0, self.d_model)
            ref_memories2 = memories[start_index:i]
            ref_memories2 = torch.cat(ref_memories2, dim=1) if len(ref_memories2) else torch.tensor([], device=memory.device).view(1, 0, self.d_model)
            padded_ref_memory = torch.cat([pad_memories, ref_memories1, ref_memories2], dim=1) # (1, nf*H_p*W_p, hd)
            padded_ref_memories.append(padded_ref_memory)

            pad_masks = torch.zeros([1, pad_num*H_p*W_p], dtype=bool, device=memory.device)
            ref_masks1 = list(self.mask_all)[prev_index:]
            ref_masks1 = torch.cat(ref_masks1, dim=1) if len(ref_masks1) else torch.tensor([], dtype=torch.bool, device=memory.device).view(1, 0)
            ref_masks2 = masks[start_index:i]
            ref_masks2 = torch.cat(ref_masks2, dim=1) if len(ref_masks2) else torch.tensor([], dtype=torch.bool, device=memory.device).view(1, 0)
            padded_ref_mask = torch.cat([pad_masks, ref_masks1, ref_masks2], dim=1) # (1, nf*H_p*W_p)
            padded_ref_masks.append(padded_ref_mask)

            memories[i], masks[i] = self.pad_to_same_size(memories[i], masks[i], (H_t, W_t), (H_p, W_p))

        padded_ref_memories = torch.cat(padded_ref_memories, dim=0) # (bs, nf*H_p*W_p, hd)
        padded_ref_masks = torch.cat(padded_ref_masks, dim=0) # (bs, nf*H_p*W_p)
        reference_shapes = [(H_p, W_p)] * (self.num_ref_frames)
        padded_ref_shapes = torch.as_tensor(reference_shapes, dtype=torch.long, device=memory.device)
        lvl_start_index = torch.cat((padded_ref_shapes.new_zeros((1, )), padded_ref_shapes.prod(1).cumsum(0)[:-1]))
        
        temporal_memory = self.temporal_encoder(memory, padded_ref_memories, spatial_shapes, padded_ref_shapes, lvl_start_index, padded_ref_masks, None, valid_ratios)

        start_idx = -min(self.num_ref_frames, memory.shape[0])
        for i in range(self.num_ref_frames):
            if i >= memory.shape[0]:
                break
            mem, mas = memory.detach(), mask.detach()
            self.src_all.append(mem[start_idx+i][None])
            self.mask_all.append(mas[start_idx+i][None])
            self.spatial_shapes_all.append([H_t, W_t])
            
        return temporal_memory, mask, spatial_shapes, valid_ratios
    
    @staticmethod
    def pad_to_same_size(tensor, mask, size, pad_size):
        '''
        :param tensor: (1, H_tW_t, hidden_dim)
        :param mask: (1, H_tW_t)
        :param size: (H_t, W_t)
        :param pad_size: (H_p, W_p)

        :return padded_tensor: (1, H_pW_p, hidden_dim)
        :return padded_mask: (1, H_pW_p)
        '''
        H_t, W_t = size
        H_p, W_p = pad_size
        _, l, c = tensor.shape
        assert H_t <= H_p and W_t <= W_p and H_t*W_t == l
        device = tensor.device
        padded_tensor = torch.zeros((1, H_p, W_p, c), dtype=tensor.dtype, device=device)
        padded_mask = torch.ones((1, H_p, W_p), dtype=bool, device=device)
        padded_tensor[:, :H_t, :W_t, :].copy_(tensor.view(1, H_t, W_t, c))
        padded_mask[:, :H_t, :W_t].copy_(mask.view(1, H_t, W_t))
        padded_tensor = padded_tensor.view(1, H_p*W_p, c)
        padded_mask = padded_mask.view(1, H_p*W_p)
        return padded_tensor, padded_mask


def build_temporal_encoder(args):
    spatial_encoder = build_spatial_encoder(args)
    encoder_layer = TemporalEncoderLayer(d_model=args.hidden_dim,
                                         d_ffn=args.dim_feedforward,
                                         dropout=args.dropout,
                                         activation="relu",
                                         n_frames=args.n_frames,
                                         n_heads=args.nheads,
                                         n_points=args.enc_n_points,
                                         fuse=False)

    encoder = TemporalEncoder(encoder_layer, args.enc_layers)
    return TemporalEncoderTransformer(num_ref_frames=args.n_frames,
                                      spatial_transformer=spatial_encoder,
                                      temporal_encoder=encoder)


def build_fusion_transformer(args):
    return Fusion_detr(d_model=args.hidden_dim,
                       nhead=args.nheads,
                       num_decoder_layers=args.dec_layers,
                       dim_feedforward=args.dim_feedforward,
                       dropout=args.dropout,
                       activation="relu",
                       return_intermediate_dec=True,
                       dec_n_points=args.dec_n_points)
