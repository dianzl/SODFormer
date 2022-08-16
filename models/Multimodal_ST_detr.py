"""
Our complete SODFormer model.
"""
import torch
from torch import nn
from util.misc import MLP, inverse_sigmoid, map_keys_t, match_keywords_t
import math
import copy
from .fusion import build_temporal_encoder, build_fusion_transformer


class Multimodal_detr(nn.Module):
    def __init__(self, train_temporal,
                 aps_tem_tran,
                 dvs_tem_tran,
                 fusion_transformer,
                 num_classes=4,
                 num_queries=300,
                 aux_loss=True):
        """
        spatial_transformer returns: memory encoding (bs, H_tW_t, hidden_dim);
        """
        super().__init__()
        self.aps_tem_tran = aps_tem_tran
        self.dvs_tem_tran = dvs_tem_tran
        if not train_temporal:
            for p in self.parameters():
                p.requires_grad = False

        self.fusion_transformer = fusion_transformer
        self.aux_loss = aux_loss

        self.d_model = self.fusion_transformer.d_model
        
        # concatenation
        # self.class_embed = nn.Linear(self.d_model*2, num_classes)
        # self.bbox_embed = MLP(self.d_model*2, self.d_model*2, 4, 3)
        # self.query_embed = nn.Embedding(num_queries, self.d_model*4)

        self.class_embed = nn.Linear(self.d_model, num_classes)
        self.bbox_embed = MLP(self.d_model, self.d_model, 4, 3)
        self.query_embed = nn.Embedding(num_queries, self.d_model*2)

        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        self.class_embed.bias.data = torch.ones(num_classes) * bias_value
        nn.init.constant_(self.bbox_embed.layers[-1].weight.data, 0)
        nn.init.constant_(self.bbox_embed.layers[-1].bias.data, 0)

        num_pred = self.fusion_transformer.decoder.num_layers
        nn.init.constant_(self.bbox_embed.layers[-1].bias.data[2:], -2.0)
        self.class_embed = nn.ModuleList([self.class_embed for _ in range(num_pred)])
        self.bbox_embed = nn.ModuleList([self.bbox_embed for _ in range(num_pred)])

    def forward(self, aps, dvs, video_id):
        """ The forward expects a NestedTensor, which consists of:
               - aps.tensor: batched images, of shape [bs x 3 x H_t x W_t]
               - aps.mask: a binary mask of shape [bs x H_t x W_t], containing 1 on padded pixels
               - dvs.tensor: batched images, of shape [bs x 3 x H_t x W_t]
               - dvs.mask: a binary mask of shape [bs x H_t x W_t], containing 1 on padded pixels
        """
        aps_src, mask_flatten, spatial_shapes, valid_ratios = self.aps_tem_tran(aps, video_id)
        dvs_src, _, _, _ = self.dvs_tem_tran(dvs, video_id)
        query_embeds = self.query_embed.weight
        
        hs, init_reference, inter_reference = self.fusion_transformer(aps_src, dvs_src, mask_flatten, spatial_shapes, query_embeds, valid_ratios)

        outputs_classes = []
        outputs_coords = []
        for lvl in range(hs.shape[0]):
            if lvl == 0:
                reference = init_reference
            else:
                reference = inter_reference[lvl - 1]
            reference = inverse_sigmoid(reference)
            outputs_class = self.class_embed[lvl](hs[lvl])
            tmp = self.bbox_embed[lvl](hs[lvl])
            if reference.shape[-1] == 4:
                tmp += reference
            else:
                assert reference.shape[-1] == 2
                tmp[..., :2] += reference
            outputs_coord = tmp.sigmoid()
            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)
        outputs_class = torch.stack(outputs_classes)
        outputs_coord = torch.stack(outputs_coords)

        out = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1]}
        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord)
        return out

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{'pred_logits': a, 'pred_boxes': b}
                for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]


def build_Multimodal_ST(args):
    temporal_trans_f = build_temporal_encoder(args)
    temporal_trans_e = copy.deepcopy(temporal_trans_f)
    fusion_trans = build_fusion_transformer(args)
    train_temporal = True

    if args.temporal_aps_model and args.temporal_dvs_model:
        print('Loading pretrained temporal model encoders')
        aps_checkpoint = torch.load(args.temporal_aps_model, map_location='cpu')
        dvs_checkpoint = torch.load(args.temporal_dvs_model, map_location='cpu')
        aps_model_dict = temporal_trans_f.state_dict()
        dvs_model_dict = temporal_trans_e.state_dict()
        aps_state_dict = {map_keys_t(k): v for k, v in aps_checkpoint['model'].items() if match_keywords_t(k)}
        dvs_state_dict = {map_keys_t(k): v for k, v in dvs_checkpoint['model'].items() if match_keywords_t(k)}
        assert len(aps_state_dict) == len(dvs_state_dict)
        print('Loaded {} parameters from model {} for APS temporal encoder and {} parameters from model {} for DVS temporal encoder'.format(len(aps_state_dict), args.spatial_aps_model, len(dvs_state_dict), args.spatial_dvs_model))
        aps_model_dict.update(aps_state_dict)
        dvs_model_dict.update(dvs_state_dict)
        f_missing_keys, f_unexpected_keys = temporal_trans_f.load_state_dict(aps_model_dict, strict=False)
        e_missing_keys, e_unexpected_keys = temporal_trans_e.load_state_dict(dvs_model_dict, strict=False)
        f_unexpected_keys = [k for k in f_unexpected_keys if not (k.endswith('total_params') or k.endswith('total_ops'))]
        e_unexpected_keys = [k for k in e_unexpected_keys if not (k.endswith('total_params') or k.endswith('total_ops'))]
        if len(f_missing_keys) > 0:
            print('APS Model Missing Keys: {}'.format(f_missing_keys))
        if len(f_unexpected_keys) > 0:
            print('APS Model Unexpected Keys: {}'.format(f_unexpected_keys))
        if len(e_missing_keys) > 0:
            print('DVS Model Missing Keys: {}'.format(e_missing_keys))
        if len(e_unexpected_keys) > 0:
            print('DVS Model Unexpected Keys: {}'.format(e_unexpected_keys))
        train_temporal = False

    Multimodal_ST_model = Multimodal_detr(train_temporal=train_temporal,
                                          aps_tem_tran=temporal_trans_f,
                                          dvs_tem_tran=temporal_trans_e,
                                          fusion_transformer=fusion_trans,
                                          num_classes=args.num_classes,
                                          num_queries=args.num_queries,
                                          aux_loss=True)
    return Multimodal_ST_model
