"""
Our Spatio-temporal baseline model.
"""
import torch
from torch import nn
from util.misc import MLP, inverse_sigmoid, map_keys_s, match_keywords_s, _max_by_axis
import math
from .temporal_detr import build_temporal_transformer, build_spatial_encoder
from collections import deque


class ST_detr(nn.Module):
    def __init__(self, num_ref_frames,
                 spatial_transformer,
                 train_spatial,
                 temporal_transformer,
                 num_classes=4,
                 num_queries=300,
                 aux_loss=True):
        super().__init__()
        self.num_ref_frames = num_ref_frames
        self.spatial_transformer = spatial_transformer
        if not train_spatial:
            for p in self.parameters():
                p.requires_grad = False
        self.temporal_transformer = temporal_transformer
        self.video_id = 0
        self.aux_loss = aux_loss

        self.src_all = deque(maxlen=num_ref_frames)
        self.mask_all = deque(maxlen=num_ref_frames)
        self.spatial_shapes_all = deque(maxlen=num_ref_frames)
        
        self.d_model = self.temporal_transformer.d_model
        self.num_queries = num_queries
        self.num_classes = num_classes
        
        self.class_embed = nn.Linear(self.d_model, num_classes)
        self.bbox_embed = MLP(self.d_model, self.d_model, 4, 3)
        self.query_embed = nn.Embedding(self.num_queries, self.d_model*2)

        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        self.class_embed.bias.data = torch.ones(num_classes) * bias_value
        nn.init.constant_(self.bbox_embed.layers[-1].weight.data, 0)
        nn.init.constant_(self.bbox_embed.layers[-1].bias.data, 0)
        num_pred = self.temporal_transformer.decoder.num_layers
        nn.init.constant_(self.bbox_embed.layers[-1].bias.data[2:], -2.0)
        self.class_embed = nn.ModuleList([self.class_embed for _ in range(num_pred)])
        self.bbox_embed = nn.ModuleList([self.bbox_embed for _ in range(num_pred)])

    def reset_history(self, video_id):
        self.src_all.clear()
        self.mask_all.clear()
        self.spatial_shapes_all.clear()
        self.video_id = video_id
        return
        
    def forward(self, samples, video_id):
        """ The forward expects a NestedTensor, which consists of:
               - samples.tensors: batched images, of shape [1 x 3 x H_t x W_t]
               - samples.mask: a binary mask of shape [1 x H_t x W_t], containing 1 on padded pixels
        """
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
        query_embeds = self.query_embed.weight

        hs, init_reference, inter_reference = self.temporal_transformer(memory,
                                                                        padded_ref_memories,
                                                                        mask,
                                                                        padded_ref_masks,
                                                                        spatial_shapes,
                                                                        padded_ref_shapes,
                                                                        lvl_start_index,
                                                                        query_embeds,
                                                                        None,
                                                                        valid_ratios)
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

        start_idx = -min(self.num_ref_frames, memory.shape[0])
        for i in range(self.num_ref_frames):
            if i >= memory.shape[0]:
                break
            self.src_all.append(memory[start_idx+i][None])
            self.mask_all.append(mask[start_idx+i][None])
            self.spatial_shapes_all.append([H_t, W_t])
        return out

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

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{'pred_logits': a, 'pred_boxes': b}
                for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]


def build_ST(args, input_type):
    temporal_transformer = build_temporal_transformer(args)
    spatial_transformer = build_spatial_encoder(args)

    train_spatial = True
    if input_type == 'frame_only':
        if args.spatial_aps_model:
            print('Loading pretrained spatial-frame model encoder')
            checkpoint = torch.load(args.spatial_aps_model, map_location='cpu')
            model_dict = spatial_transformer.state_dict()
            state_dict = {map_keys_s(k): v for k, v in checkpoint['model'].items() if match_keywords_s(k)}
            print('Loaded {} parameters from {}'.format(len(state_dict.keys()), args.spatial_aps_model))
            model_dict.update(state_dict)
            missing_keys, unexpected_keys = spatial_transformer.load_state_dict(model_dict, strict=False)
            unexpected_keys = [k for k in unexpected_keys if not (k.endswith('total_params') or k.endswith('total_ops'))]
            if len(missing_keys) > 0:
                print('Missing Keys: {}'.format(missing_keys))
            if len(unexpected_keys) > 0:
                print('Unexpected Keys: {}'.format(unexpected_keys))
            train_spatial = False
            
    elif input_type == 'event_only':
        if args.spatial_dvs_model:
            print('Loading pretrained spatial-event model encoder')
            checkpoint = torch.load(args.spatial_dvs_model, map_location='cpu')
            model_dict = spatial_transformer.state_dict()
            state_dict = {map_keys_s(k): v for k, v in checkpoint['model'].items() if match_keywords_s(k)}
            print('Loaded {} parameters from {}'.format(len(state_dict.keys()), args.spatial_dvs_model))
            model_dict.update(state_dict)
            missing_keys, unexpected_keys = spatial_transformer.load_state_dict(model_dict, strict=False)
            unexpected_keys = [k for k in unexpected_keys if not (k.endswith('total_params') or k.endswith('total_ops'))]
            if len(missing_keys) > 0:
                print('Missing Keys: {}'.format(missing_keys))
            if len(unexpected_keys) > 0:
                print('Unexpected Keys: {}'.format(unexpected_keys))
            train_spatial = False
            
        
    temporal_model = ST_detr(num_ref_frames=args.n_frames,
                             spatial_transformer=spatial_transformer,
                             train_spatial=train_spatial,
                             temporal_transformer=temporal_transformer,
                             num_classes=args.num_classes,
                             num_queries=args.num_queries)

    return temporal_model
