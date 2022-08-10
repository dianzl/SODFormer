"""
Train and eval functions used in prediction.py
"""
import torch
import util.misc as utils
from torch.utils.data import ConcatDataset
from datasets import build_dataset_asyn
import bisect


class ConcatDatasetCustom(ConcatDataset):
    def __getitem__(self, idx):
        if idx < 0:
            if -idx > len(self):
                raise ValueError("absolute value of index should not exceed dataset length")
            idx = len(self) + idx
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        return self.datasets[dataset_idx][sample_idx], (dataset_idx, idx)


@torch.no_grad()
def evaluate(model, criterion, postprocessors, data_loader, device, model_type: str = 'temporal',
             input_type: str = 'multi-modal'):
    model.eval()
    criterion.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Visulizing:'

    ress = []

    for inputs, indexes in metric_logger.log_every(data_loader, 10, header):
        samples, events, targets, corr_ids = inputs
        seq_indexes, img_ids = indexes
        samples = samples.to(device) if input_type == 'frame_only' or input_type == 'multi-modal' else samples
        events = events.to(device) if input_type == 'event_only' or input_type == 'multi-modal' else events
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        unimodal_input = samples if input_type == 'frame_only' else events
        if model_type == 'spatial':
            outputs, _, _, _, _ = model(unimodal_input)
        elif model_type == 'temporal':
            outputs = model(unimodal_input, seq_indexes[0])
        else:
            outputs = model(samples, events, seq_indexes[0], corr_ids[0])

        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        metric_logger.update(loss=sum(loss_dict_reduced_scaled.values()),
                             **loss_dict_reduced_scaled,
                             **loss_dict_reduced_unscaled)
        metric_logger.update(class_error=loss_dict_reduced['class_error'])

        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        results = postprocessors['bbox'](outputs, orig_target_sizes, 'vis')

        res = [(img_id, output) for img_id, output in zip(img_ids, results)]
        ress.extend(res)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)

    return ress


def dataset_single(image_set, args):
    entire_dataset = []
    scene = args.scene
    dataset_name = '%03d_%s_%s' % (args.datasetname, image_set, scene)
    entire_dataset.append(build_dataset_asyn(image_set, args, scene, dataset_name))
    return scene, ConcatDatasetCustom(entire_dataset), dataset_name
