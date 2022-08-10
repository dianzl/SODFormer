import torch

def to_cuda(samples, events, targets, input_type, device):
    if input_type == 'frame_only' or input_type == 'multi-modal':
        samples = samples.to(device, non_blocking=True)
    else:
        samples = samples
    if input_type == 'event_only' or input_type == 'multi-modal':
        events = events.to(device, non_blocking=True)
    else:
        events = events
    targets = [{k: v.to(device, non_blocking=True) for k, v in t.items()} for t in targets]
    return samples, events, targets

class data_prefetcher():
    def __init__(self, loader, device, input_type, prefetch=True):
        self.loader = iter(loader)
        self.prefetch = prefetch
        self.device = device
        self.input_type = input_type
        if prefetch:
            self.stream = torch.cuda.Stream()
            self.preload()

    def preload(self):
        try:
            self.next_inputs, self.next_indexes = next(self.loader)
            self.next_samples, self.next_events, self.next_targets = self.next_inputs
            self.next_seq_indexes, self.next_img_ids = self.next_indexes
        except StopIteration:
            self.next_samples = None
            self.next_events = None
            self.next_targets = None
            self.next_seq_indexes = None
            self.next_img_ids = None
            return
        with torch.cuda.stream(self.stream):
            self.next_samples, self.next_events, self.next_targets = to_cuda(self.next_samples,
                                                                             self.next_events,
                                                                             self.next_targets,
                                                                             self.input_type,
                                                                             self.device)
    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        samples = self.next_samples
        targets = self.next_targets
        events = self.next_events
        seq_indexes = self.next_seq_indexes
        img_ids = self.next_img_ids
        if not self.input_type == 'event_only' and samples is not None:
            samples.record_stream(torch.cuda.current_stream())
        if not self.input_type == 'frame_only' and events is not None:
            events.record_stream(torch.cuda.current_stream())
        if targets is not None:
            for t in targets:
                for _, v in t.items():
                    v.record_stream(torch.cuda.current_stream())
        self.preload()
        return samples, events, targets, seq_indexes, img_ids
