"""Full-timeline causal pairs for new matched executor experiments."""
import numpy as np
import torch

from dual_rgbt.data import LasHeRPairsV2, crop_pair, read_pair


def sample_full_timeline_pair(valid_frames, rng, max_interval=200):
    valid = np.asarray(valid_frames)
    candidates = valid[1:][np.diff(valid) <= max_interval]
    assert len(candidates) > 0
    search_id = int(candidates[rng.randint(len(candidates))])
    recent = valid[(valid < search_id) & (valid >= search_id - max_interval)]
    recent_id = int(recent[rng.randint(len(recent))])
    return int(valid[0]), recent_id, search_id


class LasHeRPairsC2(LasHeRPairsV2):
    def __getitem__(self, index):
        rng = np.random.RandomState(self.seed + index)
        sequence = self.sequences[rng.randint(len(self.sequences))]
        first_id, recent_id, search_id = sample_full_timeline_pair(
            sequence["valid_frames"], rng, self.max_interval)
        boxes = np.asarray(sequence["boxes"], dtype=np.float32)
        templates = [
            crop_pair(read_pair(sequence, frame), boxes[frame], 2., 128)[0]
            for frame in (first_id, recent_id)]
        search, box, _ = crop_pair(
            read_pair(sequence, search_id), boxes[search_id], 4., 256, rng)
        return dict(templates=torch.stack(templates), search=search,
                    target=torch.from_numpy(box), sequence=sequence["name"],
                    frame=search_id)
