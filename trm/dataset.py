"""
Dataset for loading pre-extracted hidden states and labels.
"""

import json
import os
import torch
from torch.utils.data import Dataset


LABEL_MAP = {'A': 0, 'B': 1, 'C': 2, 'D': 3}


class HiddenStateDataset(Dataset):
    """
    Loads pre-extracted hidden state .pt files and corresponding labels.

    Each .pt file contains a tensor of shape (1, seq_len, hidden_dim).
    We split it into:
        x = all tokens except the last (1, seq_len-1, D) — the context
        z = the last token (1, 1, D) — the generation prompt token
    """

    def __init__(self, hidden_states_dir, labels_file, label_key='answer'):
        """
        Args:
            hidden_states_dir: Directory containing {id}.pt files
            labels_file: Path to jsonl file with labels
            label_key: Key for the answer in the jsonl ('answer' for train, 'label' for val)
        """
        self.hidden_states_dir = hidden_states_dir

        # Load labels
        self.samples = []
        with open(labels_file, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line.strip())
                sample_id = item['id']
                pt_path = os.path.join(hidden_states_dir, f"{sample_id}.pt")
                if os.path.exists(pt_path):
                    label_str = item[label_key].strip().upper()
                    if label_str in LABEL_MAP:
                        self.samples.append({
                            'id': sample_id,
                            'pt_path': pt_path,
                            'label': LABEL_MAP[label_str],
                            'tag': item.get('tag', 'Other'),
                        })

        print(f"Loaded {len(self.samples)} samples from {hidden_states_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # Load hidden states: (1, seq_len, D)
        hidden = torch.load(sample['pt_path'], map_location='cpu', weights_only=True)
        if hidden.dim() == 3:
            hidden = hidden.squeeze(0)  # (seq_len, D)

        # Split: x = all except last, z = last token
        x = hidden[:-1]   # (seq_len-1, D)
        z = hidden[-1:]    # (1, D)

        return {
            'x': x,
            'z': z,
            'label': sample['label'],
            'id': sample['id'],
            'tag': sample['tag'],
        }


def collate_fn(batch):
    """
    Custom collate that pads x to max seq_len in the batch.
    z is always (1, D) so no padding needed.
    """
    # Find max x length in this batch
    max_len = max(item['x'].shape[0] for item in batch)
    D = batch[0]['x'].shape[1]

    B = len(batch)
    x_padded = torch.zeros(B, max_len, D)
    x_mask = torch.zeros(B, max_len)
    z_batch = torch.zeros(B, 1, D)
    labels = torch.zeros(B, dtype=torch.long)
    ids = []
    tags = []

    for i, item in enumerate(batch):
        seq_len = item['x'].shape[0]
        x_padded[i, :seq_len] = item['x']
        x_mask[i, :seq_len] = 1.0
        z_batch[i] = item['z']
        labels[i] = item['label']
        ids.append(item['id'])
        tags.append(item['tag'])

    return {
        'x': x_padded,
        'x_mask': x_mask,
        'z': z_batch,
        'labels': labels,
        'ids': ids,
        'tags': tags,
    }
