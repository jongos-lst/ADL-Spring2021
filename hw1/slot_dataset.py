from typing import List, Dict

from torch.utils.data import Dataset
import torch
import re

from utils import Vocab, pad_to_len


class SeqTagDataset(Dataset):
    def __init__(
        self,
        data: List[Dict],
        vocab: Vocab,
        label_mapping: Dict[str, int],
        max_len: int,
    ):
        self.data = data
        self.vocab = vocab
        self.label_mapping = label_mapping
        self._idx2label = {idx: intent for intent, idx in self.label_mapping.items()}
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index) -> Dict:
        instance = self.data[index]
        return instance

    @property
    def num_classes(self) -> int:
        return len(self.label_mapping)

    def collate_fn(self, samples: List[Dict]) -> Dict:
        # TODO: implement collate_fn

        batch_inputs = {
            "tokens" : [],
            "tags" : [],
            "tags_unpad":[],
            "id" : [],
            "length" : [],
            "tags_raw" : []
        }
        # print(self.vocab.token_to_id('i'))

        # print(samples[1]['tokens'])
        # print(samples[1]['tags'])
        if 'tags' in samples[0].keys():
            for i in range(len(samples)):
                batch_inputs['tokens'].append(samples[i]['tokens'])
                batch_inputs['tags'].append([self.label2idx(x) for x in samples[i]['tags']])
                batch_inputs['tags_unpad'].append([self.label2idx(x) for x in samples[i]['tags']])
                batch_inputs['length'].append(len(samples[i]['tokens']))
                batch_inputs['tags_raw'].append((samples[i]['tags']))
            batch_inputs['tags'] = pad_to_len(batch_inputs['tags'], self.max_len, 9)
        else:
            for i in range(len(samples)):
                batch_inputs['tokens'].append(samples[i]['tokens'])
                batch_inputs['length'].append(len(samples[i]['tokens']))

        # print(batch_inputs)
        if 'tags' in samples[0].keys():
            out = {
                'tokens': torch.LongTensor(self.vocab.encode_batch(batch_inputs['tokens'], self.max_len)),
                'tags': torch.LongTensor(batch_inputs['tags']),
                'tags_unpad': batch_inputs['tags_unpad'],
                'id': [x['id'] for x in samples],
                'length' : batch_inputs['length'],
                'tags_raw' : batch_inputs['tags_raw']
            }
        else:
            out = {
                'tokens': torch.LongTensor(self.vocab.encode_batch(batch_inputs['tokens'], self.max_len)),
                'id': [x['id'] for x in samples],
                'length' : batch_inputs['length']
            }
        # print(out['tokens'][1])
        # print(out['tags'][1])
        # assert False
        return out

    def label2idx(self, label: str):
        return self.label_mapping[label]

    def idx2label(self, idx: int):
        return self._idx2label[idx]
    

