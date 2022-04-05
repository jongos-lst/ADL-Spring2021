from typing import List, Dict

from torch.utils.data import Dataset
import torch
import re

from utils import Vocab


class SeqClsDataset(Dataset):
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

        # for i in samples:
        #     #print(i)
        #     i['text'] = str(i['text']).split(" ")
        
        batch_inputs = {
            "text" : [],
            "intent" : [],
            "id" : []
        }
        # print(self.vocab.token_to_id('i'))

        # print(samples[3]['text'])
        # print(samples[3]['intent'])
        if 'intent' in samples[0].keys():
            for i in range(len(samples)):
                batch_inputs['text'].append(samples[i]['text'].split())
                batch_inputs['intent'].append(self.label2idx(samples[i]['intent']))
        else:
            for i in range(len(samples)):
                batch_inputs['text'].append(samples[i]['text'].split())

        
        # print(batch_inputs)
        if 'intent' in samples[0].keys():
            out = {
                'text': torch.LongTensor(self.vocab.encode_batch(batch_inputs['text'], self.max_len)),
                'intent': torch.LongTensor(batch_inputs['intent']),
                'id': [x['id'] for x in samples]
            }
        else:
            out = {
                'text': torch.LongTensor(self.vocab.encode_batch(batch_inputs['text'], self.max_len)),
                'id': [x['id'] for x in samples]
            }
        # print(out['text'][3])
        # print(out['intent'][3])
        # assert False
        return out

    def label2idx(self, label: str):
        return self.label_mapping[label]

    def idx2label(self, idx: int):
        return self._idx2label[idx]
