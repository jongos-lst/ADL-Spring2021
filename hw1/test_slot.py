import json
import pickle
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict

import torch
from torch.utils.data import DataLoader, Dataset 

from tqdm import trange, tqdm
from slot_dataset import SeqTagDataset
from slot_model import SeqTagger
from utils import Vocab

import pandas as pd

def main(args):
    with open(args.cache_dir / "vocab.pkl", "rb") as f:
        vocab: Vocab = pickle.load(f)

    tag_idx_path = args.cache_dir / "tag2idx.json"
    tag2idx: Dict[str, int] = json.loads(tag_idx_path.read_text())

    data = json.loads(args.test_file.read_text())
    dataset = SeqTagDataset(data, vocab, tag2idx, 36)
    # TODO: crecate DataLoader for test dataset

    dataset._idx2label[9] = "O"

    print(dataset.label_mapping)

    test_loader = DataLoader(dataset, batch_size=args.batch_size, collate_fn = dataset.collate_fn, shuffle=False)

    embeddings = torch.load(args.cache_dir / "embeddings.pt")

    print(dataset.num_classes)

    model = SeqTagger(
        embeddings,
        args.hidden_size,
        args.num_layers,
        args.dropout,
        args.bidirectional,
        dataset.num_classes + 1,
        36
    )
    model.eval()

    ckpt = torch.load(args.ckpt_path)
    # load weights into model
    model.load_state_dict(ckpt)

    column_names = ["id", "tags"]
    df = pd.DataFrame(columns = column_names)

    # TODO: predict dataset
    for i, batch in enumerate(tqdm(test_loader)):
        outputs = model(batch['tokens'].to(args.device))
        outputs = torch.transpose(outputs, 1, 2)
        _, test_pred = torch.max(outputs, 1)
        test_pred = test_pred.tolist()
        
        for tmp in range(len(batch)):
            test_pred[tmp] = test_pred[tmp][:][:batch['length'][tmp]]
            
        for i in range(len(batch['tokens'])):
            df2 = pd.DataFrame(
                [[batch['id'][i], ' '.join([ dataset.idx2label(int(test_pred[i][j])) for j in range(batch['length'][i])]) ]],
                   columns=['id', 'tags'])
            
            df = pd.concat([df, df2]) 
    # TODO: write prediction to file (args.pred_file)
    df.to_csv(args.pred_file, index=False)

def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--test_file",
        type=Path,
        help="Path to the test file.",
        default="./data/slot/test.json"
    )
    parser.add_argument(
        "--cache_dir",
        type=Path,
        help="Directory to the preprocessed caches.",
        default="./cache/slot/",
    )
    parser.add_argument(
        "--ckpt_path",
        type=Path,
        help="Path to model checkpoint.",
        default="./ckpt/slot/best_model.ckpt"
    )
    parser.add_argument("--pred_file", type=Path, default="pred_slot.csv")
    # data
    parser.add_argument("--max_len", type=int, default=32)
    # model
    parser.add_argument("--hidden_size", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--bidirectional", type=bool, default=True)

    # data loader
    parser.add_argument("--batch_size", type=int, default=128)

    parser.add_argument(
        "--device", type=torch.device, help="cpu, cuda, cuda:0, cuda:1", default="cpu"
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
