import json
import pickle
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict

import torch
from torch.utils.data import DataLoader, Dataset 

from tqdm import trange, tqdm
from dataset import SeqClsDataset
from model import SeqClassifier
from utils import Vocab

import pandas as pd

def main(args):
    with open(args.cache_dir / "vocab.pkl", "rb") as f:
        vocab: Vocab = pickle.load(f)

    intent_idx_path = args.cache_dir / "intent2idx.json"
    intent2idx: Dict[str, int] = json.loads(intent_idx_path.read_text())

    data = json.loads(args.test_file.read_text())
    dataset = SeqClsDataset(data, vocab, intent2idx, args.max_len)
    # TODO: crecate DataLoader for test dataset

    test_loader = DataLoader(dataset, batch_size=args.batch_size, collate_fn = dataset.collate_fn, shuffle=False)

    embeddings = torch.load(args.cache_dir / "embeddings.pt")

    model = SeqClassifier(
        embeddings,
        args.hidden_size,
        args.num_layers,
        args.dropout,
        args.bidirectional,
        dataset.num_classes,
    )
    model.eval()

    ckpt = torch.load(args.ckpt_path)
    # load weights into model
    model.load_state_dict(ckpt)

    column_names = ["id", "intent"]
    df = pd.DataFrame(columns = column_names)

    # TODO: predict dataset
    for i, batch in enumerate(tqdm(test_loader)):
        outputs = model(batch['text'].to(args.device))
        _, test_pred = torch.max(outputs, 1)
        for i in range(len(batch['text'])):
            df2 = pd.DataFrame(
                [[batch['id'][i], dataset.idx2label(int(test_pred[i])) ]],
                   columns=['id', 'intent'])
            df = pd.concat([df, df2]) 
    # TODO: write prediction to file (args.pred_file)
    df.to_csv(args.pred_file, index=False)

def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--test_file",
        type=Path,
        help="Path to the test file.",
        default="./data/intent/test.json"
    )
    parser.add_argument(
        "--cache_dir",
        type=Path,
        help="Directory to the preprocessed caches.",
        default="./cache/intent/",
    )
    parser.add_argument(
        "--ckpt_path",
        type=Path,
        help="Path to model checkpoint.",
        default="./ckpt/intent/best_model.ckpt"
    )
    parser.add_argument("--pred_file", type=Path, default="pred.intent.csv")
    # data
    parser.add_argument("--max_len", type=int, default=32)
    # model
    parser.add_argument("--hidden_size", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)
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
