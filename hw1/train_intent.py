import json
import pickle
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict

import torch
from tqdm import trange, tqdm
from torch.utils.data import DataLoader, Dataset 

from dataset import SeqClsDataset
from utils import Vocab
from model import SeqClassifier

import random

TRAIN = "train"
DEV = "eval"
SPLITS = [TRAIN, DEV]

torch.manual_seed(32)
random.seed(32)
torch.backends.cudnn.enabled = True
def main(args):
    print(torch.cuda.is_available())
    with open(args.cache_dir / "vocab.pkl", "rb") as f:
        vocab: Vocab = pickle.load(f)

    intent_idx_path = args.cache_dir / "intent2idx.json"
    intent2idx: Dict[str, int] = json.loads(intent_idx_path.read_text())

    data_paths = {split: args.data_dir / f"{split}.json" for split in SPLITS}
    data = {split: json.loads(path.read_text()) for split, path in data_paths.items()}
    datasets: Dict[str, SeqClsDataset] = {
        split: SeqClsDataset(split_data, vocab, intent2idx, args.max_len)
        for split, split_data in data.items()
    }

    # TODO: crecate DataLoader for train / dev datasets
    train_set = datasets[TRAIN]
    eval_set = datasets[DEV]

    train_loader = DataLoader(train_set, batch_size=args.batch_size, collate_fn = datasets[TRAIN].collate_fn, shuffle=True)
    eval_loader = DataLoader(eval_set, batch_size=args.batch_size, collate_fn = datasets[DEV].collate_fn, shuffle=False)

    embeddings = torch.load(args.cache_dir / "embeddings.pt")
    # TODO: init model and move model to target device(cpu / gpu)
    model = SeqClassifier(
        embeddings = embeddings,
        hidden_size = args.hidden_size,
        num_layers= args.num_layers,
        dropout = args.dropout,
        bidirectional = args.bidirectional,
        num_class = datasets[DEV].num_classes
    ).to(args.device)
    # TODO: init optimizer
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay  = 1e-5
    )
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr = args.lr,
        epochs=args.num_epoch,
        steps_per_epoch=len(train_loader),
        last_epoch=-1,
        verbose=False,
    )

    criterion = torch.nn.CrossEntropyLoss()

    epoch_pbar = trange(args.num_epoch, desc="Epoch")
    best_loss = 99999
    for epoch in epoch_pbar:
        model.train()
        # TODO: Training loop - iterate over train dataloader and update model weights
        # TODO: Evaluation loop - calculate accuracy and save model weights
        train_loss = 0
        train_acc = 0
        for i, batch in enumerate(tqdm(train_loader)):
            # print(batch)
            optimizer.zero_grad()
            #print(batch)
            outputs = model(batch['text'].to(args.device))
            #print(outputs)
            #assert False
            #print(outputs.shape)
            # print(batch['text'].shape)
            # print(batch['intent'].shape)
            # print(outputs.shape)
            # assert False
            loss = criterion(outputs, batch['intent'].to(args.device))
            _, train_pred = torch.max(outputs, 1)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_acc += (train_pred.cpu() == batch['intent'].cpu()).sum().item()
            
            # print(loss)
        scheduler.step()
            
        print(f'----TRAIN---- Epoch [{epoch+1}/{args.num_epoch}], ACC == {train_acc/len(train_set)}, _______  LOSS == {train_loss/len(train_loader)}]')
        
        eval_loss = 0
        eval_acc = 0
        with torch.no_grad():
            model.eval()
            for i, eval_batch in enumerate(tqdm(eval_loader)):
                outputs = model(eval_batch['text'].to(args.device))
                loss = criterion(outputs, eval_batch['intent'].to(args.device))
                _, eval_pred = torch.max(outputs, 1)
                eval_loss += loss.item()
                eval_acc += (eval_pred.cpu() == eval_batch['intent'].cpu()).sum().item()
                
            print(f'----VAL------ Epoch [{epoch+1}/{args.num_epoch}],ACC == {eval_acc/len(eval_set)}, ________ LOSS == {eval_loss/len(eval_loader)}]')

            #torch.save(model.state_dict(), f"./ckpt/intent/model_{epoch+1}.ckpt")
            print(f'Model saved')
            if eval_loss <= best_loss:
                best_epoch = epoch
                torch.save(model.state_dict(), f"./{args.ckpt_dir}/best_model.ckpt")
                best_loss = eval_loss

    print(f"best_epoch is {best_epoch+1} best_loss is {best_loss/len(eval_loader)}")

    # TODO: Inference on test set


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=Path,
        help="Directory to the dataset.",
        default="./data/intent/",
    )
    parser.add_argument(
        "--cache_dir",
        type=Path,
        help="Directory to the preprocessed caches.",
        default="./cache/intent/",
    )
    parser.add_argument(
        "--ckpt_dir",
        type=Path,
        help="Directory to save the model file.",
        default="./ckpt/intent/",
    )

    # data
    parser.add_argument("--max_len", type=int, default=32)

    # model
    parser.add_argument("--hidden_size", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--bidirectional", type=bool, default=True)

    # optimizer
    parser.add_argument("--lr", type=float, default=1e-3)

    # data loader
    parser.add_argument("--batch_size", type=int, default=128)

    # training
    parser.add_argument(
        "--device", type=torch.device, help="cpu, cuda, cuda:0, cuda:1", default="cuda:0"
    )
    parser.add_argument("--num_epoch", type=int, default=100)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    args.ckpt_dir.mkdir(parents=True, exist_ok=True)
    main(args)
