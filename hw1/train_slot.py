import json
import pickle
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict

import torch
from tqdm import trange, tqdm
from torch.utils.data import DataLoader, Dataset 

from slot_dataset import SeqTagDataset
from utils import Vocab
from slot_model import SeqTagger

from seqeval.metrics import classification_report
from seqeval.metrics import f1_score
from seqeval.metrics import accuracy_score
from seqeval.scheme import IOB2
from sklearn.metrics import f1_score as f1

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

    tag_idx_path = args.cache_dir / "tag2idx.json"
    tag2idx: Dict[str, int] = json.loads(tag_idx_path.read_text())

    tag2idx["pad"] = 9

    data_paths = {split: args.data_dir / f"{split}.json" for split in SPLITS}
    data = {split: json.loads(path.read_text()) for split, path in data_paths.items()}
    
    max_len = max(len(x['tags']) for x in data['train']) + 1
    

    datasets: Dict[str, SeqTagDataset] = {
        split: SeqTagDataset(split_data, vocab, tag2idx, max_len)
        for split, split_data in data.items()
    }

    # TODO: crecate DataLoader for train / dev datasets
    train_set = datasets[TRAIN]
    eval_set = datasets[DEV]

    train_loader = DataLoader(train_set, batch_size=args.batch_size, collate_fn = datasets[TRAIN].collate_fn, shuffle=True)
    eval_loader = DataLoader(eval_set, batch_size=args.batch_size, collate_fn = datasets[DEV].collate_fn, shuffle=False)

    embeddings = torch.load(args.cache_dir / "embeddings.pt")
    # TODO: init model and move model to target device(cpu / gpu)
    model = SeqTagger(
        embeddings = embeddings,
        hidden_size = args.hidden_size,
        num_layers= args.num_layers,
        dropout = args.dropout,
        bidirectional = args.bidirectional,
        num_class = datasets[DEV].num_classes,
        max_len = max_len
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
            outputs = model(batch['tokens'].to(args.device))
            outputs = torch.transpose(outputs, 1, 2)
            # print(outputs.shape)
            # assert False
            # print(batch['tokens'].shape)
            # print(batch['tags'].shape)
            # print(outputs.shape)
            # assert False
            loss = criterion(outputs, batch['tags'].to(args.device))
            _, train_pred = torch.max(outputs, 1)

            #print(train_pred.shape)
            train_pred = train_pred.tolist()
            #print(train_pred)
            #print(batch['tags_unpad'])
            for tmp in range(len(batch)):
                train_pred[tmp] = train_pred[tmp][:][:batch['length'][tmp]]
            #print(train_pred)
            for tmp in range(len(batch)):
                train_acc += (train_pred[tmp] == batch['tags_unpad'][tmp])
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
            # print(loss)
            
        print(f'----TRAIN---- Epoch [{epoch+1}/{args.num_epoch}], ACC == {(train_acc/(len(train_loader)*len(batch))):.4f}, _______  LOSS == {train_loss/len(train_loader)}]')
        #print(f'----TRAIN---- Epoch [{epoch+1}/{args.num_epoch}], _______  LOSS == {train_loss/len(train_loader)}]')
        
        eval_loss = 0
        eval_acc = 0
        with torch.no_grad():
            model.eval()
            for i, eval_batch in enumerate(tqdm(eval_loader)):
                outputs = model(eval_batch['tokens'].to(args.device))
                outputs = torch.transpose(outputs, 1, 2)
                loss = criterion(outputs, eval_batch['tags'].to(args.device))
                _, eval_pred = torch.max(outputs, 1)
                eval_pred = eval_pred.tolist()

                for tmp in range(len(eval_batch)):
                    eval_pred[tmp] = eval_pred[tmp][:][:eval_batch['length'][tmp]]
                
                for tmp in range(len(eval_batch)):
                    eval_acc += (eval_pred[tmp] == eval_batch['tags_unpad'][tmp])
                #eval_acc += (eval_pred.cpu() == eval_batch['tags'].cpu()).sum().item()
                eval_loss += loss.item()

            print(f'----VAL------ Epoch [{epoch+1}/{args.num_epoch}],ACC == {(eval_acc/(len(eval_loader)*len(eval_batch))):.4f}, ________ LOSS == {eval_loss/len(eval_loader)}]')
            #print(f'----VAL------ Epoch [{epoch+1}/{args.num_epoch}], ________ LOSS == {eval_loss/len(eval_loader)}]')

            #torch.save(model.state_dict(), f"./ckpt/intent/model_{epoch+1}.ckpt")
            if eval_loss <= best_loss:
                best_epoch = epoch
                torch.save(model.state_dict(), f"./{args.ckpt_dir}/best_model.ckpt")
                best_loss = eval_loss
                print(f'Model saved')

    print(f"best_epoch is {best_epoch+1} best_loss is {best_loss/len(eval_loader)}")


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=Path,
        help="Directory to the dataset.",
        default="./data/slot/",
    )
    parser.add_argument(
        "--cache_dir",
        type=Path,
        help="Directory to the preprocessed caches.",
        default="./cache/slot/",
    )
    parser.add_argument(
        "--ckpt_dir",
        type=Path,
        help="Directory to save the model file.",
        default="./ckpt/slot/",
    )

    # model
    parser.add_argument("--hidden_size", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--bidirectional", type=bool, default=True)

    # optimizer
    parser.add_argument("--lr", type=float, default=1e-3)

    # data loader
    parser.add_argument("--batch_size", type=int, default=128)

    # training
    parser.add_argument(
        "--device", type=torch.device, help="cpu, cuda, cuda:0, cuda:1", default="cuda:0"
    )
    parser.add_argument("--num_epoch", type=int, default=50)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    args.ckpt_dir.mkdir(parents=True, exist_ok=True)
    main(args)
