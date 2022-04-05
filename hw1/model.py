from typing import Dict

import torch
from torch.nn import Embedding
from torch.autograd import Variable

class SeqClassifier(torch.nn.Module):
    def __init__(
        self,
        embeddings: torch.tensor,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        bidirectional: bool,
        num_class: int,
    ) -> None:
        super(SeqClassifier, self).__init__()
        self.embed = Embedding.from_pretrained(embeddings, freeze=False)
        # TODO: model architecture
        self.embeddings = embeddings
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.dropout = torch.nn.Dropout(dropout)
        self.bidirectional = bidirectional
        #self.num_class = num_class
        self.gru = torch.nn.RNN(
            input_size = 300,
            hidden_size = hidden_size,
            num_layers = num_layers,
            dropout = dropout,
            bidirectional = bidirectional,
            batch_first = True
        )
        self.embed_norm = torch.nn.LayerNorm(300)
        self.seq_norm = torch.nn.LayerNorm(hidden_size*2*3)
        self.fc1_norm = torch.nn.LayerNorm(hidden_size)
        #self.fc2_norm = torch.nn.LayerNorm(hidden_size*2)
        self.fc1 = torch.nn.Linear(hidden_size*2*3, hidden_size)
        self.fc2 = torch.nn.Linear(hidden_size*4, hidden_size*2)
        self.out_fc = torch.nn.Linear(hidden_size, num_class)
        self.al = torch.nn.Tanh()

    @property
    def encoder_output_size(self) -> int:
        # TODO: calculate the output dimension of rnn
        raise NotImplementedError

    def forward(self, batch) -> Dict[str, torch.Tensor]:
        # TODO: implement model forward
        # print(batch[0])
        # assert False
        batch = self.embed(batch)
        batch = self.embed_norm(batch)
        #print(batch.shape)
        out, _ = self.gru(batch)
        #out = self.seq_norm(out)
        #print(out.shape)
        out = torch.cat((out.min(dim=1).values, out.max(dim=1).values, out.mean(dim=1)), 1)
        #print(out.shape)
        #out = out.mean(dim=1)
        #out = out[:,-1,:].view(out.size(0), -1)
        out = self.dropout(out)
        #out = self.seq_norm(out)
        out = self.al(out)
        #out = self.seq_norm(out)

        out = self.fc1(out)
        #print(out.shape)
        # out = self.al(out)
        out = self.dropout(out)
        out = self.fc1_norm(out)
        out = self.al(out)
        #out = self.fc1_norm(out)
        

        # out = self.fc2(out)
        # out = self.fc2_norm(out)
        # out = self.dropout(out)
        # out = self.al(out)
        #print(out.shape)

        out = self.out_fc(out)

        #out = self.sm(out)
        #print(out.shape)
        return out
