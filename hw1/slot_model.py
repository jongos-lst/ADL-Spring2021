from typing import Dict

import torch
from torch.nn import Embedding
from torch.autograd import Variable

def weights_init_normal(m):
    '''Takes in a module and initializes all linear layers with weight
        values taken from a normal distribution.'''

    classname = m.__class__.__name__
    # for every Linear layer in a model
    if classname.find('Linear') != -1:
        y = m.in_features
    # m.weight.data shoud be taken from a normal distribution
        m.weight.data.normal_(0.0,1/np.sqrt(y))
    # m.bias.data should be 0
        m.bias.data.fill_(0)

class SeqTagger(torch.nn.Module):
    def __init__(
        self,
        embeddings: torch.tensor,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        bidirectional: bool,
        num_class: int,
        max_len: int,
    ) -> None:
        super(SeqTagger, self).__init__()
        self.embed = Embedding.from_pretrained(embeddings, freeze=False)
        # TODO: model architecture
        self.embeddings = embeddings
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.dropout = torch.nn.Dropout(dropout)
        self.bidirectional = bidirectional
        self.num_class = num_class
        self.max_len = max_len
        self.gru = torch.nn.RNN(
            input_size = 300,
            hidden_size = hidden_size,
            num_layers = num_layers,
            dropout = dropout,
            bidirectional = bidirectional,
            batch_first = True
        )
        weights_init_normal(self.gru)
        self.embed_norm = torch.nn.LayerNorm(300)
        self.seq_norm = torch.nn.LayerNorm(hidden_size*2*3)
        self.fc1_norm = torch.nn.LayerNorm(hidden_size*4)
        self.fc2_norm = torch.nn.LayerNorm(hidden_size*2)
        self.fc1 = torch.nn.Linear(hidden_size*2, hidden_size*4)
        self.fc2 = torch.nn.Linear(hidden_size*4, hidden_size*2)
        self.out_fc = torch.nn.Linear(hidden_size*2, num_class)
        self.al = torch.nn.Tanh()

    @property
    def encoder_output_size(self) -> int:
        # TODO: calculate the output dimension of rnn
        raise NotImplementedError

    def forward(self, batch) -> Dict[str, torch.Tensor]:
        # TODO: implement model forward
        batch = self.embed(batch)
        batch = self.embed_norm(batch)
        out, _ = self.gru(batch)
        #out = torch.cat((out.min(dim=1).values, out.max(dim=1).values, out.mean(dim=1)), 1)
        out = self.dropout(out)
        out = self.al(out)

        out = self.fc1(out)
        out = self.dropout(out)
        out = self.fc1_norm(out)
        out = self.al(out)

        out = self.fc2(out)
        out = self.dropout(out)
        out = self.fc2_norm(out)
        out = self.al(out)

        out = self.out_fc(out)
        #out = torch.reshape(out, (batch.size(0), self.num_class, self.max_len))
        return out
