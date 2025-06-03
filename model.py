import torch
import torch.nn as nn
import torch.nn.functional as F


class Layer(nn.Module):
    def __init__(self, d_in, d_out, dropout=0.2):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Linear(d_in, d_out),
            nn.LayerNorm(d_out),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.layer(x)


class Encoder(nn.Module):
    def __init__(self, d_v, hidden_states, d_emb, n_layer, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([Layer(d_v, hidden_states[0])] +
                                    [Layer(hidden_states[_], hidden_states[_ + 1]) for _ in range(n_layer - 1)])
        self.last = nn.Linear(hidden_states[-1], d_emb)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        x = self.last(x)
        x = self.dropout(x)  # 最后应用 Dropout
        return x


class Decoder(nn.Module):
    def __init__(self, d_v, hidden_states, d_emb, n_layer, dropout=0.1):
        super().__init__()
        self.first = nn.Linear(d_emb, hidden_states[-1])
        self.mid = nn.ModuleList(
            [Layer(hidden_states[n_layer - 1 - _], hidden_states[n_layer - 2 - _]) for _ in range(n_layer - 1)])
        self.last = nn.Linear(hidden_states[0], d_v)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.first(x)
        for layer in self.mid:
            x = layer(x)
        x = self.dropout(x)
        x = self.last(x)
        return x


class Classifier(nn.Module):
    def __init__(self, d_emb, n_cls, dropout):
        super().__init__()
        self.linear = nn.Linear(d_emb, round(0.5 * d_emb))
        self.relu = nn.LeakyReLU()
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(round(0.5 * d_emb), n_cls)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.linear(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.sigmoid(x)
        return x


class AutoEncoder(nn.Module):
    def __init__(self, r_list, d_emb, n_enc_layer, n_dec_layer, dropout=0.1):
        super().__init__()
        n_view = len(r_list)
        enc_hidden_states = []
        dec_hidden_states = []

        for _ in range(n_view):
            temp_hidden_states = []
            temp_hidden_states_ = []

            for i in range(n_enc_layer):
                hd = round(d_emb * 2)
                hd = int(hd)
                temp_hidden_states.append(hd)
            for i in range(n_dec_layer):
                hd = round(d_emb * 2)
                hd = int(hd)
                temp_hidden_states_.append(hd)

            enc_hidden_states.append(temp_hidden_states)
            dec_hidden_states.append(temp_hidden_states_)

        self.encoder_list = nn.ModuleList(
            [Encoder(r_list[v], enc_hidden_states[v], d_emb, n_enc_layer, dropout) for v in range(n_view)])
        self.decoder_list = nn.ModuleList(
            [Decoder(r_list[v], dec_hidden_states[v], d_emb, n_dec_layer, dropout) for v in range(n_view)])

        self.n_view = n_view
        self.r_list = r_list
        self.d_emb = d_emb

    def forward(self, v_list, mask):
        mid_states = []
        for enc_i, enc in enumerate(self.encoder_list):
            mid_states.append(enc(v_list[enc_i]).unsqueeze(1))
        emb = torch.cat(mid_states, dim=1)
        rec_r = []
        for dec_i, dec in enumerate(self.decoder_list):
            rec_r.append(dec(emb))
        return emb, rec_r


class Model(nn.Module):
    def __init__(self, d_list, d_emb, n_enc_layer, n_dec_layer, n_cls, theta, dropout=0.1):
        super().__init__()
        self.ae = AutoEncoder(d_list, d_emb, n_enc_layer, n_dec_layer, dropout)
        self.classifier = Classifier(d_emb, n_cls, dropout)
        self.weights = nn.Parameter(torch.softmax(torch.zeros([1, len(d_list), 1]), dim=1))
        self.d_emb = d_emb

    def forward(self, emb, mask_v):
        embs, rec_v = self.ae(emb, mask_v)
        weight = torch.pow(self.weights.expand(embs.shape[0], -1, -1), 1)
        weight = torch.softmax(weight.masked_fill(mask_v.unsqueeze(2) == 0, -1e9), dim=1)
        emb_fus = torch.sum(embs * weight, dim=1)
        pred = self.classifier(emb_fus)
        return pred, rec_v, embs
