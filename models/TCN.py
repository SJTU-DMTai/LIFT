import torch
import torch.nn as nn

from layers.ts2vec.encoder import TS2VecEncoderWrapper, TSEncoder


class Model(nn.Module):
    def __init__(self, args):
        super().__init__()
        encoder = TSEncoder(input_dims=args.enc_in + 7,
                            output_dims=320,  # standard ts2vec backbone value
                            hidden_dims=64,  # standard ts2vec backbone value
                            depth=10)
        self.encoder = TS2VecEncoderWrapper(encoder, mask='all_true')
        self.pred_len = args.pred_len
        self.dim = args.c_out * args.pred_len

        # self.regressor = nn.Sequential(nn.Linear(320, 320), nn.ReLU(), nn.Linear(320, self.dim)).to(self.device)
        self.regressor = nn.Linear(320, self.dim)

    def forward(self, x, x_mark):
        x = torch.cat([x, x_mark], dim=-1)
        rep = self.encoder(x)
        y = self.regressor(rep)
        y = y.reshape(len(y), self.pred_len, -1)
        return y

class Model_Ensemble(Model):
    def __init__(self, args):
        super().__init__(args)
        depth = 10
        encoder = TSEncoder(input_dims=args.seq_len,
                            output_dims=320,  # standard ts2vec backbone value
                            hidden_dims=64,  # standard ts2vec backbone value
                            depth=depth)
        self.encoder_time = TS2VecEncoderWrapper(encoder, mask='all_true')
        self.regressor_time = nn.Linear(320, args.pred_len)

    def forward_individual(self, x, x_mark):
        rep = self.encoder_time.encoder.forward(x.transpose(1, 2))
        y1 = self.regressor_time(rep).transpose(1, 2)
        y2 = super().forward(x, x_mark)
        return y1, y2

    def forward(self, x, x_mark, w1=0.5, w2=0.5):
        y1, y2 = self.forward_individual(x, x_mark)
        return y1 * w1 + y2 * w2, y1, y2



