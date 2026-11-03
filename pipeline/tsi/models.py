"""예측(GRU·LSTM·TCN), 이상 탐지(오토인코더), 상태 분류(MLP) 모델. 모두 ONNX·TensorRT로 바꿀 수 있는 연산만 쓴다."""
import torch
from torch import nn


class RNNForecaster(nn.Module):
    """과거 창을 GRU/LSTM으로 요약 + 미래 공변량(시간대·일정)을 펼쳐 붙여 60분을 한 번에 출력."""

    def __init__(self, f_past, f_fut, horizon, n_out=2, hidden=64, cell="gru"):
        super().__init__()
        rnn = nn.GRU if cell == "gru" else nn.LSTM
        self.rnn = rnn(f_past, hidden, num_layers=2, batch_first=True, dropout=0.1)
        self.head = nn.Sequential(nn.Linear(hidden + horizon * f_fut, 128), nn.ReLU(),
                                  nn.Linear(128, horizon * n_out))
        self.h, self.o = horizon, n_out

    def forward(self, past, fut):
        out, _ = self.rnn(past)
        z = torch.cat([out[:, -1], fut.flatten(1)], dim=1)
        return self.head(z).view(-1, self.h, self.o)


class CausalBlock(nn.Module):
    def __init__(self, c, dilation):
        super().__init__()
        self.pad = 2 * dilation
        self.conv1 = nn.Conv1d(c, c, 3, dilation=dilation)
        self.conv2 = nn.Conv1d(c, c, 3, dilation=dilation)
        self.act = nn.ReLU()

    def forward(self, x):
        y = self.act(self.conv1(nn.functional.pad(x, (self.pad, 0))))
        y = self.act(self.conv2(nn.functional.pad(y, (self.pad, 0))))
        return x + y


class TCNForecaster(nn.Module):
    def __init__(self, f_past, f_fut, horizon, n_out=2, channels=64, dilations=(1, 2, 4, 8)):
        super().__init__()
        self.inp = nn.Conv1d(f_past, channels, 1)
        self.blocks = nn.Sequential(*[CausalBlock(channels, d) for d in dilations])
        self.head = nn.Sequential(nn.Linear(channels + horizon * f_fut, 128), nn.ReLU(),
                                  nn.Linear(128, horizon * n_out))
        self.h, self.o = horizon, n_out

    def forward(self, past, fut):
        x = self.blocks(self.inp(past.transpose(1, 2)))
        z = torch.cat([x[:, :, -1], fut.flatten(1)], dim=1)
        return self.head(z).view(-1, self.h, self.o)


def make_forecaster(kind, f_past, f_fut, horizon):
    if kind == "tcn":
        return TCNForecaster(f_past, f_fut, horizon)
    return RNNForecaster(f_past, f_fut, horizon, cell=kind)


class AutoEncoder(nn.Module):
    """정상 창(센서 3개 × 과거 창)을 압축했다가 되살린다. 복원 오차가 크면 이상."""

    def __init__(self, n_in, bottleneck=8):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(n_in, 64), nn.ReLU(), nn.Linear(64, bottleneck))
        self.dec = nn.Sequential(nn.Linear(bottleneck, 64), nn.ReLU(), nn.Linear(64, n_in))

    def forward(self, x):
        return self.dec(self.enc(x))


class MLPClassifier(nn.Module):
    def __init__(self, n_in, n_out=2):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_in, 64), nn.ReLU(), nn.Dropout(0.1),
                                 nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, n_out))

    def forward(self, x):
        return self.net(x)
