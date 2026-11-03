"""모델 입력 만들기. 학습(과제 3·4·5)과 서비스(과제 8)가 똑같이 이 함수를 쓴다.

예측 입력
  past   (N, LOOKBACK, F_past)  : CO2·온도·습도(정규화) + 시간대·요일 + 일정 + 공간
  future (N, HORIZON, F_fut)    : 미래에도 알 수 있는 것 — 시간대, 일정(예정 인원)
출력
  (N, HORIZON, 2)               : 지금 값에서의 변화량(CO2, 온도) / 변화량 표준편차
"""
import numpy as np

from . import data as D

TIME_COLS = ["tod_sin", "tod_cos", "dow_sin", "dow_cos", "weekend"]
SCHED_COLS = ["sched_people", "sched_active"]


def past_cols(df, use_schedule=True):
    space_cols = [c for c in df.columns if c.startswith("space_")]
    return D.SENSORS + TIME_COLS + (SCHED_COLS if use_schedule else []) + space_cols


def future_cols(use_schedule=True):
    return ["tod_sin", "tod_cos"] + (SCHED_COLS if use_schedule else [])


class FeatureSpec:
    """정규화 값과 사용할 열을 한데 묶어 저장·복원한다(서비스가 학습과 같은 전처리를 쓰도록)."""

    def __init__(self, df_train, use_schedule=True):
        self.use_schedule = use_schedule
        self.past = past_cols(df_train, use_schedule)
        self.future = future_cols(use_schedule)
        self.sched_scale = float(max(df_train["sched_people"].max(), 1.0))
        sens = df_train[D.SENSORS]
        self.mu = sens.mean().values.astype(np.float32)
        self.sd = (sens.std().values + 1e-6).astype(np.float32)
        # 변화량 정규화: 학습 기간의 60분 변화량 표준편차
        d = df_train.groupby("space")[D.TARGETS].diff(D.HORIZON)
        self.dsd = (d.std().values + 1e-6).astype(np.float32)

    def to_dict(self):
        return dict(use_schedule=self.use_schedule, past=self.past, future=self.future, sched_scale=self.sched_scale,
                    mu=self.mu.tolist(), sd=self.sd.tolist(), dsd=self.dsd.tolist(),
                    lookback=D.LOOKBACK, horizon=D.HORIZON, step=D.STEP)

    @classmethod
    def from_dict(cls, d):
        s = cls.__new__(cls)
        s.use_schedule, s.past, s.future, s.sched_scale = d["use_schedule"], d["past"], d["future"], d["sched_scale"]
        s.mu, s.sd, s.dsd = (np.array(d[k], np.float32) for k in ("mu", "sd", "dsd"))
        return s

    def matrix(self, df):
        """df 전체를 한 번에 정규화한 (len(df), F) 행렬 두 개(과거용, 미래용)."""
        m = df[self.past].to_numpy(np.float32, copy=True)
        m[:, :3] = (m[:, :3] - self.mu) / self.sd
        for name in SCHED_COLS:
            if name == "sched_people" and name in self.past:
                m[:, self.past.index(name)] /= self.sched_scale
        f = df[self.future].to_numpy(np.float32, copy=True)
        if "sched_people" in self.future:
            f[:, self.future.index("sched_people")] /= self.sched_scale
        return np.nan_to_num(m), np.nan_to_num(f)

    def windows(self, df, idx, with_target=True):
        """기준 시점 idx(행 위치) 마다 과거·미래 창을 자른다. idx 행이 '지금'이다."""
        M, F = self.matrix(df)
        L, H = D.LOOKBACK, D.HORIZON
        past = np.stack([M[i - L + 1:i + 1] for i in idx])
        # 서비스에서는 미래 칸(시간대·일정)을 표 끝에 미리 붙여 넘긴다. 혹시 모자라면 0으로 채운다
        fut = np.stack([F[i + 1:i + 1 + H] if i + 1 + H <= len(F) else np.zeros((H, F.shape[1]), np.float32)
                        for i in idx])
        if not with_target:
            return past, fut
        y = df[D.TARGETS].to_numpy(np.float32)
        now = y[idx]
        tgt = np.stack([y[i + 1:i + 1 + H] for i in idx]) - now[:, None, :]
        return past, fut, tgt / self.dsd

    def denorm(self, pred, now):
        """모델 출력(정규화된 변화량) → 원래 단위의 예측값. now: (N, 2) 현재 CO2·온도."""
        return now[:, None, :] + pred * self.dsd
