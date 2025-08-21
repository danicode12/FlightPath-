# src/ekf_model.py
# Coordinated-Turn EKF (with optional wind) for ENU trajectories.
# Expected track columns: time, E,N,U, vE,vN,vU, turn_rate, dt
# Optional: wind_E, wind_N (ENU m/s). If use_wind=True, dynamics use airspeed.

from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable, Optional, Tuple, Dict
import numpy as np
import pandas as pd


# -----------------------------
# Parameters
# -----------------------------
@dataclass
class EKFParams:
    # Process noise stdevs
    sigma_pos: float = 5.0       # m
    sigma_vel: float = 1.0       # m/s
    sigma_omega: float = 0.01    # rad/s
    # Measurement noise (position only)
    sigma_meas_pos: float = 30.0 # m


# -----------------------------
# EKF Class
# -----------------------------
class CoordinatedTurnEKF:
    """
    7D state: x = [E, N, U, vE, vN, vU, omega]
    Measurement: z = [E, N, U]
    If use_wind=True, the turn kinematics apply to AIR velocity (v - wind),
    and wind is assumed constant over each small step.
    """
    def __init__(self, params: EKFParams = EKFParams(), use_wind: bool = False):
        self.p = params
        self.use_wind = use_wind

    # --- dynamics ---
    @staticmethod
    def _f_step_no_wind(x: np.ndarray, dt: float) -> np.ndarray:
        E,N,U,vE,vN,vU,om = x
        aE = -om * vN
        aN =  om * vE
        E2  = E  + vE*dt + 0.5*aE*dt*dt
        N2  = N  + vN*dt + 0.5*aN*dt*dt
        U2  = U  + vU*dt
        vE2 = vE + aE*dt
        vN2 = vN + aN*dt
        vU2 = vU
        return np.array([E2,N2,U2,vE2,vN2,vU2,om], dtype=float)

    @staticmethod
    def _F_no_wind(x: np.ndarray, dt: float) -> np.ndarray:
        _,_,_,vE,vN,_,om = x
        F = np.eye(7)
        F[0,3] = dt;     F[0,4] = -0.5*om*dt*dt; F[0,6] = -0.5*vN*dt*dt
        F[1,4] = dt;     F[1,3] =  0.5*om*dt*dt; F[1,6] =  0.5*vE*dt*dt
        F[2,5] = dt
        F[3,4] = -om*dt; F[3,6] = -vN*dt
        F[4,3] =  om*dt; F[4,6] =  vE*dt
        return F

    @staticmethod
    def _f_step_wind(x: np.ndarray, dt: float, wE: float, wN: float) -> np.ndarray:
        # Apply coordinated turn to AIR velocity, then add wind to get ground velocity.
        E,N,U,vE,vN,vU,om = x
        vEa, vNa = vE - wE, vN - wN
        aEa, aNa = -om * vNa, om * vEa
        # ground velocity = air + wind; ground accel = air accel
        E2  = E  + (vEa + wE)*dt + 0.5*aEa*dt*dt
        N2  = N  + (vNa + wN)*dt + 0.5*aNa*dt*dt
        U2  = U  + vU*dt
        vEa2 = vEa + aEa*dt
        vNa2 = vNa + aNa*dt
        vE2, vN2 = vEa2 + wE, vNa2 + wN
        vU2 = vU
        return np.array([E2,N2,U2,vE2,vN2,vU2,om], dtype=float)

    @staticmethod
    def _F_wind(x: np.ndarray, dt: float, wE: float, wN: float) -> np.ndarray:
        # Jacobian w.r.t state; wind treated as exogenous constants.
        _,_,_,vE,vN,_,om = x
        vEa, vNa = vE - wE, vN - wN
        F = np.eye(7)
        F[0,3] = dt;     F[0,4] = -0.5*om*dt*dt; F[0,6] = -0.5*vNa*dt*dt
        F[1,4] = dt;     F[1,3] =  0.5*om*dt*dt; F[1,6] =  0.5*vEa*dt*dt
        F[2,5] = dt
        F[3,4] = -om*dt; F[3,6] = -vNa*dt
        F[4,3] =  om*dt; F[4,6] =  vEa*dt
        return F

    @staticmethod
    def H_matrix() -> np.ndarray:
        H = np.zeros((3,7))
        H[0,0]=H[1,1]=H[2,2]=1.0
        return H

    def Q_process(self, dt: float) -> np.ndarray:
        p = self.p
        q = np.diag([p.sigma_pos**2]*3 + [p.sigma_vel**2]*3 + [p.sigma_omega**2])
        return q * dt

    def R_meas(self) -> np.ndarray:
        p = self.p
        return np.diag([p.sigma_meas_pos**2]*3)

    # --- filter a single run (DataFrame) ---
    def filter_track(self, track: pd.DataFrame) -> pd.DataFrame:
        """
        Returns filtered states aligned to 'time'.
        Required columns: time,E,N,U,vE,vN,vU,turn_rate,dt
        Optional columns (if use_wind=True): wind_E, wind_N
        """
        t = track.reset_index(drop=True).copy()

        # initialization from first two rows
        def _nz(v, fallback=0.0):
            return float(v) if np.isfinite(v) else float(fallback)

        vE0 = _nz(t["vE"].iloc[1] if len(t) > 1 else np.nan, 0.0)
        vN0 = _nz(t["vN"].iloc[1] if len(t) > 1 else np.nan, 0.0)
        vU0 = _nz(t["vU"].iloc[1] if len(t) > 1 else np.nan, 0.0)
        om0 = _nz(t["turn_rate"].iloc[1] if len(t) > 1 else np.nan, 0.0)

        x = np.array([t["E"].iloc[0], t["N"].iloc[0], t["U"].iloc[0],
                      vE0, vN0, vU0, om0], dtype=float)
        P = np.eye(7) * 100.0

        H = self.H_matrix()
        R = self.R_meas()
        out = []

        for i in range(len(t)):
            if i > 0:
                dt = float(t["dt"].iloc[i]) if np.isfinite(t["dt"].iloc[i]) else 0.0
                if dt <= 0:
                    continue

                if self.use_wind and ("wind_E" in t.columns) and ("wind_N" in t.columns):
                    wE = float(t["wind_E"].iloc[i]) if np.isfinite(t["wind_E"].iloc[i]) else 0.0
                    wN = float(t["wind_N"].iloc[i]) if np.isfinite(t["wind_N"].iloc[i]) else 0.0
                    F = self._F_wind(x, dt, wE, wN)
                    x = self._f_step_wind(x, dt, wE, wN)
                else:
                    F = self._F_no_wind(x, dt)
                    x = self._f_step_no_wind(x, dt)

                Q = self.Q_process(dt)
                P = F @ P @ F.T + Q

            # measurement update with position
            z = t.loc[i, ["E","N","U"]].to_numpy(dtype=float)
            y = z - (H @ x)
            S = H @ P @ H.T + R
            K = P @ H.T @ np.linalg.inv(S)
            x = x + K @ y
            P = (np.eye(7) - K @ H) @ P

            out.append((t["time"].iloc[i], *x))

        cols = ["time","E_hat","N_hat","U_hat","vE_hat","vN_hat","vU_hat","omega_hat"]
        return pd.DataFrame(out, columns=cols)

    # --- roll predictions from any filtered state ---
    def forecast_state(self, x: np.ndarray, horizon_s: int,
                       step_s: float = 1.0,
                       wind: Optional[Tuple[float,float]] = None) -> np.ndarray:
        """
        Roll dynamics forward without updates for 'horizon_s' seconds.
        If use_wind=True, pass wind=(wE,wN) to apply constant wind in forecast.
        """
        steps = int(np.ceil(horizon_s / step_s))
        dt_last = horizon_s - (steps-1)*step_s if steps > 0 else 0.0
        x2 = x.copy()

        for _ in range(max(steps-1, 0)):
            if self.use_wind and wind is not None:
                x2 = self._f_step_wind(x2, step_s, wind[0], wind[1])
            else:
                x2 = self._f_step_no_wind(x2, step_s)
        if steps > 0:
            dt = dt_last if dt_last > 0 else step_s
            if self.use_wind and wind is not None:
                x2 = self._f_step_wind(x2, dt, wind[0], wind[1])
            else:
                x2 = self._f_step_no_wind(x2, dt)
        return x2

    # --- convenience: compute horizon errors for one run ---
    def horizon_errors(self,
                       track: pd.DataFrame,
                       filt: pd.DataFrame,
                       horizons_s: Iterable[int],
                       tol_s: int = 5) -> Dict[int, pd.DataFrame]:
        """
        For each filtered row time t, predict to t+H and compare to nearest truth within ±tol_s.
        Returns {H: DataFrame(pred_time, E_pred, N_pred, U_pred, time_truth, E,N,U, errors...)}.
        """
        dfm = pd.merge(filt, track[["time","E","N","U"]], on="time", how="inner", suffixes=("_hat",""))
        results: Dict[int, pd.DataFrame] = {}

        # choose constant wind at forecast from the filtered state's current time, if available
        use_const_wind = self.use_wind and all(c in track.columns for c in ("wind_E","wind_N"))

        for H in horizons_s:
            preds = []
            for _, row in dfm.iterrows():
                x = np.array([row.E_hat, row.N_hat, row.U_hat, row.vE_hat, row.vN_hat, row.vU_hat, row.omega_hat], dtype=float)
                w = None
                if use_const_wind:
                    trow = track.loc[track["time"] == row["time"]]
                    if len(trow):
                        w = (float(trow["wind_E"].iloc[0]) if np.isfinite(trow["wind_E"].iloc[0]) else 0.0,
                             float(trow["wind_N"].iloc[0]) if np.isfinite(trow["wind_N"].iloc[0]) else 0.0)
                xH = self.forecast_state(x, H, step_s=1.0, wind=w)
                preds.append((row["time"] + pd.Timedelta(seconds=H), xH[0], xH[1], xH[2], row["time"]))
            pred_df = pd.DataFrame(preds, columns=["pred_time","E_pred","N_pred","U_pred","src_time"]).sort_values("pred_time")

            truth = track[["time","E","N","U"]].sort_values("time")
            joined = pd.merge_asof(pred_df, truth, left_on="pred_time", right_on="time",
                                   direction="nearest", tolerance=pd.Timedelta(seconds=tol_s))
            joined = joined.dropna(subset=["E","N","U"]).rename(columns={"time":"time_truth"})

            joined["eE"] = joined["E_pred"] - joined["E"]
            joined["eN"] = joined["N_pred"] - joined["N"]
            joined["eU"] = joined["U_pred"] - joined["U"]
            joined["err_m"] = np.sqrt(joined["eE"]**2 + joined["eN"]**2 + joined["eU"]**2)
            results[int(H)] = joined

        return results


# -----------------------------
# Simple RMSE helper
# -----------------------------
def rmse_from_results(results: Dict[int, pd.DataFrame]) -> pd.Series:
    out = {}
    for H, dfh in results.items():
        out[H] = float(np.sqrt(np.mean(dfh["err_m"]**2))) if len(dfh) else np.nan
    return pd.Series(out, name="RMSE_m")
