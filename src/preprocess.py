"""원본 MAT(HDF5) 파일 → 모델링용 중간 파일.

원본 파일은 읽기만 하고 수정하지 않는다. 필요한 값만 골라 data/processed/ 에 저장한다.

출력
- cells_meta.csv : 셀 1개 = 1행. 제공 수명 라벨, 충전 정책, 기록 길이, 마지막 용량 등
- summary.csv.gz : 사이클별 요약값 (QD, QC, IR, Tmax, Tavg, Tmin, chargetime)
- qdlin.npz      : 공통 전압축과 셀별 사이클 2·10·100의 Qdlin(전압축 보간 방전 용량)
- manifest.json  : 사용한 원본 파일 정보

사용법
    python src/preprocess.py --data-dir <MAT 파일 폴더> --out data/processed
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from config import BATCH_FILES, CURVE_CYCLES, PROCESSED_DIR

SUMMARY_FIELDS = {"QDischarge": "QD", "QCharge": "QC", "IR": "IR", "Tmax": "Tmax",
                  "Tavg": "Tavg", "Tmin": "Tmin", "chargetime": "chargetime"}


def _arr(obj) -> np.ndarray:
    return np.asarray(obj).ravel()


def _deref(f, ds, i):
    """MATLAB struct 배열의 i번째 참조를 따라간다."""
    import h5py
    ref = ds[i, 0] if ds.shape[1] == 1 else ds[0, i]
    return f[ref] if h5py.check_dtype(ref=ds.dtype) else ref


def _text(obj) -> str:
    vals = _arr(obj)
    return "".join(chr(int(v)) for v in vals if v)


def extract(data_dir: Path, out: Path) -> None:
    import h5py

    out.mkdir(parents=True, exist_ok=True)
    meta, frames, curves, manifest = [], [], {}, []
    voltage_ref = None

    for batch, filename in BATCH_FILES.items():
        path = Path(data_dir) / filename
        manifest.append({"batch": batch, "file": filename, "bytes": path.stat().st_size})
        with h5py.File(path, "r") as f:
            b = f["batch"]
            n = b["summary"].size
            print(f"{batch}: {n} cells", flush=True)
            for i in range(n):
                uid = f"b{batch[-1]}c{i:03d}"
                sm = _deref(f, b["summary"], i)
                cy = _deref(f, b["cycles"], i)
                cyc = _arr(sm["cycle"]).astype(float)
                qd = _arr(sm["QDischarge"]).astype(float)
                assert len(cyc) == len(qd), uid
                assert np.all(np.diff(cyc) > 0), f"{uid}: cycle 번호가 단조 증가하지 않음"

                frame = pd.DataFrame({"cell_uid": uid, "batch": batch, "cycle": cyc.astype(int)})
                for src, dst in SUMMARY_FIELDS.items():
                    frame[dst] = _arr(sm[src]).astype(float)
                frames.append(frame)

                n_curves = cy["Qdlin"].size
                aligned = n_curves == len(cyc)
                voltage = _arr(_deref(f, b["Vdlin"], i)).astype(float)
                if voltage_ref is None:
                    voltage_ref = voltage
                same_axis = bool(np.allclose(voltage, voltage_ref))
                for c in CURVE_CYCLES:
                    idx = np.where(cyc == c)[0]
                    if aligned and len(idx) == 1:
                        curves[f"{uid}_q{c}"] = _arr(_deref(f, cy["Qdlin"], int(idx[0]))).astype(float)

                meta.append({
                    "cell_uid": uid, "batch": batch, "source_index": i,
                    "cycle_life_provided": float(_arr(_deref(f, b["cycle_life"], i))[0]),
                    "policy": _text(_deref(f, b["policy_readable"], i)),
                    "n_cycles_recorded": len(cyc), "last_cycle": int(cyc[-1]),
                    "end_QD": float(qd[-1]),
                    "curve_summary_aligned": aligned, "same_voltage_axis": same_axis,
                })
        print(f"{batch}: done", flush=True)

    cells = pd.DataFrame(meta)
    assert cells.cell_uid.is_unique
    cells.to_csv(out / "cells_meta.csv", index=False)
    pd.concat(frames, ignore_index=True).to_csv(out / "summary.csv.gz", index=False, compression="gzip")
    np.savez_compressed(out / "qdlin.npz", voltage=voltage_ref, **curves)
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    print(f"saved {len(cells)} cells → {out}", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", required=True, help="원본 .mat 파일이 있는 폴더")
    p.add_argument("--out", default=str(PROCESSED_DIR))
    a = p.parse_args()
    extract(Path(a.data_dir), Path(a.out))
