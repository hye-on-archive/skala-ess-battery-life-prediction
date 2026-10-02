# 데이터

## 원본 데이터

- Kaggle : https://www.kaggle.com/datasets/itshpark/data-driven-prediction-of-battery-cycle
- 원출처 : https://data.matr.io/1/ (Severson et al., *Nature Energy* 2019). 이용 조건은 원출처 페이지를 따릅니다.

원본 파일은 용량이 커서(약 8GB) 저장소에 넣지 않았습니다. 사용하는 파일은 아래 세 개입니다.

| 배치 | 파일 | 크기 | 용도 |
|---|---|---|---|
| Batch 1 | `2017-05-12_batchdata_updated_struct_errorcorrect.mat` | 2.8GB | 학습 |
| Batch 2 | `2018-02-20_batchdata_updated_struct_errorcorrect.mat` | 1.9GB | 필수 평가 |
| Batch 3 | `2018-04-12_batchdata_updated_struct_errorcorrect.mat` | 3.0GB | 추가 평가 |

`2018-04-03_varcharge_batchdata_updated_struct_errorcorrect.mat`은 다른 실험이라 과제 지침대로 쓰지 않습니다.

## 중간 파일 다시 만들기

세 파일을 한 폴더에 두고 실행합니다(예: `data/raw/`, 이 폴더는 `.gitignore`에 포함).

```bash
python src/preprocess.py --data-dir data/raw
```

## processed/ 내용

| 파일 | 내용 |
|---|---|
| `cells_meta.csv` | 셀 1개 = 1행. `cell_uid`(b{배치}c{번호}), 제공 수명 `cycle_life_provided`, 충전 정책, 기록 길이, 마지막 용량 |
| `summary.csv.gz` | 사이클별 QD·QC(Ah), IR(Ω), Tmax·Tavg·Tmin(°C), chargetime(분). 원본 값을 그대로 보존 |
| `qdlin.npz` | `voltage`(3.5→2.0V, 1,000점, 139셀 공통) + `{cell_uid}_q2`, `_q10`, `_q100` (Ah) |
| `features.csv` | `features.py` 결과. 셀 단위 피처, 정제 라벨 `cycle_life`, 모델 사용 여부 `in_model`, 제외 사유 |

`features.csv`의 `cycle_life`, `cycle_life_provided`, `end_QD`, `last_cycle`, `n_cycles_recorded`는 미래 정보이므로 입력 피처로 쓰지 않습니다.
