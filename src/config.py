"""경로, 데이터 규칙, 실험 설정을 한곳에 모은다."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT / "data" / "processed"
RESULTS_DIR = ROOT / "results"
FIG_DIR = RESULTS_DIR / "figures"

BATCH_FILES = {
    "Batch 1": "2017-05-12_batchdata_updated_struct_errorcorrect.mat",  # 학습
    "Batch 2": "2018-02-20_batchdata_updated_struct_errorcorrect.mat",  # 필수 평가
    "Batch 3": "2018-04-12_batchdata_updated_struct_errorcorrect.mat",  # 추가 평가
}
CURVE_CYCLES = (2, 10, 100)  # Qdlin을 저장할 사이클
FEATURE_MAX_CYCLE = 100      # 예측에 쓰는 마지막 사이클
EOL_AH = 0.88                # 공칭 1.1Ah × 80%

# 원논문 공식 코드(Load Data.ipynb, LoadData.m)의 셀 정제 규칙
# github.com/rdbraatz/data-driven-prediction-of-battery-cycle-life-before-capacity-degradation
CONTINUED_CELLS = {  # 2017-06-30 파일에서 이어 측정한 사이클 수 → 수명에 더함
    "b1c000": 662, "b1c001": 981, "b1c002": 1060, "b1c003": 208, "b1c004": 482,
}
NOT_FINISHED_CELLS = ["b1c008", "b1c010", "b1c012", "b1c013", "b1c022"]  # 80% 미도달
NOISY_B3_CELLS = ["b3c002", "b3c023", "b3c032", "b3c037", "b3c042", "b3c043"]

# 검증 설정
SEED = 42
N_HOLDOUT_POLICIES = 5   # Batch 1 Hold-out 정책 수
N_FOLDS = 5
N_REPEATS = 5
TARGET_MAPE = 9.1        # 원논문 회귀 목표 (%)
