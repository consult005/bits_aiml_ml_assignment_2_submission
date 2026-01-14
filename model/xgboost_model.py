from xgboost import XGBClassifier
from datapreperation.config import RANDOM_STATE

def build_model():
    return XGBClassifier(
        n_estimators=160,        # limiting due streamlit CPU limits contrainst
        max_depth=4,
        learning_rate=0.08,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        n_jobs=1,
        eval_metric="logloss",
        random_state=RANDOM_STATE
    )
