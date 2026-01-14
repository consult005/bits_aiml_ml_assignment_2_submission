from sklearn.ensemble import RandomForestClassifier
from datapreperation.config import RANDOM_STATE

def build_model():
     return RandomForestClassifier(
        n_estimators=120,       #  limiting due streamlit CPU limits contrainst
        max_depth=12,           # limiting due streamlit CPU limits contrainst
        min_samples_leaf=2,   # reduces overfitting + size
        max_features="sqrt",  
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
