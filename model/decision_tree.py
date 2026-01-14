from sklearn.tree import DecisionTreeClassifier
from datapreperation.config import RANDOM_STATE

def build_model():
    return DecisionTreeClassifier(random_state=RANDOM_STATE)