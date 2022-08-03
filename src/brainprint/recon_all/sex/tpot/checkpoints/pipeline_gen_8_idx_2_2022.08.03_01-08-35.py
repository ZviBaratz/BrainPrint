import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from tpot.builtins import OneHotEncoder
from tpot.export_utils import set_param_recursive

# NOTE: Make sure that the outcome column is labeled 'target' in the data file
tpot_data = pd.read_csv('PATH/TO/DATA/FILE', sep='COLUMN_SEPARATOR', dtype=np.float64)
features = tpot_data.drop('target', axis=1)
training_features, testing_features, training_target, testing_target = \
            train_test_split(features, tpot_data['target'], random_state=0)

# Average CV score on the training set was: 0.8603030303030303
exported_pipeline = make_pipeline(
    StandardScaler(),
    OneHotEncoder(minimum_fraction=0.2, sparse=False, threshold=10),
    LinearSVC(C=0.01, dual=True, loss="squared_hinge", penalty="l2", tol=0.01)
)
# Fix random state for all the steps in exported pipeline
set_param_recursive(exported_pipeline.steps, 'random_state', 0)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)