from brainprint.atlas import Atlas
from brainprint.protocol import Protocol
from brainprint.recon_all.execution_configuration import ExecutionConfiguration
from brainprint.recon_all.results import ReconAllResults
from brainprint.recon_all.sex.model_fitting import EstimatorSearch
from sklearn.model_selection import RepeatedStratifiedKFold


def main():
    results = ReconAllResults(
        atlas=Atlas.DESTRIEUX,
        protocol=Protocol.BASE,
        completed_only=True,
        multi_only=False,
        questionnaire_only=False,
        configuration=[
            ExecutionConfiguration.MPRAGE_AND_3T_AND_T2,
            ExecutionConfiguration.T2,
            ExecutionConfiguration.DEFAULT,
            ExecutionConfiguration.FLAIR,
            ExecutionConfiguration.MPRAGE_AND_3T_AND_FLAIR,
        ],
    )
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=0)
    estimator_search = EstimatorSearch(
        results, target="Sex", cv=cv, random_state=0, scoring="roc_auc"
    )
    estimator_search.run()


if __name__ == "__main__":
    main()
