from brainprint.atlas import Atlas
from brainprint.protocol import Protocol
from brainprint.recon_all.execution_configuration import ExecutionConfiguration
from brainprint.recon_all.models.model_fitting import EstimatorSearch
from brainprint.recon_all.results import ReconAllResults
from sklearn.model_selection import KFold


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
    results.context["Age"] = results.context["Age (days)"].astype(int)
    cv = KFold(n_splits=5, random_state=0, shuffle=True)
    estimator_search = EstimatorSearch(
        results, target="Age", cv=cv, random_state=0, scoring="r2"
    )
    estimator_search.run()


if __name__ == "__main__":
    main()
