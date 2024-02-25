import numpy as np


# INFORMATION-BASED DISENTANGLEMENT METRICS #
class DisentanglementMetric:
    """
    Class that implements information-based disentanglement metrics
    """

    def __init__(self, sim_measure, dis_metric):
        """

        Args:
            sim_measure (SimilarityMeasure): an object containing DataPreprocessor data, a similarity measure
                and num_bins
            dis_metric (function): a disentanglement metric
            mi_scores (np.ndarray): array of shape (D,F) where each value corresponds to the mutual information between
                the generative factor and the latent dimension
        """
        self.sim_measure = sim_measure
        self.dis_metric = dis_metric
        self.mi_scores = sim_measure.scores

    def compute_score(self):
        return self.dis_metric(self)

    def get_meta_data_entropies(self):
        return get_entropies(self.sim_measure.data.meta_data.values,
                             self.sim_measure.sim_function,
                             discrete_features=self.sim_measure.data.discrete_features,
                             num_bins=self.sim_measure.num_bins)

    def get_z_entropies(self):
        return get_entropies(self.sim_measure.data.z,
                             self.sim_measure.sim_function,
                             num_bins=self.sim_measure.num_bins)


def get_entropies(matrix, sim_function, discrete_features=False, num_bins=16):
    """
    compute entropy for features of meta_data
    exploit that the mutual of a random variable with itself is its entropy: I(X;X)=H(X)
    (Cover, Thomas M., and Joy A. Thomas. "Entropy, relative entropy and mutual information."
    Elements of information theory 2.1 (1991): 12-13.)

    Args:
        matrix (): array of shape (N,F) including continuous values for each factor and each data sample
        sim_function (function): a similarity measure
        discrete_features (): list of bool indicating which factors are discrete (True) or continuous (False)
        num_bins (int): the number of bins in which the continuous values are discretized

    Returns:
        np.ndarray: array of shape (F,) where each value corresponds to the factor's entropy

    """
    mi_scores = sim_function(matrix, matrix, discrete_features=discrete_features, num_bins=num_bins)
    entropies = np.diag(mi_scores)
    return entropies


def mutual_information_gap(dis_metric_object):
    """
    compute the Mutual Information Gap from a given matrix of Mutual Information and Entropies
    MIG = sum(1 / H(v_i) * I(v_i, z*) - I(v_i,z°))

    Args:
        dis_metric_object (DisentanglementMetrics): a DisentanglementMetricsInfo object

    Returns:
        float: a single MIG value (good value: high)

    """
    mi_scores = dis_metric_object.mi_scores
    entropies = dis_metric_object.get_meta_data_entropies()

    num_factors = mi_scores.shape[1]

    gap = 0
    for factor in range(num_factors):
        mi_f_sorted = np.flip(np.sort(mi_scores[:, factor]))
        gap += (mi_f_sorted[0] - mi_f_sorted[1]) / entropies[factor]
    gap /= num_factors

    return gap


def rmig(dis_metric_object):
    """
    compute the Robust Mutual Information Gap (RMIG) from a given matrix of Mutual Information
    RMIG = sum(I(v_i, z*) - I(v_i,z°))

     Args:
        dis_metric_object (DisentanglementMetrics): a DisentanglementMetricsInfo object

    Returns:
        float: a single MIG value (good value: high)

    """
    mi_scores = dis_metric_object.mi_scores

    num_factors = mi_scores.shape[1]

    gap = 0
    for factor in range(num_factors):
        mi_f_sorted = np.flip(np.sort(mi_scores[:, factor]))
        gap += (mi_f_sorted[0] - mi_f_sorted[1])
    gap /= num_factors

    return gap


def jemmig(dis_metric_object):
    """
    compute the Joint Entropy Minus Mutual Information Gap (JEMMIG) from a given matrix of Mutual Information
    JEMMIG = sum(H(v_i, z*) - I(v_i, z*) + I(v_i,z°))

    Args:
        dis_metric_object (DisentanglementMetrics): a DisentanglementMetricsInfo object

    Returns:
        float: a single MIG value (good value: low)

    """
    mi_scores = dis_metric_object.mi_scores
    entropies_meta_data = dis_metric_object.get_meta_data_entropies()
    entropies_z = dis_metric_object.get_z_entropies()

    num_factors = mi_scores.shape[1]

    gap = 0
    for factor in range(num_factors):
        mi_f = mi_scores[:, factor]
        z_star = np.argmax(mi_f)

        h_z = entropies_z[z_star]
        h_f = entropies_meta_data[factor]

        mi_f_sorted = np.flip(np.sort(mi_f))
        # use: I(X,Y) = H(X) + H(Y) - H(X,Y) -> H(X,Y) = H(X) + H(Y) - I(X,Y)
        gap += h_z + h_f - 2 * mi_f_sorted[0] + mi_f_sorted[1]
    gap /= num_factors

    return gap


def mig_sup(dis_metric_object):
    """
    compute the MIG-sup from a given matrix of Mutual Information
    MIG-sup = sum(I(z_i, v*) - I(z_i,v°))
    (note: Mutual Information is symmetric)

    Args:
        dis_metric_object (DisentanglementMetrics): a DisentanglementMetricsInfo object

    Returns:
        float: a single MIG value (good value: high)
    """
    mi_scores = dis_metric_object.mi_scores

    num_dims = mi_scores.shape[0]

    gap = 0
    for dim in range(num_dims):
        mi_dim_sorted = np.flip(np.sort(mi_scores[dim]))
        gap += mi_dim_sorted[0] - mi_dim_sorted[1]
    gap /= num_dims

    return gap


def modularity_score(dis_metric_object):
    """
    compute the Modularity Score from a given matrix of Mutual Information
    mod_score = sum(1 - (sigma_(all factors except v*) I(v_j, z_i)^2) / ( I(v*, z_j)^2 * (M-1) ))

    Args:
        dis_metric_object (DisentanglementMetrics): a DisentanglementMetricsInfo object

    Returns:
        float: a single MIG value (good value: high)

    """
    mi_scores = dis_metric_object.mi_scores

    num_dims = mi_scores.shape[0]
    num_factors = mi_scores.shape[1]

    gap = 0
    for dim in range(num_dims):
        mi_dim_sorted = np.flip(np.sort(mi_scores[dim]))
        mi_dim_sorted_squared = np.square(mi_dim_sorted)
        gap += 1 - (np.sum(mi_dim_sorted_squared[1:]) / (mi_dim_sorted_squared[0] * (num_factors - 1)))
    gap /= num_dims

    return gap


def dcimig(dis_metric_object):
    """
    compute the DCIMIG from a given matrix of Mutual Information
    DCIMIG = sum(S_i) / sum(H(v_i))
    (S_i for each factor i relate to the highest R_j values, the differences between the top2 factors
    for each dimension j)

    Args:
        dis_metric_object (DisentanglementMetrics): a DisentanglementMetricsInfo object

    Returns:
        float: a single MIG value (good value: high)

    """
    mi_scores = dis_metric_object.mi_scores

    num_dims = mi_scores.shape[0]
    num_factors = mi_scores.shape[1]

    max_factors = list(map(lambda x: np.argmax(x), mi_scores))

    R_gaps = [0] * num_dims
    for dim in range(num_dims):
        mi_dim_sorted = np.flip(np.sort(mi_scores[dim]))
        R_gaps[dim] = mi_dim_sorted[0] - mi_dim_sorted[1]

    S_gaps = [0] * num_factors
    for dim in range(num_dims):
        if R_gaps[dim] > S_gaps[max_factors[dim]]:
            S_gaps[max_factors[dim]] = R_gaps[dim]

    gap = np.sum(S_gaps) / np.sum(dis_metric_object.get_meta_data_entropies())

    return gap


# PREDICTION-BASED DISENTANGLEMENT METRICS #
def dci(dis_metric_object):
    """
    compute the DCI from a given matrix of importance weights computed with a Lasso or Random Forest Regressor
    D = 1/D * sum(rho_i * D_i) -> D_i = 1 + sum_i(p_ij * log(p_ij) with p_ij = R_ij / sum(R_kj)
    C = 1/M * sum(C_i) -> C_i = 1 + sum_j(p_ij * log(p_ij) with p_ij = R_ij / sum(R_ik)
    I = 1 - 6 * MSE

    Args:
        dis_metric_object (DisentanglementMetrics): a DisentanglementMetricsInfo object

    Returns:
        np.array: 3 values for disentanglement, completeness, and informativeness (good values: high)

    """
    r_scores = dis_metric_object.mi_scores
    mse = dis_metric_object.sim_measure.extra
    assert mse  # TODO

    num_dims = r_scores.shape[0]
    num_factors = r_scores.shape[1]

    # disentanglement
    scores_dim_sum = np.array([np.sum(x) for x in r_scores])

    probs = r_scores / scores_dim_sum[:, None] + np.finfo(float).eps  # avoid nan because of log(0)
    log_func = lambda x: x * np.log(x) / np.log(num_factors)
    probs = log_func(probs)
    dis = 1 + np.array([np.sum(x) for x in probs])

    rho = np.array([np.sum(x) for x in r_scores]) / np.sum(r_scores)
    disentanglement = 1 / num_dims * np.sum(rho * dis)

    # completeness
    scores_factor_sum = np.array([np.sum(x) for x in r_scores.T])

    probs = r_scores / scores_factor_sum[None, :] + np.finfo(float).eps  # avoid nan because of log(0)
    log_func = lambda x: x * np.log(x) / np.log(num_dims)
    probs = log_func(probs)
    comps = 1 + np.array([np.sum(x) for x in probs.T])
    completeness = 1 / num_factors * np.sum(comps)

    # informativeness
    informativeness = 1 - 6 * mse

    return disentanglement, completeness, informativeness


def sap(dis_metric_object):
    """
    compute the SAP from a given matrix of importance weights computed with a Linear Regressor
    SAP = 1/M * sum(S_i* - S_i°)

    Args:
        dis_metric_object (DisentanglementMetrics): a DisentanglementMetricsInfo object

    Returns:
        float: a single disentanglement value (good value: high)

    """
    s_scores = dis_metric_object.mi_scores

    num_factors = s_scores.shape[1]

    gap = 0
    for factor in range(num_factors):
        s_f_sorted = np.flip(np.sort(s_scores[:, factor]))
        gap += (s_f_sorted[0] - s_f_sorted[1])
    gap /= num_factors

    return gap


def explicitness_score(dis_metric_object):
    """
    compute the explicitness score based on the average AUC-ROC
    exp_score = ((1/(M*D) * sum(auc_ij)) - 0.5) * 2

    Args:
        dis_metric_object (DisentanglementMetrics): a DisentanglementMetricsInfo object

    Returns:
        float: a single disentanglement value (good value: high)

    """
    auc = dis_metric_object.sim_measure.extra  # average AUCROC
    assert auc  # TODO

    auc = (auc - 0.5) * 2  # scale AUCROC to be in [0,1]

    return auc

# INTERVENTION-BASED DISENTANGLEMENT METRICS #
# not relevant for task because they do not have a scoring matrix for dimensions and factors
