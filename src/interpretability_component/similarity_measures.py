import numpy as np
from sklearn.feature_selection import mutual_info_regression, mutual_info_classif
from sklearn.preprocessing import KBinsDiscretizer
from scipy.stats import pearsonr, wasserstein_distance, entropy
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import Lasso, LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, roc_auc_score
from sklearn.preprocessing import MultiLabelBinarizer


class SimilarityMeasure:
    """
    Methods that compute the similarity between each factor and each dimension with continuous values
    """

    def __init__(self, data, sim_function, num_bins=16):
        """
        initialize Similarity Measure with DataPreprocessor object and chosen measure function

        Args:
            data (DataPreprocessor): an object with meta_data, z and discrete_features
            sim_function (function): a similarity measure
            num_bins (int): number of bins for discretization
        """
        self.data = data
        self.sim_function = sim_function
        self.num_bins = num_bins
        self.scores, self.extra = self.compute_scores()

    def compute_scores(self):
        results = self.sim_function(self.data.meta_data, self.data.z, discrete_features=self.data.discrete_features,
                                    num_bins=self.num_bins)
        if type(results) == tuple:
            scores, extra = results
        else:
            scores = results
            extra = None
        return scores, extra

    def get_interpreted_features(self, method=None):
        if not method:
            method = get_top_features
        return method(self.scores, self.data.meta_data.columns)

    def get_interpretable_dims(self, method=None):
        if not method:
            method = get_top_dimension
        dims = method(self.scores)
        features = self.data.meta_data.columns
        return dict(zip(features, dims))


def get_top_features(scores, feature_labels):
    top_features = list(map(lambda x: feature_labels[np.argmax(x)], scores))
    return top_features


def get_top_feature_for_meaningful_dims(scores, feature_labels):
    top_features = get_top_features(scores, feature_labels)
    meaningful_dims = dict(zip(feature_labels, get_meaningful_dimensions(scores)))
    return list(map(lambda x: x[1] if x[0] in meaningful_dims[x[1]] else "no feature", enumerate(top_features)))


def get_top_dimension(scores):
    top_dimension = list(map(lambda x: np.argmax(x), scores.T))
    return top_dimension


def get_meaningful_dimensions(scores):
    half_max_score = np.max(scores) / 2
    meaningful_dims = []
    for factor_scores in scores.T:
        dims = list(np.where(factor_scores > half_max_score)[0])
        meaningful_dims.append(dims)
    return meaningful_dims


def _discretize_meta_and_z_data(meta_data, z, num_bins):
    discretizer = KBinsDiscretizer(n_bins=num_bins, encode='ordinal',
                                   strategy='uniform')  # strategy can be uniform, quantile, kmeans
    meta_data_bins = discretizer.fit_transform(meta_data)
    z_bins = discretizer.fit_transform(z)
    return meta_data_bins, z_bins


def _get_kwarg(key, args, default_value):
    if key in args:
        return args.get(key)
    else:
        return default_value


# INFORMATION-BASED SIMILARITY MEASURES #
def mutual_information_regression(meta_data, z, **kwargs):
    """
    compute matrix of mutual information between all generative factors and all latent dimensions,
    allowing continuous values for generative factors

    Args:
        meta_data: array of shape (N,F) including continuous values for each factor and each data sample
        z: array of shape (N,D) with the encoded, latent representation of all data samples
        kwargs: including discrete_features list of bool indicating which factors are discrete (True) or
            continuous (False)

    Returns:
        np.ndarray: array of shape (D,F) where each value corresponds to the mutual information between
                    the generative factor and the latent dimension
    """

    discrete_features = _get_kwarg('discrete_features', kwargs, False)

    num_dims = z.shape[1]
    num_factors = meta_data.shape[1]

    scores = np.ndarray(shape=(num_dims, num_factors))
    for dim in range(num_dims):
        # method computes MI for any combination of discrete and continuous vectors
        scores[dim] = (mutual_info_regression(meta_data,
                                              z[:, dim],
                                              discrete_features=discrete_features))
    return scores


def mutual_information_bins(meta_data, z, **kwargs):
    """
    compute matrix of mutual information between all generative factors and all latent dimensions,
    allowing continuous values for generative factors, but discretizing them with bins

    Args:
        meta_data: array of shape (N,F) including continuous values for each factor and each data sample
        z: array of shape (N,D) with the encoded, latent representation of all data samples
        kwargs: including num_bins, the number of bins in which the continuous values are discretized

    Returns:
        np.ndarray: array of shape (D,F) where each value corresponds to the mutual information between
                    the generative factor and the latent dimension
    """

    num_bins = _get_kwarg('num_bins', kwargs, 16)

    num_dims = z.shape[1]
    num_factors = meta_data.shape[1]

    meta_data_bins, z_bins = _discretize_meta_and_z_data(meta_data, z, num_bins)

    scores = np.ndarray(shape=(num_dims, num_factors))
    for dim in range(num_dims):
        # method computes MI for any combination of discrete vectors
        scores[dim] = mutual_info_classif(meta_data_bins, z_bins[:, dim])
    return scores


def kl_divergence_bins(meta_data, z, **kwargs):
    # inspired by: https://gist.github.com/rlangone/181ef3ae8187799f5ff842f86f52bc8d (18.10.2023)
    # inspired by: https://www.kaggle.com/code/nhan1212/some-statistical-distances (18.10.2023)
    """
    compute matrix of mutual information with KL divergence (computed with scipy.entropy (Shannon entropy)) between
    all generative factors and all latent dimensions, allowing continuous values for generative factors

    Args:
        meta_data: array of shape (N,F) including continuous values for each factor and each data sample
        z: array of shape (N,D) with the encoded, latent representation of all data samples
        kwargs: including num_bins, the number of bins in which the continuous values are discretized

    Returns:
        np.ndarray: array of shape (D,F) where each value corresponds to the mutual information between
                    the generative factor and the latent dimension

    """
    num_bins = _get_kwarg('num_bins', kwargs, 16)

    num_dims = z.shape[1]
    num_factors = meta_data.shape[1]

    meta_data_bins, z_bins = _discretize_meta_and_z_data(meta_data, z, num_bins)

    scores = np.ndarray(shape=(num_dims, num_factors))
    for dim in range(num_dims):
        for factor in range(num_factors):
            meta_vector = meta_data_bins[:, factor]
            z_vector = z_bins[:, dim] + np.finfo(float).eps  # no division by zero
            scores[dim][factor] = entropy(meta_vector, z_vector)
    return scores


def wasserstein_dist(meta_data, z, **kwargs):
    """
    compute matrix of mutual information with Wasserstein distance between all generative factors and all latent
    dimensions, allowing continuous values for generative factors

    Args:
        meta_data: array of shape (N,F) including continuous values for each factor and each data sample
        z: array of shape (N,D) with the encoded, latent representation of all data samples
        kwargs: can be empty

    Returns:
        np.ndarray: array of shape (D,F) where each value corresponds to the mutual information between
                    the generative factor and the latent dimension

    """

    num_dims = z.shape[1]
    num_factors = meta_data.shape[1]

    scores = np.ndarray(shape=(num_dims, num_factors))
    for dim in range(num_dims):
        for factor in range(num_factors):
            meta_vector = np.array(meta_data)[:, factor]
            normalized_meta_vector = meta_vector / np.linalg.norm(meta_vector)
            z_vector = z[:, dim]
            normalized_z_vector = z_vector / np.linalg.norm(z_vector)
            scores[dim][factor] = wasserstein_distance(normalized_meta_vector, normalized_z_vector)
    return scores


def jensen_shannon_divergence(meta_data, z, **kwargs):
    """
    compute matrix of mutual information with Jensen-Shannon divergence
    (computed with scipy.entropy (Shannon entropy)) between all generative factors and all latent dimensions,
    allowing continuous values for generative factors

    Args:
        meta_data: array of shape (N,F) including continuous values for each factor and each data sample
        z: array of shape (N,D) with the encoded, latent representation of all data samples
        kwargs: including num_bins, the number of bins in which the continuous values are discretized

    Returns:
        np.ndarray: array of shape (D,F) where each value corresponds to the mutual information between
                    the generative factor and the latent dimension

    """
    num_bins = _get_kwarg('num_bins', kwargs, 16)

    num_dims = z.shape[1]
    num_factors = meta_data.shape[1]

    meta_data_bins, z_bins = _discretize_meta_and_z_data(meta_data, z, num_bins)

    scores = np.ndarray(shape=(num_dims, num_factors))
    for dim in range(num_dims):
        for factor in range(num_factors):
            meta_vector = meta_data_bins[:, factor]
            z_vector = z_bins[:, dim] + np.finfo(float).eps  # no division by zero
            scores[dim][factor] = 0.5 * entropy(meta_vector, (meta_vector + z_vector) / 2) + \
                                  0.5 * entropy(z_vector, (meta_vector + z_vector) / 2)
    return scores


def total_variation_distance(): ...


# might not implement, because also f-divergence and holds same information as kl-divergence and js-divergence


def pearson_correlation(meta_data, z, **kwargs):
    """
    Pearson's correlation measures the linear relationship of 2 variables with Gaussian distribution

    Args:
        meta_data: array of shape (N,F) including continuous values for each factor and each data sample
        z: array of shape (N,D) with the encoded, latent representation of all data samples
        kwargs: can be empty

    Returns:
        np.ndarray: array of shape (D,F) where each value corresponds to the Pearson's correlation between
                    the generative factor and the latent dimension
    """
    num_dims = z.shape[1]
    num_factors = meta_data.shape[1]

    scores = np.ndarray(shape=(num_dims, num_factors))
    for dim in range(num_dims):
        for factor in range(num_factors):
            p_obj = pearsonr(np.array(meta_data)[:, factor], z[:, dim])
            scores[dim][factor] = p_obj.statistic
    return scores


# PREDICTION-BASED SIMILARITY MEASURES #
def random_forest_regressor(meta_data, z, **kwargs):  # TODO: improve grid search
    """
    computes the relative importance of each latent dimension for a generative factor with a Random Forest Regressor

    Args:
        meta_data: array of shape (N,F) including continuous values for each factor and each data sample
        z: array of shape (N,D) with the encoded, latent representation of all data samples
        kwargs: can be empty

    Returns:
        np.ndarray: array of shape (D,F) where each value corresponds to the relative importance between
                        the generative factor and the latent dimension
    """
    # https: // scikit - learn.org / stable / modules / generated / sklearn.ensemble.RandomForestRegressor.html
    # inspired by: https://github.com/ubisoft/ubisoft-laforge-disentanglement-metrics/blob/main/src/metrics/dci.py#L159
    max_depth = [8, 16, 64, 128]
    max_features = [0.3, 0.6]  # default 1
    n_estimators = 100  # default:100
    # random forest do not overfit see: https://www.stat.berkeley.edu/~breiman/RandomForests/cc_home.htm#remarks

    num_dims = z.shape[1]
    num_factors = meta_data.shape[1]

    scores = np.ndarray(shape=(num_factors, num_dims))
    for factor in range(num_factors):
        best_mse = np.inf
        best_depth = 0
        best_features = 0

        for depth in max_depth:
            for feature in max_features:
                forest = RandomForestRegressor(n_estimators=n_estimators, max_depth=depth,
                                               max_features=feature)
                # https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_val_score.html
                mse = cross_val_score(forest, np.array(z), np.array(meta_data)[:, factor], cv=10,
                                      scoring='neg_mean_squared_error')
                mse = -mse.mean()

                if mse < best_mse:
                    best_mse = mse
                    best_depth = depth
                    best_features = feature

        print("Factor " + str(factor) + ", best depth: " + str(best_depth) + ", best feature: " + str(best_features))
        forest = RandomForestRegressor(n_estimators=n_estimators, max_depth=best_depth, max_features=best_features)
        forest.fit(z, np.array(meta_data)[:, factor])
        scores[factor] = forest.feature_importances_

        y_pred = forest.predict(z)
        mse = mean_squared_error(y_pred, np.array(meta_data)[:, factor])
    return scores.T, mse


def lasso_regressor(meta_data, z, **kwargs):  # TODO: improve grid search
    """
    computes the relative importance of each latent dimension for a generative factor with a Lasso Regressor

    Args:
        meta_data: array of shape (N,F) including continuous values for each factor and each data sample
        z: array of shape (N,D) with the encoded, latent representation of all data samples
        kwargs: can be empty

    Returns:
        np.ndarray: array of shape (D,F) where each value corresponds to the relative importance between
                            the generative factor and the latent dimension
        int: mse
    """
    alphas = [0.4, 0.8, 1]

    num_dims = z.shape[1]
    num_factors = meta_data.shape[1]

    scores = np.ndarray(shape=(num_factors, num_dims))
    for factor in range(num_factors):
        best_mse = np.inf
        best_alpha = 0

        for alpha in alphas:
            regressor = Lasso(alpha=alpha, max_iter=100000)  # max_iter because of ConvergenceWarning
            mse = cross_val_score(regressor, np.array(z), np.array(meta_data)[:, factor], cv=10,
                                  scoring='neg_mean_squared_error')
            mse = -mse.mean()

            if mse < best_mse:
                best_mse = mse
                best_alpha = alpha

        print("Factor " + str(factor) + ", best alpha: " + str(best_alpha))
        regressor = Lasso(alpha=best_alpha, max_iter=10000)
        regressor.fit(z, np.array(meta_data)[:, factor])
        scores[factor] = np.abs(regressor.coef_)

        y_pred = regressor.predict(z)
        mse = mean_squared_error(y_pred, np.array(meta_data)[:, factor])
    return scores.T, mse


def linear_regressor(meta_data, z, **kwargs):
    """
    computes the relative importance of each latent dimension for a generative factor with a Linear Regressor

    Args:
        meta_data: array of shape (N,F) including continuous values for each factor and each data sample
        z: array of shape (N,D) with the encoded, latent representation of all data samples
        kwargs: can be empty

    Returns:
        np.ndarray: array of shape (D,F) where each value corresponds to the relative importance between
                            the generative factor and the latent dimension
    """
    num_dims = z.shape[1]
    num_factors = meta_data.shape[1]

    scores = np.ndarray(shape=(num_dims, num_factors))
    for dim in range(num_dims):
        for factor in range(num_factors):
            regressor = LinearRegression()
            regressor.fit(np.array(z[:, dim]).reshape(-1, 1), np.array(meta_data)[:, factor])

            score = regressor.score(np.array(z[:, dim]).reshape(-1, 1), np.array(meta_data)[:, factor])
            scores[dim][factor] = score  # can be negative
    return scores


def logistic_regressor(meta_data, z, **kwargs):
    """
    computes the relative importance of each latent dimension for a generative factor with a Logistic Regressor

    Args:
        meta_data: array of shape (N,F) including continuous values for each factor and each data sample
        z: array of shape (N,D) with the encoded, latent representation of all data samples
        kwargs: including num_bins, the number of bins in which the continuous values are discretized

    Returns:
        np.ndarray: array of shape (D,F) where each value corresponds to the relative importance between
                            the generative factor and the latent dimension
        int: average auc roc over all classes for all factors
    """
    num_bins = _get_kwarg('num_bins', kwargs, 16)

    num_dims = z.shape[1]
    num_factors = meta_data.shape[1]

    meta_data_bins, z_bins = _discretize_meta_and_z_data(meta_data, z, num_bins)

    auc = 0
    scores = np.ndarray(shape=(num_dims, num_factors))
    for dim in range(num_dims):
        for factor in range(num_factors):
            regressor = LogisticRegression(class_weight='balanced', multi_class='multinomial')
            regressor.fit(np.array(z[:, dim]).reshape(-1, 1), np.array(meta_data_bins)[:, factor])

            score = regressor.score(np.array(z[:, dim]).reshape(-1, 1), np.array(meta_data_bins)[:, factor])
            scores[dim][factor] = score

            y_pred = regressor.predict_proba(np.array(z[:, dim]).reshape(-1, 1))

            y_true = MultiLabelBinarizer().fit_transform(np.expand_dims(np.array(meta_data_bins)[:, factor], 1))
            auc += roc_auc_score(y_true, y_pred, multi_class='multinomial') / (num_factors * num_dims)
    return scores, auc

# INTERVENTION-BASED SIMILARITY MEASURES #
# not relevant for task because they do not have a scoring matrix for dimensions and factors
