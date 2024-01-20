# ----------------------------------------------------------------------------
# Written by Francisco Jose Manjon Cabeza Garcia for his Master Thesis at    |
# Technische Universität Berlin.                                             |
#                                                                            |
# This file contains the ClassicPortfolio class, which implements the        |
# classical (discrete) portfolio optimisation methods based on Markowitz'    |
# portfolio optimisation model as well as model parameter estimation methods.|
# ----------------------------------------------------------------------------

from .ConstantCorrelationModel import LedoitWolfCovShrink
from itertools import product
import numpy as np
import pandas as pd
from scipy.linalg import solve
from typing import Tuple


class ClassicalPortfolio:
    def __init__(self,
                 asset_returns_ts: pd.DataFrame | None = None,
                 asset_volumes_ts: pd.DataFrame | None = None,
                 number_of_assets: int | None = None):
        """Initialise a Portfolio class object. Either both parameters
        asset_returns_ts and asset_volumes_ts are passed or the parameter
        number_of_assets must be passed.

        Parameters
        ----------
        asset_returns_ts : pandas.DataFrame, default None
            pandas DataFrame with the asset tickers as rows and timestamps as
            columns containing the asset returns in each period.

        asset_volumes_ts : pandas.DataFrame, default None
            pandas DataFrame with the asset tickers as rows and timestamps as
            columns containing the asset dollar volumes in each period.

        number_of_assets : int, default None
            Number of assets in the portfolio.

        Returns
        -------
        None
        """

        self._AssetReturnsTS = asset_returns_ts
        self._AssetVolumesTS = asset_volumes_ts

        if self._AssetReturnsTS is not None:
            assert self._AssetVolumesTS is not None, \
                ("No asset volumes timeseries was provided,"
                 " although an asset returns timeseries was provided.")

            self._AssetReturnsTS = self._AssetReturnsTS.sort_index(axis=0)
            self._AssetVolumesTS = self._AssetVolumesTS.sort_index(axis=0)

            assert self._AssetReturnsTS.index.equals(
                self._AssetVolumesTS.index), \
                ("The row index of the asset returns timeseries and asset"
                 " volumes timeseries do not match.")

            assert self._AssetReturnsTS.columns.equals(
                self._AssetVolumesTS.columns), \
                ("The column index of the asset returns timeseries and asset"
                 " volumes timeseries do not match.")

            self.NumberOfAssetsInPortfolio = self._AssetReturnsTS.shape[0]
            self.NumberOfDataPeriods = self._AssetReturnsTS.shape[1]

        else:
            assert number_of_assets is not None, \
                ("No asset returns timeseries DataFrame was given. Hence, it"
                 " is necessary to specify the number of assets.")

            assert isinstance(number_of_assets, int), \
                "The number of assets given is not an integer."

            assert number_of_assets > 0, \
                "The number of assets given is not strictly greater than zero."

            self.NumberOfAssetsInPortfolio = number_of_assets
            self.NumberOfDataPeriods = 0

        return

    def GetAssets(self):
        if hasattr(self, "_AssetReturnsTS"):
            return self._AssetReturnsTS.index
        else:
            return None

    def GenerateRandomAssetReturnsTS(self,
                                     number_of_assets: int,
                                     number_of_obs: int) -> pd.DataFrame:
        """Generate random asset returns using the standard normal
        distribution and save them in the self._AssetReturnsTS class attribute.

        Parameters
        ----------
        number_of_assets : int
            Number of assets to simulate returns for.
        number_of_obs : int
            Number of returns observations to generate for each asset.

        Returns
        -------
        self._AssetReturnsTS : pandas.DataFrame
            pandas DataFrame with the simulated asset returns, where each row
            contains the returns of one asset for each time period (columns).
        """

        random_data = np.random.normal(size=(number_of_assets,
                                             number_of_obs))
        index = [f"Asset{i+1}" for i in range(number_of_assets)]
        self._AssetReturnsTS = pd.DataFrame(data=random_data,
                                            index=index)

        return self._AssetReturnsTS

    def ComputeExpectedAssetReturns(
            self,
            method="sample-mean",
            riskfree_rate: pd.Series | None = None,
            exp_mean_alpha: float | None = None,
            shrinkage_intensity: float | None = None,
            structured_estimator: pd.Series | None = None,
            timeframe_adj_factor: float = 1,
            cov_matrix: pd.DataFrame | None = None,
            market_pf: pd.Series | None = None,
            risk_aversion_param: float | None = None
            ) -> pd.Series:
        """Compute a vector of expected returns for the assets in the
        portfolio, i.e., for the assets in self._AssetReturnsTS, using the
        historical data contained in self._AssetReturnsTS. The flow of this
        method is as follows
        1. If the riskfree_rate parameter is not None, adjust the returns in
           self._AssetReturnsTS to excess returns by substracting the risk-free
           rate of the corresponding timestamp.
        2. Estimate the vector of expected returns from the historical data
           using the method defined in the parameter method.
        3. If the shrinkage_intensity parameter is not zero, shrink the
           estimation from 1. towards the structured estimator.
        4. If the timeframe_adj_factor is not 1, adjust the output from 2.
           using it.

        Parameters
        ----------
        method : str, default "sample-mean"
            Method used to compute the vector of expected returns. Currently
            the following methods are implemented:
            - "sample-mean", which computes the simple mean of in-sample
              returns.
            - "sample-exp-mean", which computes the exponentially weighted
              mean of in-sample returns with smoothing factor (alpha)
              exp_mean_alpha using pandas.ewm.
            - "sample-total-return", which computes the total return of the
              in-sample returns.
            - "black-litterman", which computes the market-implied expected
              returns from the product of the covariance matrix in parameter
              cov_matrix, the risk-aversion factor from parameter
              risk_aversion_param and the market portfolio in parameter
              market_pf.

        riskfree_rate : pandas.Series, default None
            If not None, this pandas Series must contain a risk-free rate for
            each timestamp in the return observations in self._AssetReturnsTS.
            The asset returns will be adjusted to excess returns by
            substracting the risk-free rate of the corresponding timestamp.
        exp_mean_alpha : float, default None
            If the method "sample-exp-mean" is selected, then this parameter
            will be passed to pandas.ewm as alpha attribute for the
            computation of the exponentially weighted mean of in-sample
            returns.
        shrinkage_intensity : float, default None
            If not equal to zero, the estimated expected returns vector is
            shrunk towards the structured estimator given by
            structured_estimator with this intensity, such that the final
            estimation is equal to:
            (shrinkage_factor * estimation
             + (1-shrinkage_factor) * structued_estimator)
        structured_estimator : pandas.Series, default None
            Structured estimator towards which to shrink the estimated
            expected returns vector if the shrinkage_intensity parameter is
            not zero. If set to None, the structured_estimator is defined as
            the mean of the estimated expected returns vector.
        timeframe_adj_factor : float, default 1
            If this factor is set to a value different from 1, the computed
            expected returns using the selected method are then adjusted to
            (1+exp_returns)**timeframe_adj_factor - 1.
            For example, if monthly returns are used to compute the vector of
            expected returns with the sample-mean method, one can set this
            parameter to 12 to get a vector of expected annualised returns.
        cov_matrix : pandas.DataFrame, default None
            If the "black-litterman" method is selected, this covariance
            matrix of asset returns is used for the computation of the
            market-implied expected returns vector.
        market_pf : pandas.Series, default None
            If the "black-litterman" method is selected, this parameter
            specifies the market portfolio used for the computation of the
            market-implied expected returns vector.
        risk_aversion_param : float, default None
            If the "black-litterman" method is selected, this is the
            investor's risk-aversion parameter used for the computation of the
            market-implied expected returns vector.

        Returns
        -------
        exp_returns : pandas.Series
            Vector of expected returns index by asset name as in
            self._AssetReturnsTS.
        """

        assert self._AssetReturnsTS is not None, \
            ("The portfolio asset returns timeseries data is empty.")

        if riskfree_rate is not None:
            riskfree_rate = riskfree_rate.sort_index()

            assert riskfree_rate.index.equals(self._AssetReturnsTS.columns), \
                ("The passed riskfree_rate pandas Series is not indexed the"
                 " same as the columns of the returns DataFrame.")

            # Substract the risk-free rate from the raw asset returns.
            returns = self._AssetReturnsTS - riskfree_rate

        else:
            # Do not substract any risk-free rate from the raw asset returns.
            returns = self._AssetReturnsTS

        if method == "sample-mean":
            # Compute the vector of expected returns as the in-sample mean
            # asset returns.
            exp_returns = returns.mean(axis=1)

        elif method == "sample-exp-mean":
            # Compute the vector of expected returns as the in-sample
            # exponentially weighted mean asset returns.
            exp_returns = returns.ewm(alpha=exp_mean_alpha, axis=1).mean()
            # The .ewm().mean() methods return a pandas DataFrame with all the
            # partial exponentially weighted means. Select only the last
            # elements to get the exponentially weighted means computed using
            # of the sample observations.
            exp_returns = exp_returns.iloc[:, -1]

        elif method == "sample-total-return":
            # Compute the vector of expected returns as the total in-sample
            # asset returns.
            exp_returns = (1 + returns).prod(axis=1) - 1

        elif method == "black-litterman":
            assert cov_matrix is not None, \
                ("To compute the expected return vector with the"
                 " black-litterman method a covariance matrix needs to be"
                 " passed in the paramter cov_matrix.")

            assert market_pf is not None, \
                ("To compute the expected return vector with the"
                 " black-litterman method a market portfolio needs to be"
                 " passed in the paramter market_pf.")

            assert risk_aversion_param is not None, \
                ("To compute the expected return vector with the"
                 " black-litterman method a risk-aversion parameter needs to "
                 " be passed in the paramter risk_aversion_param.")

            cov_matrix = cov_matrix.sort_index(axis=0)
            cov_matrix = cov_matrix.sort_index(axis=1)
            market_pf = market_pf.sort_index()

            index_match = (cov_matrix.index.equals(cov_matrix.columns)
                           and cov_matrix.index.equals(market_pf.index))

            assert index_match, \
                ("The indices of the covariance matrix and market portfolio"
                 " vector do not match.")

            # Compute the market-implied neutral view expected returns vector
            # as in the Black-Litterman model.
            exp_returns = cov_matrix.dot(market_pf)
            exp_returns = risk_aversion_param * exp_returns

        else:
            raise ValueError("The method selected is not implemented.")

        if shrinkage_intensity is not None:
            # The estimated returns need to be shrunk.

            if structured_estimator is None:
                # No structured estimator was passsed for shrinkage.
                # Compute the structured estimator as the mean of the
                # return estimations.
                structured_estimator = pd.Series(data=exp_returns.mean(),
                                                 index=exp_returns.index)

            else:
                # A structured estimator was passed.
                # Check that it can be used.
                structured_estimator = structured_estimator.sort_index()

                assert returns.index.equals(structured_estimator), \
                    ("The structured estimator passed is not indexed as the"
                     " asset returns DataFrame.")

            # Shrink the computed return estimations towards the structured
            # estimator according to the shrinkage intensity parameter.
            exp_returns = (shrinkage_intensity * exp_returns
                           + (1-shrinkage_intensity) * structured_estimator)

        if timeframe_adj_factor != 1:
            # Adjust the computed expected returns to a certain timeframe using
            # the timeframe adjustment factor.
            exp_returns = (1 + exp_returns).pow(timeframe_adj_factor) - 1

        exp_returns = exp_returns.sort_index()

        return exp_returns

    def ComputeAssetReturnCovarianceMatrix(
            self,
            method="sample",
            use_log_returns: bool = False,
            timeframe_adj_factor: float = 1,
            exp_alpha: float | None = None
            ) -> pd.DataFrame:
        """Compute covariance matrix of asset returns using the data and the
        assets in self._AssetReturnsTS.

        Parameters
        ----------
        method : str, default "sample"
            Method used to compute the covariance matrix of asset returns.
            Currently the following methods are implemented:
            - "sample", which computes the in-sample unbiased covariance
              matrix of asset returns.
            - "sample-exp", which computes the in-sample covariance matrix
              using exponential weighting.
            - "constant-correlation-model", which computes the sample
              covariance matrix and applies shrinkage towards the constant
              correlation matrix with the optimal shrinkage intensity as in
              Ledoit and Wolf 2003.

        use_log_returns : bool, default False
            If True, the input asset returns from self._AssetReturnsTS are
            transformed using the function log(1+r), where r is the return of
            an asset at some point in time. This is useful, in case we want to
            annualise an in-sample covariance matrix of asset returns.
        timeframe_adj_factor : float, default 1
            If this factor is set to a value different from 1, the computed
            covariance matrix of asset returns using the selected method are
            then adjusted to
            cov_matrix * timeframe_adj_factor.
            For example, if monthly returns are used to compute the covariance
            matrix of asset returns with the sample method, one can set this
            parameter to 12 to get an annualised covariance matrix of asset
            returns.
        exp_alpha : float, default None
            If the method "sample-exp" is selected, then this parameter is
            used as alpha attribute for the computation of the exponentially
            weighted mean of in-sample returns.

        Returns
        -------
        cov_matrix : pandas.DataFrame
            Covariance matrix of asset returns index by row and column using
            the asset names in the self._AssetReturnsTS index.
        """

        assert self._AssetReturnsTS is not None, \
            ("The portfolio asset returns timeseries data is empty.")

        if use_log_returns:
            # Use log(1+r) returns for the computations.
            sample_returns = self._AssetReturnsTS.apply(np.log1p)

        else:
            # Use the raw returns for the computations.
            sample_returns = self._AssetReturnsTS

        if method == "sample":
            cov_matrix = sample_returns.T.cov()

        elif method == "sample-exp":
            assert exp_alpha is not None, \
                ("To compute the sample covariance matrix with exponential"
                 " weighting, an exponential alpha parameter must be passed.")

            periods = sample_returns.shape[1]
            ewm_weights = (1 - exp_alpha) ** np.arange(periods)[::-1]
            mean_returns = sample_returns.mean(axis=1)
            normalised_returns = sample_returns.subtract(mean_returns, axis=0)
            cov_matrix = (ewm_weights * normalised_returns)
            cov_matrix = cov_matrix.dot(normalised_returns.T)
            cov_matrix = cov_matrix / ewm_weights.sum()

        elif method == "constant-correlation-model":
            cov_matrix = LedoitWolfCovShrink(Y=sample_returns.T)

        else:
            raise ValueError("The method selected is not implemented.")

        if timeframe_adj_factor != 1:
            # Adjust the computed covariance matrix of assets returns to a
            # certain timeframe using the timeframe adjustment factor.
            cov_matrix = timeframe_adj_factor * cov_matrix

        cov_matrix = cov_matrix.sort_index(axis=0)
        cov_matrix = cov_matrix.sort_index(axis=1)

        return cov_matrix

    def ComputeExpectedTransactionCosts(self,
                                        cm: pd.Series,
                                        bidask_spread_pct: pd.Series,
                                        thetas: pd.Series,
                                        abs_trade_size: float) -> pd.Series:
        """Compute the expected transactions costs in percentage using the
        formula:
        tc = cm + bidask_spread_pct + theta * abs_trade_size/expected_volume,
        where the expected_volume is computed using the mean of the volume
        timeseries in self._AssetVolumesTS.

        Parameters
        ----------
        cm : pandas.Series
            Fixed trade commissions in percent.
        bidask_spread_pct : pandas.Series
            Bid/ask spreads as percentage of current price.
        thetas : pandas.Series
            Multipliers for the market-impact factor in the formula, i.e., the
            trade size w.r.t. the average historical trade volume. This
            parameter is usually set to the volatility (standard deviation) of
            the asset return in the timeframe of the data. For example, to
            annual asset volatility divided by square root fo 250, or the
            asset daily volatility.
        abs_trade_size : float
            Trade size in units of currency, i.e., trade size as percentage of
            total wealth/portfolio value times the total wealth or portfolio
            value.

        Returns
        -------
        transaction_costs_pct : pandas.Series
            Vector with the transaction costs in percentage for each asset in
            the portfolio.
        """

        assert self._AssetVolumesTS is not None, \
            ("No volume timeseries in self._AssetVolumesTS")

        cm = cm.sort_index()
        bidask_spread_pct = bidask_spread_pct.sort_index()
        thetas = thetas.sort_index()

        index_match = (self._AssetVolumesTS.index.equals(cm.index)
                       and cm.index.equals(bidask_spread_pct.index)
                       and bidask_spread_pct.index.equals(thetas.index))

        assert index_match, \
            ("The indices of the asset volumes timeseries, fixed-commission"
             " vector, the bid-ask spread in percentage vector and the thetas"
             " vector do not match.")

        expected_volumes = self._AssetVolumesTS.mean(axis=1)
        trade_pct_of_exp_vol = (abs_trade_size/expected_volumes).apply(np.sqrt)
        market_impact = thetas.multiply(trade_pct_of_exp_vol)
        transaction_costs_pct = cm + bidask_spread_pct + market_impact

        transaction_costs_pct = transaction_costs_pct.sort_index()

        return transaction_costs_pct

    def ComputePortfolioObjectiveValue(
            self,
            portfolio: pd.Series,
            asset_manager_lambda: float,
            net_positions: int | None,
            cov_matrix: pd.DataFrame,
            exp_returns: pd.Series,
            transaction_costs: pd.Series,
            alpha: float,
            initial_portfolio: pd.Series) -> float:
        """Compute the portfolio objective function value with a
        soft-constraint for the cardinality constraint for a given portfolio.
        The objective function used is defined as:
        f(w) := asset_manager_lambda * pf_risk
                - (1-asset_manager_lambda) * (pf_gross_expected_return
                                              - pf_rebalancing_cost) +
                alpha * (count - net_positions)^2,

        where count is equal to the number of entries in the portfolio which
        are not zero and

        pf_risk := w^T * cov_matrix * w,

        pf_gross_expected_return := exp_returns^T * w,

        pf_rebalancing_cost := transaction_costs^T * abs(portfolio -
        initial_portfolio).

        Parameters
        ----------
        portfolio : pandas.Series
            Indexed vector containing the target portfolio asset weights for
            which to compute the objective function value.
        asset_manager_param : float
            Expresses the trade-off between risk and return. Setting it to 0
            maximises return only. Setting it to 1 minimises volatility only.
        net_positions : int
            If not None, only portfolios with exactly this number of positions,
            i.e., non-zero elements, will be considered feasible solutions in
            the soft-constraint.
        cov_matrix : pandas.DataFrame
            Covariance matrix of asset returns.
        exp_returns : pandas.Series
            Expected returns indexed vector.
        transaction_costs : pandas.Series
            Indexed vector containing the transaction costs of each asset in
            percentage terms.
        alpha : float
            Penalty parameter for the cardinality soft-constraint.
        initial_portfolio : pandas.Series
            Indexed vector containing the initial portfolio asset weights,
            which will be considered in the computation of transaction costs
            when rebalancing the portfolio from `initial_portfolio` to optimal
            portfolio.

        Returns
        -------
        pf_obj_func_val : float
            Portfolio objective function value with a soft-constraint for the
            cardinality constraint.
        """

        # Get the number of assets in the portfolio.
        N = self.NumberOfAssetsInPortfolio

        portfolio = portfolio.sort_index()
        cov_matrix = cov_matrix.sort_index(axis=0)
        cov_matrix = cov_matrix.sort_index(axis=1)
        exp_returns = exp_returns.sort_index()
        transaction_costs = transaction_costs.sort_index()
        initial_portfolio = initial_portfolio.sort_index()

        index_match = (
            portfolio.index.equals(cov_matrix.index)
            and cov_matrix.index.equals(cov_matrix.columns)
            and cov_matrix.index.equals(exp_returns.index)
            and exp_returns.index.equals(transaction_costs.index)
            and transaction_costs.index.equals(initial_portfolio.index)
            )

        assert index_match, \
            ("The indices of the portfolio, covariance matrix, expected"
             " returns vector, transaction costs vector and initial portfolio"
             " vector do not match.")

        assert portfolio.shape[0] == N, \
            ("The dimension of the portfolio vector does match the number of"
             " assets in the portfolio.")

        assert cov_matrix.shape == (N, N), \
            (f"The covariance matrix was expected to have shape ({N}, {N})"
             f" but it has shape {cov_matrix.shape}.")

        assert exp_returns.shape[0] == N, \
            ("The dimension of vector of expected returns does not match the"
             " number of assets in the portfolio.")

        assert transaction_costs.shape[0] == N, \
            ("The dimension of vector of transaction costs does not match the"
             " number of assets in the portfolio.")

        assert initial_portfolio.shape[0] == N, \
            ("The dimension of initial portfolio vector does not match the"
             " number of assets in the portfolio.")

        # Compute portfolio risk as total portfolio variance.
        pf_risk = portfolio.dot(cov_matrix.dot(portfolio))
        # Compute portfolio return as the total portfolio expected return.
        pf_gross_expected_return = exp_returns.dot(portfolio)

        # Compute the rebalancing cost to transform the initial portfolio into
        # the given target portfolio.
        pf_diff = (portfolio - initial_portfolio).abs()
        pf_rebalancing_cost = transaction_costs.dot(pf_diff)

        # Substract the rebalancing cost from the total portfolio gross
        # expected return to get the net expected return.
        pf_net_expected_return = (pf_gross_expected_return
                                  - pf_rebalancing_cost)

        if alpha != 0 and net_positions is not None:
            # Compute the value of the cardinality soft-constraint.
            card_soft_constr_val = (portfolio != 0)
            card_soft_constr_val = card_soft_constr_val.sum()
            card_soft_constr_val = card_soft_constr_val - net_positions
            card_soft_constr_val = alpha * card_soft_constr_val**2
        else:
            card_soft_constr_val = 0

        # Compute the total portfolio objective function value.
        pf_obj_func_val = (asset_manager_lambda * pf_risk
                           - (1-asset_manager_lambda) * pf_net_expected_return
                           + card_soft_constr_val)

        return pf_obj_func_val

    def ComputeEqualWeightedPortfolioObjectiveValueBounds(
            self,
            asset_manager_lambda: float,
            net_positions: int | None,
            cov_matrix: pd.DataFrame,
            exp_returns: pd.Series,
            transaction_costs: pd.Series,
            initial_portfolio: pd.Series
            ) -> Tuple[float, float]:
        """Compute the portfolio objective function value upper and lower
        bounds without soft-constraints. These correspond to the maximum and
        minimum estimations respectively of the following function:
        f(w) := asset_manager_lambda * pf_risk
                - (1-asset_manager_lambda) * (pf_gross_expected_return
                                              - pf_rebalancing_cost)

        where w is a vector with elements in
        {-1/net_positions, 0, 1/net_positions} and

        pf_risk := w^T * cov_matrix * w,

        pf_gross_expected_return := exp_returns^T * w,

        pf_rebalancing_cost := transaction_costs^T * abs(portfolio -
        initial_portfolio).

        Parameters
        ----------
        asset_manager_param : float
            Expresses the trade-off between risk and return. Setting it to 0
            maximises return only. Setting it to 1 minimises volatility only.
        net_positions : int
            Number of positions, i.e., non-zero elements, allowed in the
            portfolios.
        cov_matrix : pandas.DataFrame
            Covariance matrix of asset returns.
        exp_returns : pandas.Series
            Expected returns indexed vector.
        transaction_costs : pandas.Series
            Indexed vector containing the transaction costs of each asset in
            percentage terms.
        initial_portfolio : pandas.Series
            Indexed vector containing the initial portfolio asset weights,
            which will be considered in the computation of transaction costs
            when rebalancing the portfolio from `initial_portfolio` to optimal
            portfolio.

        Returns
        -------
        lower_bound : float
            Value lower than or equal to the minimum of the function f for a
            portfolio where each non-zero position is equally-weighted.
        upper_bound : float
            Value greater than or equal to the maximum of the function f for a
            portfolio where each non-zero position is equally-weighted.
        """

        # Get the number of assets in the portfolio.
        N = self.NumberOfAssetsInPortfolio

        cov_matrix = cov_matrix.sort_index(axis=0)
        cov_matrix = cov_matrix.sort_index(axis=1)
        exp_returns = exp_returns.sort_index()
        transaction_costs = transaction_costs.sort_index()
        initial_portfolio = initial_portfolio.sort_index()

        index_match = (
            cov_matrix.index.equals(cov_matrix.columns)
            and cov_matrix.index.equals(exp_returns.index)
            and exp_returns.index.equals(transaction_costs.index)
            and transaction_costs.index.equals(initial_portfolio.index)
            )

        assert index_match, \
            ("The indices of the portfolio, covariance matrix, expected"
             " returns vector, transaction costs vector and initial portfolio"
             " vector do not match.")

        assert cov_matrix.shape == (N, N), \
            (f"The covariance matrix was expected to have shape ({N}, {N})"
             f" but it has shape {cov_matrix.shape}.")

        assert exp_returns.shape[0] == N, \
            ("The dimension of vector of expected returns does not match the"
             " number of assets in the portfolio.")

        assert transaction_costs.shape[0] == N, \
            ("The dimension of vector of transaction costs does not match the"
             " number of assets in the portfolio.")

        assert initial_portfolio.shape[0] == N, \
            ("The dimension of initial portfolio vector does not match the"
             " number of assets in the portfolio.")

        # Compute the total portfolio risk lower bound, which is zero because
        # the covariance matrix is positive semidefinite, i.e., the scalar
        # product of any portfolio and the covariance matrix is greater than
        # or equal to 0.
        pf_risk_lower_bound = 0
        # Compute the total portfolio risk upper bound, which is equal to the
        # sum of all absolute valued entries in the covariance matrix divided
        # by the net number of positions. This value is greater than or equal
        # to the maximum total equal-weighted portfolio variance possible.
        pf_risk_upper_bound = (
            1/net_positions * np.sum(np.abs(cov_matrix.to_numpy()))
            )
        # Compute the total portfolio net return, i.e., after transaction
        # costs, lower bound, which corresponds to trading all assets in the
        # opposite direction of their expected returns sign, and incurring the
        # maximum transaction costs by having to rebalance all positions from
        # the opposite direction to the one mentioned.
        pf_net_return_lower_bound = (
            1/net_positions * np.sum(- np.abs(exp_returns))
            - transaction_costs.dot(initial_portfolio.abs() + 1/net_positions)
            )
        # Compute the total portfolio net return, i.e., after transaction
        # costs, upper bound, which corresponds to trading all assets in the
        # direction of their expected returns sign, and incurring zero
        # transaction costs.
        pf_net_return_upper_bound = (
            1/net_positions * np.sum(np.abs(exp_returns))
            )

        # Compute the equal-weighted portfolio objective function lower bound
        # as the convex combination of the total portfolio risk lower bound
        # (in the objective function with positive sign contribution) and the
        # total portfolio net return upper bound (in the objective function
        # with negative sign contribution) according to the
        # asset_manager_lambda parameter.
        lower_bound = (
            asset_manager_lambda * pf_risk_lower_bound
            - (1-asset_manager_lambda) * pf_net_return_upper_bound
            )
        # Compute the equal-weighted portfolio objective function upper bound
        # as the convex combination of the total portfolio risk upper bound
        # (in the objective function with positive sign contribution) and the
        # total portfolio net return lower bound (in the objective function
        # with negative sign contribution) according to the
        # asset_manager_lambda parameter.
        upper_bound = (
            asset_manager_lambda * pf_risk_upper_bound
            - (1-asset_manager_lambda) * pf_net_return_lower_bound
            )

        return (lower_bound, upper_bound)

    def ComputeMarkowitzPortfolio(self,
                                  asset_manager_param: float,
                                  cov_matrix: pd.DataFrame,
                                  expected_returns: pd.Series,
                                  normalise_solution: bool = False,
                                  verbose: bool = True) -> tuple[pd.Series,
                                                                 float]:
        """"Compute the portfolio which minimises the function
        f(w) := asset_manager_param * w^T * cov_matrix * w
                - (1-asset_manager_param) * expected_returns^T * w
        by solving the linear system Aw=b, where

        A = asset_manager_param * cov_matrix,

        b = (1 - asset_manager_param) * expected_returns,

        w is the solution, i.e., the portfolio weights vector.

        Parameters
        ----------
        asset_manager_param : float
            Expresses the trade-off between risk and return. Setting it to 0
            maximises return only. Setting it to 1 minimises volatility only.
        cov_matrix : pandas.DataFrame
            Covariance matrix of asset returns.
        expected_returns : pandas.Series
            Expected returns indexed vector.
        normalise_solutions : bool, default False
            If True the solution portfolio vector is normalised such that the
            sum of all allocations is equal to 1. If False, the solution
            portfolio vector is not normalised and the allocations can be
            leveraged.
        verbose : bool, default True
            If true, messages will be printed to notify if the covariance
            matrix is singular.

        Returns
        -------
        opt_sol : pandas.Series
            Indexed vector of optimal portfolio weights.
        min_cost : float
            Objective function value of the optimal portfolio weights.
        """

        # Because the solvers below are numpy solvers, they do not take into
        # account the indices of the covariance matrix and the expected
        # returns vector. Hence, we sort all those indices, such that we can
        # make sure all elements are handled correctly and we can index the
        # solution provided by the solvers, which is a numpy.array, which is
        # not indexed.
        cov_matrix = cov_matrix.sort_index(axis=0)
        cov_matrix = cov_matrix.sort_index(axis=1)
        expected_returns = expected_returns.sort_index()

        index_match = (cov_matrix.index.equals(cov_matrix.columns)
                       and cov_matrix.index.equals(expected_returns.index))

        assert index_match, \
            ("The indices of the covariance matrix and expected returns vector"
             " do not match.")

        # Get the number of assets in the portfolio.
        N = self.NumberOfAssetsInPortfolio

        assert cov_matrix.shape == (N, N), \
            (f"The covariance matrix was expected to have shape ({N}, {N})"
             f" but it has shape {cov_matrix.shape}.")

        assert expected_returns.shape[0] == N, \
            ("The dimension of vector of expected returns does not match the"
             " number of assets in the portfolio.")

        # Try to solve the linear system using the Cholesky decomposition of
        # the covariance matrix.
        try:
            opt_sol = solve(a=asset_manager_param * cov_matrix,
                            b=(1 - asset_manager_param) * expected_returns,
                            assume_a="pos")

        # If the Cholesky decomposition fails, the covariance matrix does not
        # have full rank.
        # Solve the linear system with the least-squares method.
        except ValueError:
            if verbose:
                print("The covariance matrix is singular.")
                print("Solving linear system with the least-squares method...")
            opt_sol, _, _, _ = np.linalg.lstsq(
                a=asset_manager_param * cov_matrix,
                b=(1 - asset_manager_param) * expected_returns
                )

        # Index optimal solution.
        opt_sol = pd.Series(opt_sol, index=expected_returns.index)

        # Compute the objective function value for the optimal portfolio
        # computed above.
        min_cost = (asset_manager_param * opt_sol.dot(cov_matrix.dot(opt_sol))
                    - (1-asset_manager_param) * expected_returns.dot(opt_sol))

        if normalise_solution:
            opt_sol = opt_sol / opt_sol.sum()

        return opt_sol, min_cost

    def ComputeOptimalEqualWeightMarkowitzPortfolioByBruteForce(
            self,
            asset_manager_param: float,
            net_positions: int | None,
            cov_matrix: pd.DataFrame,
            expected_returns: pd.Series,
            transaction_costs: pd.Series,
            initial_portfolio: pd.Series
            ) -> tuple[pd.Series, float]:
        """"Compute the portfolio which minimises the function

        f(w) := asset_manager_lambda * pf_risk
                - (1-asset_manager_lambda) * (pf_gross_expected_return
                                              - pf_rebalancing_cost) +
                alpha * (count - net_positions)^2,

        where count is equal to the number of entries in the portfolio which
        are not zero and

        pf_risk := w^T * cov_matrix * w,

        pf_gross_expected_return := exp_returns^T * w,

        pf_rebalancing_cost := transaction_costs^T * abs(portfolio
                                                         - initial_portfolio),

        and w is a vector with elements in
        {-1/net_positions, 0, 1/net_positions},
        by brute force, i.e., by computing the objective value of all possible
        portfolios w.

        Parameters
        ----------
        asset_manager_param : float
            Expresses the trade-off between risk and return. Setting it to 0
            maximises return only. Setting it to 1 minimises volatility only.
        net_positions : int
            If not None, only portfolios with exactly this number of positions,
            i.e., non-zero elements, will be considered feasible solutions.
        cov_matrix : pandas.DataFrame
            Covariance matrix of asset returns.
        expected_returns : pandas.Series
            Expected returns indexed vector.
        transaction_costs : pandas.Series
            Indexed vector containing the transaction costs of each asset in
            percentage terms.
        initial_portfolio : pandas.Series
            Indexed vector containing the initial portfolio asset weights,
            which will be considered in the computation of transaction costs
            when rebalancing the portfolio from initial_portfolio to optimal
            portfolio.

        Returns
        -------
        opt_sol : pandas.Series
            Indexed vector of optimal portfolio weights, where each vector
            element is in { -1/net_positions, 0, 1/net_positions}.
        min_cost : float
            Objective function value of the optimal portfolio weights.
        """

        cov_matrix = cov_matrix.sort_index(axis=0)
        cov_matrix = cov_matrix.sort_index(axis=1)
        expected_returns = expected_returns.sort_index()
        transaction_costs = transaction_costs.sort_index()
        initial_portfolio = initial_portfolio.sort_index()

        index_match = (
            cov_matrix.index.equals(cov_matrix.columns)
            and cov_matrix.index.equals(expected_returns.index)
            and expected_returns.index.equals(transaction_costs.index)
            and transaction_costs.index.equals(initial_portfolio.index)
            )

        assert index_match, \
            ("The indices of the covariance matrix, expected returns vector,"
             " transaction costs vector and initial portfolio vector do not"
             " match.")

        # Get the number of assets in the portfolio.
        N = self.NumberOfAssetsInPortfolio

        assert cov_matrix.shape == (N, N), \
            (f"The covariance matrix was expected to have shape ({N}, {N})"
             f" but it has shape {cov_matrix.shape}.")

        assert expected_returns.shape[0] == N, \
            ("The dimension of vector of expected returns does not match the"
             " number of assets in the portfolio.")

        assert transaction_costs.shape[0] == N, \
            ("The dimension of vector of transaction costs does not match the"
             " number of assets in the portfolio.")

        assert initial_portfolio.shape[0] == N, \
            ("The dimension of initial portfolio vector does not match the"
             " number of assets in the portfolio.")

        min_cost = np.inf
        opt_sol = pd.Series(data=0, index=expected_returns.index)

        search_space = product([-1, 0, 1], repeat=N)
        for element in search_space:
            solution = pd.Series(data=element, index=expected_returns.index)
            sol_cardinality = solution.abs().sum()

            if (net_positions is None or sol_cardinality == net_positions):
                # Compute the equal weight portfolio associated with solution.
                if net_positions is None:
                    portfolio = solution
                else:
                    portfolio = 1/net_positions * solution

                # We set the parameter alpha to zero, because we have already
                # excluded solutions where the number of positions is not
                # equal to net_positions.
                cost = self.ComputePortfolioObjectiveValue(
                                portfolio=portfolio,
                                asset_manager_lambda=asset_manager_param,
                                net_positions=net_positions,
                                cov_matrix=cov_matrix,
                                exp_returns=expected_returns,
                                transaction_costs=transaction_costs,
                                alpha=0,
                                initial_portfolio=initial_portfolio)

                if cost < min_cost:
                    min_cost = cost
                    opt_sol = portfolio

        return opt_sol, min_cost

    def GetDiscreteEqualWeightMarkowitzPortfolioQUBOMatrix(
            self,
            asset_manager_param: float,
            net_positions: int,
            alpha: float,
            beta: float,
            cov_matrix: pd.DataFrame,
            expected_returns: pd.Series,
            transaction_costs: pd.Series,
            initial_portfolio: pd.Series,
            test_runs: int = 20,
            test_tol: float = 10e-8,
            qs_export_path: str | None = None,
            export_decimal_places: int = 4) -> pd.DataFrame:
        """"Compute the quadratic matrix Q which encodes the following binary
        optimisation model:
        f(z+, z-) := asset_manager_lambda * pf_risk
                     - (1-asset_manager_lambda) * (pf_gross_expected_return
                                                   - pf_rebalancing_cost) +
                     alpha * (count - net_positions)^2 +
                     beta * (z+^T z-),
        where

        pf_risk := 1/net_positions * w^T * cov_matrix * w,

        pf_gross_expected_return := exp_returns^T * w,

        pf_rebalancing_cost := transaction_costs^T * abs(w
                                                         - initial_portfolio),

        count is equal to the number of entries in the portfolio which
        are not zero., and w := 1/net_positions * (z+ - z-) is a vector with
        elements in {-1/net_positions, 0, 1/net_positions}.

        Parameters
        ----------
        asset_manager_param : float
            Expresses the trade-off between risk and return. Setting it to 0
            maximises return only. Setting it to 1 minimises volatility only.
        net_positions : int
            Only portfolios with exactly this number of positions, i.e.,
            non-zero elements, will be considered feasible solutions.
        alpha : float
            Penalty parameter for the cardinality soft-constraint.
        beta : float
            Penalty parameter for the degenerate solutions infeasibility
            soft-constraint.
        cov_matrix : pandas.DataFrame
            Covariance matrix of asset returns.
        expected_returns : pandas.Series
            Expected returns indexed vector.
        transaction_costs : pandas.Series
            Indexed vector containing the transaction costs of each asset in
            percentage terms.
        initial_portfolio : pandas.Series
            Indexed vector containing the initial portfolio asset weights,
            which will be considered in the computation of transaction costs
            when rebalancing the portfolio from initial_portfolio to optimal
            portfolio.
        test_runs : int, default 20
            Number of random binary vectors of size 2*N used to test that each
            QUBO matrix used to construct the final Q matrix were correctly
            constructed.
        test_tol : float, default 10e-8
            Maximum deviations from the actual value allowed in the testing of
            each QUBO matrix used to construct the final Q matrix were
            correctly constructed.
        qs_export_path : str, default None
            If not None, the resulting Q matrix is exported in .qs integer
            format to this path.
        export_decimal_places : int, default 4
            Number of decimal places to round the entries in Q before export.

        Returns
        -------
        Q : pandas.DataFrame
            Row and column indexed matrix of size (2*N, 2*N) which encondes
            the discrete equal weight Markowitz portfolio optimisation model
            as a quadratic unconstrained binary optimisation model, for N
            assets.
        """

        cov_matrix = cov_matrix.sort_index(axis=0)
        cov_matrix = cov_matrix.sort_index(axis=1)
        expected_returns = expected_returns.sort_index()
        transaction_costs = transaction_costs.sort_index()
        initial_portfolio = initial_portfolio.sort_index()

        index_match = (
            cov_matrix.index.equals(cov_matrix.columns)
            and cov_matrix.index.equals(expected_returns.index)
            and expected_returns.index.equals(transaction_costs.index)
            and transaction_costs.index.equals(initial_portfolio.index)
            )

        assert index_match, \
            ("The indices of the covariance matrix, expected returns vector,"
             " transaction costs vector and initial portfolio vector do not"
             " match.")

        # Get the number of assets in the portfolio.
        N = self.NumberOfAssetsInPortfolio

        assert cov_matrix.shape == (N, N), \
            (f"The covariance matrix was expected to have shape ({N}, {N})"
             f" but it has shape {cov_matrix.shape}.")

        assert expected_returns.shape[0] == N, \
            ("The dimension of vector of expected returns does not match the"
             " number of assets in the portfolio.")

        assert transaction_costs.shape[0] == N, \
            ("The dimension of vector of transaction costs does not match the"
             " number of assets in the portfolio.")

        assert initial_portfolio.shape[0] == N, \
            ("The dimension of initial portfolio vector does not match the"
             " number of assets in the portfolio.")

        # Build Q_risk matrix.
        Q_risk = np.block([[cov_matrix, -cov_matrix],
                           [-cov_matrix, cov_matrix]])
        Q_risk = asset_manager_param * 1/net_positions**2 * Q_risk

        assert Q_risk.shape == (2*N, 2*N), \
            (f"The resulting Q_risk matrix has shape {Q_risk.shape},"
             f" but shape ({2*N}, {2*N}) was expected.")

        # Build Q_netreturn matrix.
        Delta_0 = np.diag(np.multiply(transaction_costs,
                                      np.abs(initial_portfolio)))
        Delta_plus = np.diag(np.multiply(transaction_costs,
                                         np.abs(initial_portfolio
                                                - 1/net_positions)))
        Delta_minus = np.diag(np.multiply(transaction_costs,
                                          np.abs(initial_portfolio
                                                 + 1/net_positions)))
        Q_netreturn = np.block([
            [1/net_positions * np.diag(expected_returns) + Delta_0-Delta_plus,
             -Delta_0 + Delta_plus],
            [-Delta_0 + Delta_minus,
             -1/net_positions * np.diag(expected_returns) + Delta_0-Delta_minus]
            ])
        Q_netreturn = (1-asset_manager_param) * Q_netreturn

        assert Q_netreturn.shape == (2*N, 2*N), \
            (f"The resulting Q_netreturn matrix has shape {Q_netreturn.shape},"
             f" but shape ({2*N}, {2*N}) was expected.")

        # Build Q_csc matrix.
        Q_csc = alpha * (np.ones((2*N, 2*N)) - 2 * net_positions * np.eye(2*N))

        assert Q_csc.shape == (2*N, 2*N), \
            (f"The resulting Q_csc matrix has shape {Q_csc.shape},"
             f" but shape ({2*N}, {2*N}) was expected.")

        # Build Q_dssc matrix.
        Q_dssc = beta/2 * np.block([[np.zeros((N, N)), np.eye(N)],
                                    [np.eye(N), np.zeros((N, N))],
                                    ])
        assert Q_dssc.shape == (2*N, 2*N), \
            (f"The resulting Q_dssc matrix has shape {Q_dssc.shape},"
             f" but shape ({2*N}, {2*N}) was expected.")

        for i in range(test_runs):
            # Generate random binary solution of the form:
            # [z+, z-] in {0,1}^(2*N).
            random_solution = np.random.choice([0, 1], size=2*N)
            pf_solution = random_solution[:N] - random_solution[N:]

            # Test the Q_risk matrix definition.
            qubo_total_pf_risk = np.dot(random_solution,
                                        np.dot(Q_risk,
                                               random_solution))
            total_pf_risk = np.dot(pf_solution,
                                   np.dot(cov_matrix, pf_solution))
            total_pf_risk = (asset_manager_param
                             * 1/net_positions**2
                             * total_pf_risk)

            assert np.abs(total_pf_risk - qubo_total_pf_risk) < test_tol, \
                (f"The QUBO total portfolio risk of {qubo_total_pf_risk}"
                 " does not match the obj. function total portfolio risk of"
                 f" {total_pf_risk}.")

            # Test the Q_netreturn matrix definition.
            qubo_total_pf_exp_net_return = (np.dot(random_solution,
                                                   np.dot(Q_netreturn,
                                                          random_solution)))
            # Add a constant which is not included in the QUBO matrix or the
            # QUBO formulation.
            missing_cnst = (1-asset_manager_param) * np.sum(Delta_0)
            qubo_total_pf_exp_net_return = (qubo_total_pf_exp_net_return
                                            - missing_cnst)

            total_pf_exp_gross_return = (1/net_positions
                                         * np.dot(expected_returns,
                                                  pf_solution))
            total_pf_transaction_costs = np.dot(transaction_costs,
                                                np.abs(initial_portfolio
                                                       - 1/net_positions
                                                       * pf_solution))
            total_pf_exp_net_return = (total_pf_exp_gross_return
                                       - total_pf_transaction_costs)
            total_pf_exp_net_return = ((1-asset_manager_param)
                                       * total_pf_exp_net_return)

            assert np.abs(total_pf_exp_net_return
                          - qubo_total_pf_exp_net_return) < test_tol, \
                ("The QUBO total portfolio expected net return of"
                 f" {qubo_total_pf_exp_net_return} does not match the obj."
                 " function total portfolio expected net return of"
                 f" {total_pf_exp_net_return}.")

            # Test Q_csc matrix definition.
            qubo_card_soft_constr_val = (np.dot(random_solution,
                                                np.dot(Q_csc, random_solution))
                                         + alpha * net_positions**2)
            card_soft_constr_val = alpha * (np.sum(random_solution)
                                            - net_positions)**2

            assert np.abs(card_soft_constr_val
                          - qubo_card_soft_constr_val) < test_tol, \
                ("The QUBO cardinality soft-constraint value of"
                 f" {qubo_card_soft_constr_val} does not match the obj."
                 " function cardinality soft-constraint of"
                 f" {card_soft_constr_val}.")

            # Test Q_dssc matrix definition.
            qubo_deg_sol_soft_constr_val = np.dot(random_solution,
                                                  np.dot(Q_dssc,
                                                         random_solution))
            deg_sol_soft_constr_val = beta * np.dot(random_solution[:N],
                                                    random_solution[N:])

            assert np.abs(deg_sol_soft_constr_val
                          - qubo_deg_sol_soft_constr_val) < test_tol, \
                ("The QUBO degenerate solutions infeasibility soft-constraint"
                 f" value of {qubo_deg_sol_soft_constr_val} does not match the"
                 " objective function degenerate solutions infeasibility"
                 f" soft-constraint value of {deg_sol_soft_constr_val}.")

        # After testing all matrices, construct the final matrix which
        # summarises the whole optimisation model.
        Q = Q_risk - Q_netreturn + Q_csc + Q_dssc

        # Index the matrix Q with the names of the corresponding variables.
        Q_index = expected_returns.index
        Q_index_long = Q_index.map(lambda asset: asset + "_long")
        Q_index_short = Q_index.map(lambda asset: asset + "_short")
        Q_index = Q_index_long.union(Q_index_short, sort=False)
        Q = pd.DataFrame(data=Q, index=Q_index, columns=Q_index)

        if qs_export_path is not None:
            # Export matrix Q as .qs file.
            matrix_to_export = Q.round(decimals=export_decimal_places)
            matrix_to_export = matrix_to_export * 10**export_decimal_places

            with open(qs_export_path, "w") as f:
                n, m = matrix_to_export.shape
                f.write(f"{n} {m}")

                for i in range(n):
                    for j in range(i, m):
                        if matrix_to_export.iloc[i, j] != 0:
                            f.write((f"\n{i+1}"
                                     f" {j+1}"
                                     f" {int(matrix_to_export.iloc[i, j])}"))

        return Q
