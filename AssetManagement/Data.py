# ----------------------------------------------------------------------------
# Written by Francisco Jose Manjon Cabeza Garcia for his Master Thesis at    |
# Technische Universität Berlin.                                             |
#                                                                            |
# This file contains a collection of methods to get data from the            |
# AlphaVantage API and prepare it for further use in portfolio optimisation  |
# methods.                                                                   |
# ----------------------------------------------------------------------------

from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import requests
import time

# Maximum number of calls to the AlphaVantage API per minute.
alpha_vantage_API_call_limit = 75
# Number of seconds to wait after a GET request fails before trying again.
on_get_request_error_waiting_time = 60


def GetOverviewFromAlphaVantage(alpha_vantage_api_key: str,
                                asset_tickers: list[str],
                                overview_keys: list[str] | None = None,
                                verbose: bool = False
                                ) -> pd.DataFrame:
    """Pull an overview of the passed asset tickers from AlphaVantage. An
    overview contains data like the country, currency, industry, etc. More
    information can be found here:
    https://www.alphavantage.co/documentation/#company-overview

    Parameters
    ----------
    alpha_vantage_api_key : str
        API key for Alpha Vantage.
    asset_tickers : list[str]
        Ticker for which to retrieve an overview.
    overview_keys : list[str] or None, default None
        List of overview attributes to select and save for returning. A list
        of all available attributes which can be selected can be found at
        https://www.alphavantage.co/documentation/#company-overview. By
        default, i.e., for overview_keys=None, all attributes are saved and
        returned for each ticker.
    verbose : bool, default False
        If True, the success or failure retrieving the overview of each ticker
        passed is printed out.

    Returns
    -------
    overview_data : pandas.DataFrame
        Pandas DataFrame row-indexed by the tickers initially passed and with
        the (selected) overview attributes in the columns.
    """

    # Datetime object for the last AlphaVantage API call.
    last_request_time = datetime.min

    # Define the parametrised URL for the GET request.
    parametrised_request_url = ("https://www.alphavantage.co/query?"
                                "function=OVERVIEW"
                                "&symbol={0}"
                                "&apikey={1}")

    # Define an empty dictionary where the request results are saved indexed
    # by asset ticker.
    overview_data = {}

    # Iterate over all passed asset ticker.
    for ticker in asset_tickers:
        # Get the request URL for the current ticker.
        request_url = parametrised_request_url.format(ticker,
                                                      alpha_vantage_api_key)

        # Compute the seconds passed since the last AlphaVantage API call.
        seconds_since_last_api_call = datetime.now() - last_request_time
        seconds_since_last_api_call = (seconds_since_last_api_call.microseconds
                                       * 10e-6)
        if seconds_since_last_api_call < 60 / alpha_vantage_API_call_limit:
            # The last AlphaVantage API call lies less than the minimum
            # seconds given by the API call limit away.
            # Compute the number of seconds to wait to stay within the API
            # calls limit.
            seconds_to_wait = (60 / alpha_vantage_API_call_limit
                               - seconds_since_last_api_call + 10e-6)
            if verbose:
                print(f"Waiting {seconds_to_wait} seconds for the next"
                      " AlphaVantage API call to remain within the "
                      f" {alpha_vantage_API_call_limit} calls/minute limit.")
            # Wait.
            time.sleep(seconds_to_wait)

        try:
            # Run GET request for the current ticker.
            request_result = requests.get(request_url)
            # Parse the GET request result into a dictionary.
            ticker_overview = request_result.json()
        except Exception as e:
            # The GET request failed.

            if verbose:
                print(e)
                print(f"Waiting {on_get_request_error_waiting_time} seconds"
                      " to retry...")

            # Wait some time and try again.
            time.sleep(on_get_request_error_waiting_time)

            if verbose:
                print("Retrying GET request...")

            # Run GET request for the current ticker once again.
            request_result = requests.get(request_url)
            # Parse the GET request result into a dictionary.
            ticker_overview = request_result.json()

        # Save the time of the GET request to AlphaVantage.
        last_request_time = datetime.now()

        if overview_keys is not None:
            # A list of overview attributes was passed.
            # Only save those attributes for the ticker.
            overview_data[ticker] = {
                key: value for key, value in ticker_overview.items()
                if key in overview_keys
                }
        else:
            # No list of overview attributes was passed.
            # Save all overview attributes for the ticker.
            overview_data[ticker] = ticker_overview

        if verbose:
            print(f"Overview for {ticker} retrieved.")

    # Convert the dictionary with the overview data indexed by ticker into a
    # pandas DataFrame object also indexed by ticker.
    overview_data = pd.DataFrame.from_dict(overview_data, orient="index")

    return overview_data


def GetHistoricalDataFromAlphaVantage(
        alpha_vantage_api_key: str,
        asset_tickers: list[str],
        data_periods: int | None = None,
        data_start_date: datetime | None = None,
        on_no_data_error: str = "raise",
        on_data_periods_error: str = "raise",
        on_na_error: str = "raise",
        na_interpolation_method: str = "linear",
        export_path: str = "",
        verbose: bool = False
        ) -> pd.DataFrame:
    """Pull historical data from Alpha Vantage. More information can be found
    here: https://www.alphavantage.co/documentation/#dailyadj

    Parameters
    ----------
    alpha_vantage_api_key : str
        API key for Alpha Vantage.
    asset_tickers : list
        List of asset tickers to fetch data for.
    data_periods : int, default None
        If not None, number of data points to retrieve for each ticker. If
        None, then all data is returned.
    data_start_date : datetime.datetime, default None
        If not None, all the data points with date strictly before this
        parameter will be ignored, and data_periods many data points from this
        date on will be selected. If None, then the last data_periods many
        data points available will be selected.
    on_no_data_error : str, default "raise"
        If no data can be retrieved for some ticker ,i.e., the API request
        returns an empty response, do the following:
        - if "raise", raise and assertion error.
        - if "skip", ignore the affected ticker form the output DataFrames.

    on_data_periods_error : str, default "raise"
        If the parameter data_periods is None and the retrieved data for some
        ticker contains less than data_periods many observations, do the
        following:
        - if "raise", raise and assertion error.
        - if "skip", ignore the affected ticker form the output DataFrames.

    on_na_error : str, default "raise"
        If the final DataFrame with historical data for some ticker contains
        NaNs, do the following:
        - if "raise", raise and assertion error.
        - if "skip", ignore the affected ticker form the output DataFrames.
        - "interpolate", missing values are interpolated according to the
          method given by the parameter na_interpolation_method.

    na_interpolation_method : str, default "linear"
        Interpolation method passed to pandas.DataFrame.interpolate in case
        the "interpolate" method has been selected in the parameter
        on_na_error.
    export_path : str, default ""
        If not "", then the raw selected data is saved as a .csv file to this
        path.
    verbose : bool, default False
        If True, log messages are printed to document the steps in the method.

    Returns
    -------
    history : pandas.DataFrame
        If no valid data can be retrieved, an empty pandas DataFrame object is
        returned. Else, a pandas DataFrame with multi-indexed rows by asset
        ticker and timestamp (named "ticker", "timestamp"), and with the
        following columns is returned.
        - "open", open price of the asset at the corresponding timestamp,
        - "high", highest price reached during the day of the corresponding
          row timestamp,
        - "low", lowest price reached during the day of the corresponding row
          timestamp,
        - "close", close price at the end of the day of the corresponding row
          timestamp,
        - "adjusted_close", close price at the end of the day of the
          corresponding row timestamp adjusted for dividends and splits,
        - "volume", number of units (not currency units) traded during the day
          of the corresponding row timestamp,
        - "dividend_amount", dividend payed the day of the day of the
          corresponding row timestamp,
        - "split_coefficient", split coefficient until the day of the
          corresponding row timestamp,
        - "adj_return", return of the asset between the close of the previous
          day and the day of the corresponding row timestamp, computed using
          the "adjusted_close" column data.
    """

    # Datetime object for the last AlphaVantage API call.
    last_request_time = datetime.min

    # Define the parametrised URL for the GET request to AlphaVantage.
    parametrised_request_url = ("https://www.alphavantage.co/query?"
                                "function=TIME_SERIES_DAILY_ADJUSTED"
                                "&symbol={0}"
                                "&outputsize=full"
                                "&datatype=csv"
                                "&apikey={1}")

    # Define an empty list to save the DataFrames with the historical data
    # of each asset, and later concatenate them all togheter.
    history = []
    for ticker in sorted(asset_tickers):
        # Create the request URL for the ticker's GET request to AlphaVantage.
        request_url = parametrised_request_url.format(ticker,
                                                      alpha_vantage_api_key)

        # Compute the seconds passed since the last AlphaVantage API call.
        seconds_since_last_api_call = datetime.now() - last_request_time
        seconds_since_last_api_call = (seconds_since_last_api_call.microseconds
                                       * 10e-6)
        if seconds_since_last_api_call < 60 / alpha_vantage_API_call_limit:
            # The last AlphaVantage API call lies less than the minimum
            # seconds given by the API call limit away.
            # Compute the number of seconds to wait to stay within the API
            # calls limit.
            seconds_to_wait = (60 / alpha_vantage_API_call_limit
                               - seconds_since_last_api_call + 10e-6)
            if verbose:
                print(f"Waiting {seconds_to_wait} seconds for the next"
                      " AlphaVantage API call to remain within the "
                      f" {alpha_vantage_API_call_limit} calls/minute limit.")
            # Wait.
            time.sleep(seconds_to_wait)

        try:
            # Get data for the ticker.
            data = pd.read_csv(request_url, index_col=0, parse_dates=True)
        except Exception as e:
            # The GET request failed.

            if verbose:
                print(e)
                print(f"Waiting {on_get_request_error_waiting_time} seconds"
                      " to retry...")

            # Wait some time and try again.
            time.sleep(on_get_request_error_waiting_time)

            if verbose:
                print("Retrying GET request...")

            # Get data for the ticker.
            data = pd.read_csv(request_url, index_col=0, parse_dates=True)

        # Save the time of the GET request to AlphaVantage.
        last_request_time = datetime.now()

        # Check if we got data for the ticker.
        if data.shape[0] < 1 or data.shape[1] < 1:
            # No data could be retrieved for the ticker.
            if on_no_data_error == "raise":
                raise AssertionError(
                    f"For the ticker {ticker} not data could be retrived"
                    )

            elif on_no_data_error == "skip":
                if verbose:
                    print(f"The ticker {ticker} has been excluded from the"
                          " output because no data could be retrieved.")

                # Do not append the data for this ticker.
                continue
            else:
                raise ValueError(
                    "The selected on_no_data_error option is not implemented."
                    )

        # Sort index to sort the data by timestamps.
        data = data.sort_index()

        if verbose:
            print(f"For {ticker} the oldest data available is from"
                  f" {data.index[0]} and the lastest from {data.index[-1]}.")

        if data_periods is not None:
            if data_start_date is not None:
                # Select only the data starting at data_start_date.
                data = data.loc[data.index >= data_start_date]

                # Select only the next data_periods+1 data points after
                # data_start_date. We keep one more data point than as
                # specified by data_periods to compute the last period return.
                data = data.iloc[:data_periods+1, :]

            else:
                # Select only the last data_periods+1 data points after
                # data_start_date. We keep one more data point than as
                # specified by data_periods to compute the last period return.
                data = data.iloc[-(data_periods+1):, :]

            # Check that the selected data contains as many observations as
            # expected.
            if data.shape[0] != data_periods+1:
                if on_data_periods_error == "raise":
                    raise AssertionError(
                        f"For the ticker {ticker} there are only"
                        f" {data.shape[0]} periods worth of historical data."
                        )

                elif on_data_periods_error == "skip":
                    if verbose:
                        print(f"The ticker {ticker} has been excluded from the"
                              " output because there are not enough"
                              " observations to return the required data"
                              " periods.")

                    # Do not append the data for this ticker.
                    continue

                else:
                    raise ValueError(
                        "The selected on_data_periods_error option is"
                        " not implemented.")

        elif data_start_date is not None:
            # Select only the data starting at data_start_date.
            data = data.loc[data.index >= data_start_date]

        # Compute the adjusted return for each period.
        data.loc[:, "adj_return"] = (data.loc[:, "adjusted_close"]
                                     / data.loc[:, "adjusted_close"].shift(-1)
                                     ) - 1

        # Remove oldest item to have T many return periods. The return for
        # the last element cannot be computed, because we do not have the
        # adjusted close of the next period after it (it lies in the future).
        data = data.iloc[:-1, :]
        # Add the ticker name to the ticker column to later use it as index
        # level when all ticker DataFrames are concatenated.
        data.loc[:, "ticker"] = ticker
        # Set the index to the pairs (ticker, timestamp).
        data = data.reset_index().set_index(["ticker", "timestamp"])

        if data.isna().sum().sum() > 0:
            # There are observations for which we do not have data (NaN).
            if on_na_error == "raise":
                raise AssertionError(
                    f"For the ticker {ticker} there are NA observations."
                    )

            elif on_na_error == "skip":
                continue

            elif on_na_error == "interpolate":
                data = data.interpolate(method=na_interpolation_method,
                                        axis=0)

                if data.isna().sum().sum() > 0:
                    continue

            else:
                raise ValueError(
                    "The selected on_na_error option is not implemented."
                    )

        # Append the ticker's data to the list of history DataFrames.
        history.append(data)

    # Get the number of tickers for which data was retrieved (if
    # on_data_periods_error == "skip", some tickers might have been excluded
    # due to insufficient observations).
    num_tickers_in_hist = len(history)

    if len(history) <= 0:
        return pd.DataFrame()

    # Concatenate all ticker DataFrames.
    history = pd.concat(history, axis=0)

    if data_periods is not None:
        # Check that the final history DataFrame contains as many observations
        # as expected.
        assert history.shape[0] == num_tickers_in_hist * data_periods, \
            (f"Historical data DataFrame shape is {history.shape}."
             f" Expected were {num_tickers_in_hist * data_periods}"
             " rows.")

    if verbose:
        print(history)

    if export_path != "":
        # Export historical data.
        history.to_csv(export_path+"Historical_data.csv", sep=";")

    return history


def GetListedStocksFromAlphaVantage(date: datetime,
                                    min_days_since_ipo: int,
                                    alpha_vantage_api_key: str
                                    ) -> pd.DataFrame:
    """Get a pandas DataFrame with the summary information of the stocks
    listed on a particular date from AlphaVantage.

    Parameters
    ----------
    date : datetime.datetime
        Datetime object containing the date at which the stocks returned must
        have been listed.
    min_days_since_ipo : int,
        Minimum number of days between the IPO date and the parameter date.
        Stocks with less than this amount of days between the two mentioned
        dates are filtered out.
    alpha_vantage_api_key : str
        API key for Alpha Vantage.

    Returns
    -------
    selected_stocks : pandas.DataFrame
        pandas.DataFrame containing the list of stocks which were active on
        the given date and with at least min_days_since_ipo days passed since
        their IPO. The DataFrame contains the following columns:
        - "symbol", which contains the ticker of the stock,
        - "name", which contains the full name of the stock company,
        - "exchange", which contains the exchange at which the stock is listed
          with such ticker ("symbol").
        - "assetType", which contains the asset type, i.e., "stock",
        - "ipoDate", which contains the date of the stock's IPO,
        - "delistingDate", which contains the date of the delisting of the
          stock or None if the stock has not yet been delisted,
        - "status", which contains the status of the stock.
    """

    # Define the parametrised URL for the GET request to AlphaVantage API.
    parametrised_request_url = ("https://www.alphavantage.co/query?"
                                "function=LISTING_STATUS"
                                "&date={0}"
                                "&state=active"
                                "&apikey={1}")

    # Get all listed assets on the given date.
    request_url = parametrised_request_url.format(str(date.date()),
                                                  alpha_vantage_api_key)
    active_tickers = pd.read_csv(request_url,
                                 parse_dates=["ipoDate"])

    # Filter out assets which are not stocks.
    stocks_filter = (active_tickers.loc[:, "assetType"] == "Stock")
    active_stocks = active_tickers.loc[stocks_filter]

    # Filter out all tickers which contain spaces.
    spaces_filter = (active_stocks.loc[:, "symbol"].str.contains(" "))
    spaces_filter = ~(spaces_filter.astype(bool))
    active_stocks = active_stocks.loc[spaces_filter]

    # Filter out assets with IPO date less than min_days_since_ipo days until
    # the date given by parameter date.
    days_since_ipo_filter = date - active_stocks.loc[:, "ipoDate"]
    days_since_ipo_filter = days_since_ipo_filter.apply(lambda diff: diff.days)
    days_since_ipo_filter = (days_since_ipo_filter > min_days_since_ipo)
    selected_stocks = active_stocks.loc[days_since_ipo_filter]

    return selected_stocks


def GetNYSEListedStocksFromAlphaVantage(date: datetime,
                                        min_days_since_ipo: int,
                                        alpha_vantage_api_key: str
                                        ) -> pd.DataFrame:
    """Get a pandas DataFrame with the summary information of the stocks
    listed on a particular date on NYSE from AlphaVantage.

    Parameters
    ----------
    date : datetime.datetime
        Datetime object containing the date at which the stocks returned must
        have been listed.
    min_days_since_ipo : int,
        Minimum number of days between the IPO date and the parameter date.
        Stocks with less than this amount of days between the two mentioned
        dates are filtered out.
    alpha_vantage_api_key : str
        API key for Alpha Vantage.

    Returns
    -------
    nyse_listed_stocks : pandas.DataFrame
        pandas.DataFrame containing the list of stocks which were active on
        the given date and with at least min_days_since_ipo days passed since
        their IPO. The DataFrame contains the following columns:
        - "symbol", which contains the ticker of the stock,
        - "name", which contains the full name of the stock company,
        - "exchange", which contains the exchange at which the stock is listed
          with such ticker ("symbol"), i.e., "NYSE".
        - "assetType", which contains the asset type, i.e., "stock",
        - "ipoDate", which contains the date of the stock's IPO,
        - "delistingDate", which contains the date of the delisting of the
          stock or None if the stock has not yet been delisted,
        - "status", which contains the status of the stock.
    """

    # Retrieve listed stocks on the given date.
    listed_stocks = GetListedStocksFromAlphaVantage(
        date=date,
        min_days_since_ipo=min_days_since_ipo,
        alpha_vantage_api_key=alpha_vantage_api_key)

    # Filter out stocks not listed on the NYSE.
    nyse_filter = (listed_stocks.loc[:, "exchange"] == "NYSE")
    nyse_listed_stocks = listed_stocks.loc[nyse_filter]

    return nyse_listed_stocks


def GetRandomDate(min_days_dist_to_today: int,
                  start_year: int,
                  is_workday: bool) -> datetime:
    """Generate a random datetime object with a date at least some days away
    from today's date. Dates are generated by an uniform distribution sampling
    from 1 to 31 for the day, from 1 to 13 for the month and from start_year
    to today's year plus one for the year. That means that no 31st date can be
    generated and returned.

    Parameters
    ----------
    min_days_dist_to_today : int
        Minimum days distance from today's date for a random generated date to
        be valid and returned.
    start_year: int
        Minimum year for the random generated date.
    is_workday : bool
        If True, the randomly generated date is adjusted to one day before if
        it lies on a Saturday and to one day later if it lies on Sunday, such
        that the returned date lies on a workday.

    Returns
    -------
    random_date : datetime.datetime
        datetime object with a valid random date.
    """

    # Generate a random day.
    day = np.random.uniform(1, 31)
    day = int(day)

    # Generate a random month.
    month = np.random.uniform(1, 13)
    month = int(month)

    # Generate a random year.
    year = np.random.uniform(start_year, datetime.now().year+1)
    year = int(year)

    # If the generated month is February, cap the day at 28.
    if month == 2 and day > 28:
        day = 28

    # Format the generated date in a datetime object.
    random_date = datetime(year=year, month=month, day=day)

    if is_workday:
        if random_date.weekday() == 5:
            # The generated random date lies on a Saturday.
            # Set the random date to one day before, i.e., to Friday.
            random_date = random_date - timedelta(days=1)

        elif random_date.weekday() == 6:
            # The generated random date lies on a Sunday.
            # Set the random date to one day after, i.e., to Monday.
            random_date = random_date + timedelta(days=1)

    # If the generated date is not at least min_days_dist_to_today days from
    # today's date, try generating a new one.
    if (datetime.now() - random_date).days < min_days_dist_to_today:
        return GetRandomDate(min_days_dist_to_today=min_days_dist_to_today,
                             start_year=start_year,
                             is_workday=is_workday)
    else:
        return random_date


def ChooseRandomAssets(assets_set: list,
                       num_assets: int,
                       exclude_assets: list | None = None) -> list:
    """Choose a random sample of given length from a given set after
    excluding some elements using np.random.shuffle.

    Parameters
    ----------
    assets_set: list,
        List of assets from which to sample.
    num_assets : int
        Number of assets to sample.
    exclude_assets : list, default []
        List of assets to exclude from the sampling.

    Returns
    -------
    random_asset_selection : list
        List of randomly sampled assets.
    """

    # If the list of assets to exclude is not empty, exclude them from the
    # list of assets.
    if exclude_assets is not None:
        assets_set = [asset for asset in assets_set
                      if asset not in exclude_assets]

    # Perform a random shuffle of the the list of assets.
    np.random.shuffle(assets_set)
    # Select the first num_assets from the shuffled list of assets.
    random_asset_selection = assets_set[:num_assets]

    return random_asset_selection
