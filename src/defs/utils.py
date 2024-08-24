import io
import json
import os
import random
import string
from datetime import datetime
from pathlib import Path
from typing import Any, List, Optional, Tuple

import pandas as pd


class DateEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, datetime):
            return o.isoformat()
        return super().default(o)


def loadJson(fPath: Path):
    return json.loads(fPath.read_bytes())


def writeJson(fPath: Path, data):
    fPath.write_text(json.dumps(data, indent=3, cls=DateEncoder))


def randomChar(length):
    return "".join(random.choice(string.ascii_lowercase) for _ in range(length))


def csv_loader(
    file_path: Path,
    period=160,
    end_date: Optional[datetime] = None,
    chunk_size=1024 * 6,
) -> pd.DataFrame:

    def get_date(start, chunk) -> datetime:
        end = chunk.find(b",", start)

        date_str = chunk[start:end].decode()

        if len(date_str) > 10:
            date = datetime.strptime(date_str[:16], datetime_fmt)
        else:
            date = datetime.strptime(date_str, date_fmt)

        return date

    size = os.path.getsize(file_path)
    datetime_fmt = "%Y-%m-%d %H:%M"
    date_fmt = "%Y-%m-%d"

    if size <= chunk_size and not end_date:
        return pd.read_csv(
            file_path, index_col="Date", parse_dates=["Date"]
        ).iloc[-period:]

    chunks_read = []  # store the bytes chunk in a list
    start_date = None
    prev_chunk_start_line = None
    holiday_offset = max(3, period // 50 * 3)

    if end_date:
        start_date = end_date - pd.offsets.BDay(period + holiday_offset)

    # Open in binary mode and read from end of file
    with file_path.open(mode="rb") as f:

        # Read the first line of file to get column names
        columns = f.readline()

        curr_pos = size

        while curr_pos > 0:
            read_size = min(chunk_size, curr_pos)

            # Set the current read position in the file
            f.seek(curr_pos - read_size)

            # From the current position read n bytes
            chunk = f.read(read_size)

            if end_date:
                # First line in a chunk may not be complete line
                # So skip the first line and parse the first date in chunk
                newline_index = chunk.find(b"\n")

                start = newline_index + 1

                current_dt = get_date(start, chunk)

                # start storing chunks once end date has reached
                if current_dt <= end_date:
                    if prev_chunk_start_line:
                        chunk = chunk + prev_chunk_start_line
                        prev_chunk_start_line = None

                    if start_date and current_dt <= start_date:
                        # reached starting date
                        # add the columns to chunk and append it
                        chunks_read.append(columns + chunk[start:])
                        break

                    chunks_read.append(chunk)
                else:
                    prev_chunk_start_line = chunk[: chunk.find(b"\n")]

            else:

                if curr_pos == size:
                    # On first chunk, get the last date to calculate start_date
                    last_newline_index = chunk[:-1].rfind(b"\n")

                    start = last_newline_index + 1
                    last_dt = get_date(start, chunk)

                    start_date = last_dt - pd.offsets.BDay(
                        period + holiday_offset
                    )

                # First line may not be a complete line.
                # To skip this line, find the first newline character
                newline_index = chunk.find(b"\n")

                start = newline_index + 1

                try:
                    current_dt = get_date(start, chunk)
                except ValueError:
                    # reached start of file. No valid date to parse
                    chunks_read.append(chunk)
                    break

                if start_date is None:
                    start_date = datetime.now() - pd.offsets.BDay(
                        period + holiday_offset
                    )

                if current_dt <= start_date:
                    # Concatenate the columns and chunk together
                    # and append to list
                    chunks_read.append(columns + chunk[start:])
                    break

                # we are storing the chunks in bottom first order.
                # This has to be corrected later by reversing the list
                chunks_read.append(chunk)

            curr_pos -= read_size

        if end_date and not chunks_read:
            # If chunks_read is empty, end_date was not found in file
            raise IndexError("Date out of bounds of current DataFrame")

        # Reverse the list and join it into a bytes string.
        # Store the result in a buffer
        buffer = io.BytesIO(b"".join(chunks_read[::-1]))

    df = pd.read_csv(
        buffer,
        parse_dates=["Date"],
        index_col="Date",
    )

    if end_date:
        return df.loc[:end_date].iloc[-period:]
    else:
        return df.iloc[-period:]


def getDataFrame(
    fpath: Path,
    tf: str,
    period: int,
    column: Optional[str] = None,
    toDate: Optional[datetime] = None,
) -> Any:
    candle_count = period * 5 if tf == "weekly" else period

    df = csv_loader(fpath, candle_count, end_date=toDate)

    dct: dict = {
        "Open": "first",
        "High": "max",
        "Low": "min",
        "Close": "last",
        "Volume": "sum",
    }

    if tf == "weekly":
        if column:
            return df[column].resample("W").apply(dct[column])[-period:]

        return df.resample("W").apply(dct)[-period:]

    return df[-period:] if column is None else df[column][-period:]


def arg_parse_dict(dct: dict) -> list:
    """
    Convert a dictionary of arguments and values into a list of command-line
    arguments.

    Parameters:
    - dct (dict): Dictionary containing argument names and values.

    Returns:
    - list: List of command-line style arguments.

    Example:
    ```python
    args = {'input_file': 'data.txt', 'output_dir': '/output', 'verbose': True}
    command_line_args = arg_parse_dict(args)
    ```
    """

    result = []

    for arg, val in dct.items():
        if val is False or val is None:
            continue

        arg = arg.replace("_", "-")

        result.append(f"--{arg}")

        if val is not True:
            if isinstance(val, list):
                result.extend(map(str, val))
            else:
                result.append(str(val))

    return result


def getDeliveryLevels(df, config):
    # Average of traded volume
    avgTrdQty = df["QTY_PER_TRADE"].rolling(config.DLV_AVG_LEN).mean().round(2)

    # Average of delivery
    avgDlvQty = df["DLV_QTY"].rolling(config.DLV_AVG_LEN).mean().round(2)

    # above average delivery days
    df["DQ"] = df["DLV_QTY"] / avgDlvQty

    # above average Traded volume days
    df["TQ"] = df["QTY_PER_TRADE"] / avgTrdQty

    # get combination of above average traded volume and delivery days
    df["IM_F"] = (df["TQ"] > 1.2) & (df["DQ"] > 1.2)

    # see https://github.com/matplotlib/mplfinance/blob/master/examples/marketcolor_overrides.ipynb
    df["MCOverrides"] = None
    df["IM"] = float("nan")

    for idx in df.index:
        dq, im = df.loc[idx, ["DQ", "IM_F"]]

        if im:
            df.loc[idx, "IM"] = df.loc[idx, "Low"] * 0.99

        if dq >= config.DLV_L3:
            df.loc[idx, "MCOverrides"] = config.PLOT_DLV_L1_COLOR
        elif dq >= config.DLV_L2:
            df.loc[idx, "MCOverrides"] = config.PLOT_DLV_L2_COLOR
        elif dq > config.DLV_L1:
            df.loc[idx, "MCOverrides"] = config.PLOT_DLV_L3_COLOR
        else:
            df.loc[idx, "MCOverrides"] = config.PLOT_DLV_DEFAULT_COLOR


def isFarFromLevel(
    level: float,
    levels: List[Tuple[pd.DatetimeIndex, float]],
    mean_candle_size: float,
) -> bool:
    """Returns true if difference between the level and any of the price levels
    is greater than the mean_candle_size."""
    # Detection of price support and resistance levels in Python -Gianluca Malato
    # source: https://towardsdatascience.com/detection-of-price-support-and-resistance-levels-in-python-baedc44c34c9
    return sum([abs(level - x[1]) < mean_candle_size for x in levels]) == 0


def getLevels(
    df: pd.DataFrame, mean_candle_size: float
) -> List[
    Tuple[Tuple[pd.DatetimeIndex, float], Tuple[pd.DatetimeIndex, float]]
]:
    """
    Identify potential support and resistance levels in a DataFrame.

    Parameters:
    - df (pd.DataFrame): DataFrame containing at least 'High' and 'Low' columns.
    - mean_candle_size (float): The mean size of a candle, used as a threshold for level clustering.

    Returns:
    - list of tuples: Each tuple represents a horizontal line segment, defined by two points.
      Each point is a tuple containing date and price.
      The list represents identified support and resistance levels.

    Algorithm:
    - The function uses local maxima and minima in the 'High' and 'Low' prices to identify potential reversal points.
    - It filters for rejection from the top (local maxima) and from the bottom (local minima).
    - To avoid clustering of support and resistance lines, it utilizes the isFarFromLevel function.
    - Identified levels are returned as horizontal line segments for visualization.

    Example Usage:
    ```python
    # Example DataFrame df with 'High' and 'Low' columns
    levels = getLevels(df, mean_candle_size=2.0)
    ```

    Note:
    - It is recommended to provide a DataFrame with sufficient historical price data for accurate level identification.
    - The function is designed for use in financial technical analysis.
    """

    levels = []

    # filter for rejection from top
    # 2 succesive highs followed by 2 succesive lower highs
    local_max = df["High"][
        (df["High"].shift(1) < df["High"])
        & (df["High"].shift(2) < df["High"].shift(1))
        & (df["High"].shift(-1) < df["High"])
        & (df["High"].shift(-2) < df["High"].shift(-1))
    ].dropna()

    # filter for rejection from bottom
    # 2 succesive highs followed by 2 succesive lower highs
    local_min = df["Low"][
        (df["Low"].shift(1) > df["Low"])
        & (df["Low"].shift(2) > df["Low"].shift(1))
        & (df["Low"].shift(-1) > df["Low"])
        & (df["Low"].shift(-2) > df["Low"].shift(-1))
    ].dropna()

    for idx in local_max.index:
        level = local_max[idx]

        # Prevent clustering of support and resistance lines
        # Only add a level if it at a distance from any other price lines
        if isFarFromLevel(level, levels, mean_candle_size):
            levels.append((idx, level))

    for idx in local_min.index:
        level = local_min[idx]

        if isFarFromLevel(level, levels, mean_candle_size):
            levels.append((idx, level))

    alines = []
    lastDt = df.index[-1]

    for dt, price in levels:
        # a tuple containing start and end point coordinates for a horizontal line
        # Each tuple is composed of date and price.
        seq = ((dt, price), (lastDt, price))
        alines.append(seq)

    return alines


def isFarFromLevel_v2(
    level: float,
    levels: List[Tuple[pd.Timestamp, float]],
    mean_candle_size: float,
):
    """Returns true if difference between the level and any of the price levels
    is greater than the mean_candle_size."""
    # Detection of price support and resistance levels in Python -Gianluca Malato
    # source: https://towardsdatascience.com/detection-of-price-support-and-resistance-levels-in-python-baedc44c34c9
    return sum([abs(level - x[1]) < mean_candle_size for x in levels]) == 0


def getLevels_v2(df: pd.DataFrame, mean_candle_size: float):

    levels = []

    highs_mask = (
        (df.High.shift(1) < df.High)
        & (df.High.shift(2) < df.High)
        & (df.High.shift(3) < df.High)
        & (df.High.shift(-1) < df.High)
        & (df.High.shift(-2) < df.High)
        & (df.High.shift(-3) < df.High)
    )

    lows_mask = (
        (df.Low.shift(1) > df.Low)
        & (df.Low.shift(2) > df.Low)
        & (df.Low.shift(3) > df.Low)
        & (df.Low.shift(-1) > df.Low)
        & (df.Low.shift(-2) > df.Low)
        & (df.Low.shift(-3) > df.Low)
    )

    # filter for rejection from top
    # 2 succesive highs followed by 2 succesive lower highs
    max = df["High"].loc[highs_mask].dropna()
    min = df["Low"].loc[lows_mask].dropna()

    max_min = pd.concat([max, min], axis=0)

    max_min = max_min.loc[~max_min.index.duplicated()]

    for i, lv in max_min.items():

        touch_count = max_min.loc[
            (max_min - lv).abs() < mean_candle_size
        ].count()

        if touch_count > 1 and isFarFromLevel_v2(lv, levels, mean_candle_size):
            levels.append((i, lv))

    return [((i, lv), (df.index[-1], lv)) for i, lv in levels]


# def getScreenSize():
#     root = Tk()
#     root.withdraw()
#     mm = 25.4
#
#     width, height = root.winfo_screenmmwidth(), root.winfo_screenmmheight()
#
#     return (round(width / mm), round(height / mm))


def relativeStrength(close: pd.Series, index_close: pd.Series) -> pd.Series:
    return (close / index_close * 100).round(2)


def manfieldRelativeStrength(
    close: pd.Series, index_close: pd.Series, period: int
) -> pd.Series:
    rs = relativeStrength(close, index_close)

    sma_rs = rs.rolling(period).mean()
    return ((rs / sma_rs - 1) * 100).round(2)
