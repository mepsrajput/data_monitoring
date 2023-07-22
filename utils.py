# utils.py
import pyspark.sql.functions as F
from typing import Tuple

def determine_variable_type(df, column) -> str:
    """
    Determines the variable type of a column in a Spark DataFrame.

    Args:
        df: The Spark DataFrame.
        column: The name of the column.

    Returns:
        'numeric' if the column is numeric, 'categorical' if it is categorical, None if the column is not recognized.
    """
    data_type = df.schema[column].dataType
    if isinstance(data_type, (F.IntegerType, F.DoubleType, F.FloatType)):
        return 'numeric'
    elif isinstance(data_type, F.StringType):
        return 'categorical'
    else:
        logging.warning(f"Unrecognized data type for column '{column}': {data_type}")
        return None

def calculate_count_missing(df, column) -> Tuple[int, int]:
    """
    Calculates the count and missing values for a column.

    Args:
        df: The Spark DataFrame.
        column: The name of the column.

    Returns:
        A tuple (count, missing_count) containing the count and missing values.
    """
    count = df.select(column).count()
    missing_count = df.select(column).filter(F.isnan(column) | F.col(column).isNull() | (F.col(column) == "")).count()
    return count, missing_count

def calculate_numeric_summary(df, column, summary_level, use_approx_quantile=True):
    """
    Calculates summary statistics for a numeric column.

    Args:
        df: The Spark DataFrame.
        column: The name of the column.
        summary_level: The level of detail in the summary: 'summary' for selected statistics, 'detailed' for detailed summary.
        use_approx_quantile: Whether to use the approximate quantile calculation for percentiles.

    Returns:
        A dictionary containing the summary statistics.
    """
    count, missing_count = calculate_count_missing(df, column)
    percentiles = [0.05, 0.5, 0.75]

    if summary_level == 'summary':
        selected_stats = ['Count', 'Missing', 'Min', 'Max'] + [f'P{int(p * 100)}' for p in percentiles]
    else:
        selected_stats = ['Count', 'Missing', 'Min', 'Max'] + [f'P{int(p * 100)}' for p in percentiles] + ['Mean', 'Stddev', 'Variance', 'Skewness', 'Kurtosis', 'Outliers']

    stats = {
        "Column": column,
        "Variable Type": 'numeric'
    }

    for stat_name in selected_stats:
        try:
            if stat_name == 'Count':
                stats[stat_name] = count
            elif stat_name == 'Missing':
                stats[stat_name] = missing_count
            elif stat_name == 'Min':
                stats[stat_name] = df.select(F.min(column)).first()[0]
            elif stat_name == 'Max':
                stats[stat_name] = df.select(F.max(column)).first()[0]
            elif stat_name.startswith('P'):
                percentile = float(stat_name[1:]) / 100
                if use_approx_quantile:
                    stats[stat_name] = df.approxQuantile(column, [percentile], APPROX_QUANTILE_ERROR)[0]
                else:
                    stats[stat_name] = df.approxQuantile(column, [percentile], 0)[0]
            elif stat_name == 'Outliers':
                q1, q3 = df.approxQuantile(column, [0.25, 0.75], 0)
                iqr = q3 - q1
                lower_bound = q1 - OUTLIER_THRESHOLD * iqr
                upper_bound = q3 + OUTLIER_THRESHOLD * iqr
                outlier_count = df.filter((
