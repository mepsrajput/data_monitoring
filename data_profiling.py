import argparse
import pyspark.sql.functions as F
from pyspark.sql import SparkSession
from tabulate import tabulate
import unittest
import logging

def determine_variable_type(df, column):
    """
    Determines the variable type of a column in a Spark DataFrame.

    Args:
        df: The Spark DataFrame.
        column: The name of the column.

    Returns:
        'numeric' if the column is numeric, 'categorical' if it is categorical, None if the column is not recognized.
    """
    data_type = df.schema[column].dataType
    if data_type in ['integer', 'double', 'float']:
        return 'numeric'
    elif data_type == 'string':
        return 'categorical'
    else:
        logging.warning(f"Unrecognized data type for column '{column}': {data_type}")
        return None

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
    count = df.select(column).count()
    missing_count = df.select(column).filter(F.isnan(column) | F.col(column).isNull()).count()
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
                    stats[stat_name] = df.approxQuantile(column, [percentile], 0.01)[0]
                else:
                    stats[stat_name] = df.approxQuantile(column, [percentile], 0)[0]
            elif stat_name == 'Outliers':
                q1, q3 = df.approxQuantile(column, [0.25, 0.75], 0)
                iqr = q3 - q1
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                outlier_count = df.filter((F.col(column) < lower_bound) | (F.col(column) > upper_bound)).count()
                stats[stat_name] = outlier_count
            else:
                stats[stat_name] = df.select(getattr(F, stat_name.lower())(column)).first()[0]
        except Exception as e:
            stats[stat_name] = None

    return stats

def calculate_categorical_summary(df, column):
    """
    Calculates summary statistics for a categorical column.

    Args:
        df: The Spark DataFrame.
        column: The name of the column.

    Returns:
        A dictionary containing the summary statistics.
    """
    count = df.select(column).count()
    missing_count = df.select(column).filter(F.col(column).isNull() | (F.col(column) == "")).count()
    mode_result = df.groupBy(column).count().orderBy(F.desc("count")).first()
    mode = mode_result[column] if mode_result else None

    stats = {
        "Column": column,
        "Variable Type": 'categorical',
        "Count": count,
        "Missing": missing_count,
        "Distinct Values": df.agg(F.countDistinct(column)).first()[0],
        "Mode": mode
    }

    return stats

def format_summary(stats_list):
    """
    Formats the summary statistics for printing.

    Args:
        stats_list: A list of dictionaries containing the summary statistics.

    Returns:
        A formatted string of the statistics.
    """
    return tabulate(stats_list, headers="keys", tablefmt="pipe")

def profile_df(df, summary_level):
    """
    This function profiles a Spark DataFrame and logs the output.

    Args:
        df: The Spark DataFrame to be profiled.
        summary_level: The level of detail in the summary: 'summary' for selected statistics, 'detailed' for detailed summary.

    Returns:
        None
    """
    # Get the column names.
    columns = df.columns

    # Separate numeric and categorical variables.
    numeric_vars = []
    categorical_vars = []

    # Iterate over the columns and determine the variable type.
    for column in columns:
        var_type = determine_variable_type(df, column)
        if var_type == 'categorical':
            categorical_vars.append(column)
        else:
            numeric_vars.append(column)

    # Calculate statistics for numeric variables.
    numeric_stats = []
    for column in numeric_vars:
        numeric_stats.append(calculate_numeric_summary(df, column, summary_level))

    # Calculate statistics for categorical variables.
    categorical_stats = []
    for column in categorical_vars:
        categorical_stats.append(calculate_categorical_summary(df, column))

    # Format the statistics for printing.
    formatted_numeric_stats = format_summary(numeric_stats)
    formatted_categorical_stats = format_summary(categorical_stats)

    # Log the statistics.
    logging.info("Distribution of Numeric Variables")
    logging.info("---------------------------------")
    logging.info(formatted_numeric_stats)
    logging.info("")
    logging.info("Distribution of Categorical Variables")
    logging.info("-------------------------------------")
    logging.info(formatted_categorical_stats)

class TestDataProfiling(unittest.TestCase):
    def setUp(self):
        # Initialize Spark session for testing
        self.spark = SparkSession.builder.appName("Test Data Profile").getOrCreate()

    def tearDown(self):
        # Stop Spark session after testing
        self.spark.stop()

    def test_profile_df_with_empty_dataframe(self):
        # Test case with an empty DataFrame
        columns = ["ID", "Category", "Value"]
        df = self.spark.createDataFrame([], columns)

        # Call the function and capture the output
        output = profile_df(df, summary_level="summary")

        # Define the expected output
        expected_output = """
        |   Column | Variable Type   |   Count |   Missing |   Min |   Max |   P5 |   P50 |   P75 |
        |---------:|:----------------|--------:|----------:|------:|------:|-----:|------:|------:|
        |       ID | None            |       0 |         0 |  None |  None | None |  None |  None |
        | Category | None            |       0 |         0 |  None |  None | None |  None |  None |
        |    Value | None            |       0 |         0 |  None |  None | None |  None |  None |
        """

        # Remove leading white spaces from the expected output (tabulate adds extra spaces)
        expected_output = "\n".join(line.lstrip() for line in expected_output.strip().split("\n"))

        # Compare the output with the expected output
        self.assertEqual(output.strip(), expected_output)

    def test_profile_df_with_unrecognized_data_type(self):
        # Test case with an unrecognized data type
        data = [(1, "A", 100), (2, "B", "Invalid"), (3, "C", 300)]
        columns = ["ID", "Category", "Value"]
        df = self.spark.createDataFrame(data, columns)

        # Call the function and capture the output
        output = profile_df(df, summary_level="summary")

        # Define the expected output
        expected_output = """
        |   Column | Variable Type   |   Count |   Missing |   Min |   Max |   P5 |   P50 |   P75 |
        |---------:|:----------------|--------:|----------:|------:|------:|-----:|------:|------:|
        |       ID | numeric         |       3 |         0 |     1 |     3 | 1.1  |     2 |     3 |
        | Category | categorical     |       3 |         0 |  None |  None | None |  None |  None |
        |    Value | numeric         |       3 |         1 |   100 |   300 | 100  |   200 |   300 |
        """

        # Remove leading white spaces from the expected output (tabulate adds extra spaces)
        expected_output = "\n".join(line.lstrip() for line in expected_output.strip().split("\n"))

        # Compare the output with the expected output
        self.assertEqual(output.strip(), expected_output)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    args = parser.parse_args()

    # Initialize Spark session.
    spark = SparkSession.builder.appName("Data Profile").getOrCreate()

    # Load the data.
    df = spark.read.csv(args.data_path, inferSchema=True, header=True)

    # Profile the data.
    summary_level = "summary"
    profile_df(df, summary_level)

    # Stop Spark session after usage
    spark.stop()
