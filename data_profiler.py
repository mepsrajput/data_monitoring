# data_profiler.py
import argparse
import pyspark.sql.functions as F
from pyspark.sql import SparkSession
from tabulate import tabulate
from utils import determine_variable_type, calculate_count_missing, calculate_numeric_summary, calculate_categorical_summary, format_summary
import logging

# Logging Configuration
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# Magic Numbers
APPROX_QUANTILE_ERROR = 0.01
OUTLIER_THRESHOLD = 1.5

def profile_df(df, summary_level):
    """
    Profile a Spark DataFrame and print the summary statistics.

    Args:
        df: The Spark DataFrame to be profiled.
        summary_level: The level of detail in the summary: 'summary' for selected statistics, 'detailed' for detailed summary.
    """
    # Data profiling results will be stored in this list
    summary_stats = []

    # Get the list of columns in the DataFrame
    columns = df.columns

    for column in columns:
        # Determine the variable type of the column
        variable_type = determine_variable_type(df, column)

        # Calculate summary statistics based on the variable type
        if variable_type == 'numeric':
            stats = calculate_numeric_summary(df, column, summary_level)
        elif variable_type == 'categorical':
            stats = calculate_categorical_summary(df, column)
        else:
            # Column has an unrecognized data type
            stats = {
                "Column": column,
                "Variable Type": None,
                "Count": df.count(),
                "Missing": df.select(column).filter(F.isnan(column) | F.col(column).isNull() | (F.col(column) == "")).count(),
                "Distinct Values": df.agg(F.countDistinct(column)).first()[0]
            }
            logging.warning(f"Unrecognized data type for column '{column}': {df.schema[column].dataType}")

        # Append the statistics to the summary_stats list
        summary_stats.append(stats)

    # Format the summary statistics for printing
    summary_str = format_summary(summary_stats)
    print(summary_str)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Profile a Spark DataFrame.")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the data file.")
    parser.add_argument("--summary_level", type=str, choices=["summary", "detailed"], default="summary", help="Level of detail in the summary.")
    parser.add_argument("--log_level", type=str, choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], default="INFO", help="Logging level (optional).")
    args = parser.parse_args()

    # Logging Configuration
    logging.basicConfig(level=args.log_level, format='%(levelname)s: %(message)s')

    # Input Validation
    if not args.data_path:
        logging.error("Data path not provided. Please provide the path to the data file using --data_path argument.")
        exit(1)

    # Initialize Spark session.
    spark = SparkSession.builder.appName("Data Profile").getOrCreate()

    # Load the data.
    df = spark.read.csv(args.data_path, inferSchema=True, header=True)

    # Profile the data.
    profile_df(df, summary_level=args.summary_level)

    # Stop Spark session after usage
    spark.stop()
