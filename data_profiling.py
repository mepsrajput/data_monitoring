import argparse
import pyspark.sql.functions as F
from tabulate import tabulate

def determine_variable_type(df, column):
    """
    Determines the variable type (numeric or categorical) for a given column.

    Args:
        df: The Spark DataFrame.
        column: The name of the column.

    Returns:
        The variable type as a string: 'numeric' or 'categorical'.
    """
    distinct_count = df.select(column).distinct().count()
    if distinct_count <= 20:
        return 'categorical'
    else:
        return 'numeric'

def calculate_numeric_stats(df, column, summary_level, progress):
    """
    Calculates statistics for a numeric column.

    Args:
        df: The Spark DataFrame.
        column: The name of the column.
        summary_level: The level of detail in the summary: 'summary' for selected statistics, 'detailed' for detailed summary.
        progress: A reference to the progress variable.

    Returns:
        A dictionary containing the statistics.
    """
    count = df.select(column).count()
    missing_count = df.select(column).filter(df[column].isNull() | F.isnan(column)).count()
    percentiles = [0.05, 0.5, 0.75]

    if summary_level == 'summary':
        selected_stats = ['Count', 'Missing', 'Min'] + [f'P{int(p * 100)}' for p in percentiles] + ['Max']
    else:
        selected_stats = ['Count', 'Missing', 'Min'] + [f'P{int(p * 100)}' for p in percentiles] + ['Max', 'Mean', 'Stddev', 'Variance', 'Skewness', 'Kurtosis', 'Outliers']

    stats = {
        "Column": column,
        "Variable Type": 'numeric'
    }

    total_stats = len(selected_stats)
    completed_stats = 0

    for stat_name in selected_stats:
        if stat_name == 'Count':
            stats[stat_name] = count
        elif stat_name == 'Missing':
            stats[stat_name] = missing_count
        elif stat_name == 'Min':
            stats[stat_name] = df.select(column).min()
        elif stat_name == 'Max':
            stats[stat_name] = df.select(column).max()
        elif stat_name.startswith('P'):
            percentile = float(stat_name[1:]) / 100
            stats[stat_name] = df.approxQuantile(column, [percentile], 0.01)[0]
        else:
            stats[stat_name] = df.select(column).agg(getattr(F, stat_name.lower())(column)).first()[0]

        completed_stats += 1
        progress[0] = completed_stats / total_stats

    return stats

def calculate_categorical_stats(df, column, progress):
    """
    Calculates statistics for a categorical column.

    Args:
        df: The Spark DataFrame.
        column: The name of the column.
        progress: A reference to the progress variable.

    Returns:
        A dictionary containing the statistics.
    """
    count = df.select(column).count()
    missing_count = df.select(column).filter(df[column].isNull() | (df[column] == "")).count()
    stats = {
        "Column": column,
        "Variable Type": 'categorical',
        "Count": count,
        "Missing": missing_count,
        "Distinct Values": df.select(column).distinct().count(),
        "Mode": df.groupBy(column).count().orderBy(F.desc("count")).first()[column]
    }

    progress[0] += 1 / len(df.columns)

    return stats

# Rest of the code...

def profile_df(df, summary_level):
    """
    This function profiles a Spark DataFrame and prints the output with formatting.

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
    progress = [0]  # Variable to track progress
    for column in numeric_vars:
        numeric_stats.append(calculate_numeric_stats(df, column, summary_level, progress))

    # Calculate statistics for categorical variables.
    categorical_stats = []
    for column in categorical_vars:
        categorical_stats.append(calculate_categorical_stats(df, column, progress))

    # Format the statistics for printing.
    formatted_numeric_stats = format_stats(numeric_stats)
    formatted_categorical_stats = format_stats(categorical_stats)

    # Print the statistics.
    print("Distribution of Numeric Variables")
    print("---------------------------------")
    print(formatted_numeric_stats)
    print()
    print("Distribution of Categorical Variables")
    print("-------------------------------------")
    print(formatted_categorical_stats)

if __name__ == "__main__":
    # Rest of the code...
