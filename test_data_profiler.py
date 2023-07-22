import unittest
from data_profiler import profile_df
from pyspark.sql import SparkSession

class TestDataProfiling(unittest.TestCase):
    def setUp(self):
        self.spark = SparkSession.builder.appName("Test Data Profile").getOrCreate()

    def tearDown(self):
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
        |    Value | numeric         |       2 |         1 |   100 |   300 | 100  |   200 |   300 |
        """

        # Remove leading white spaces from the expected output (tabulate adds extra spaces)
        expected_output = "\n".join(line.lstrip() for line in expected_output.strip().split("\n"))

        # Compare the output with the expected output
        self.assertEqual(output.strip(), expected_output)


# Add more test cases as needed

if __name__ == "__main__":
    unittest.main()
