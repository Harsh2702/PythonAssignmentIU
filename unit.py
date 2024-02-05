import unittest
import pandas as pd
import numpy as np
from assignment import DataProcessorBase  # Replace 'your_module' with the actual module name
database_url = "sqlite:///database.db"
class TestDataProcessorBase(unittest.TestCase):

    def setUp(self):
        # Set up sample data for testing
        self.sample_train_data = pd.DataFrame({
            'A': [1, 2, 3],
            'B': [4, 5, 6],
            'C': [7, 8, 9]
        })

        self.sample_ideal_data = pd.DataFrame({
            'X': [10, 11, 12],
            'Y': [13, 14, 15],
            'Z': [16, 17, 18]
        })

        self.sample_test_data = pd.DataFrame({
            'T': [19, 20, 21],
            'U': [22, 23, 24],
            'V': [25, 26, 27]
        })

    def test_preprocess_data(self):
        processor = DataProcessorBase("", "", database_url )
        processed_data = processor.preprocess_data(self.sample_train_data)
        self.assertTrue(isinstance(processed_data, pd.DataFrame))

    def test_calculate_squared_errors(self):
        processor = DataProcessorBase("", "", database_url )
        errors = processor.calculate_squared_errors([1, 2, 3], [4, 5, 6])
        self.assertEqual(errors, 27)

    def test_find_best_fit_column(self):
        processor = DataProcessorBase("", "", database_url )
        normalized_train = processor.preprocess_data(self.sample_train_data)
        normalized_ideal = processor.preprocess_data(self.sample_ideal_data)

        best_fit_indices = processor.find_best_fit_column(normalized_train, normalized_ideal)
        self.assertTrue(isinstance(best_fit_indices, np.ndarray))

    # Add more tests as needed

if __name__ == '__main__':
    unittest.main()
