import os
import unittest

import joblib
import pandas as pd


class TestMLPipeline(unittest.TestCase):

    def test_processed_data(self):
        file_path = "data/processed.csv"

        self.assertTrue(
            os.path.exists(file_path),
            "processed.csv does not exist"
        )

        df = pd.read_csv(file_path)

        expected_columns = [
            "study_hours",
            "attendance",
            "previous_marks",
            "assignment_score",
            "result"
        ]

        self.assertEqual(list(df.columns), expected_columns)
        self.assertGreater(len(df), 0)

    def test_model_exists_and_predicts(self):
        model_path = "model/model.pkl"

        self.assertTrue(
            os.path.exists(model_path),
            "model.pkl does not exist"
        )

        model = joblib.load(model_path)

        df = pd.read_csv("data/processed.csv")

        X = df.drop("result", axis=1)

        predictions = model.predict(X)

        self.assertEqual(len(predictions), len(X))

        for prediction in predictions:
            self.assertIn(prediction, [0, 1])


if __name__ == "__main__":
    unittest.main()