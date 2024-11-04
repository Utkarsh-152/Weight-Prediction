import sys
import pandas as pd
from src.exceptions import CustomException
from src.logger import logging
from src.utils import load_object

class PredictPipeline:
    def __init__(self):
        pass
    def predict(self, features):
        try:
            model_path = 'artifacts/model.pkl'
            preprocessor_path = 'artifacts\preprocessor.pkl'
            model = load_object(file_path = model_path)
            preprocessor = load_object(file_path = preprocessor_path)
            data_scaled = preprocessor.transform(features)
            preds = model.predict(data_scaled)
            return preds
        except Exception as e:
            raise CustomException(e,sys)


class CustomData:
    def __init__(  self,
        Gender:str, 
        PhysicalActivityLevel: str, 
        SleepQuality: str,
        Age: int,
        CurrentWeight: int,
        BMR: int,
        CaloricSurplusOrDeficit: int,
        DailyCaloriesConsumed: int,
        Duration: int,
        StressLevel: int,
        WeightChange: int):

        self.Gender = Gender

        self.PhysicalActivityLevel = PhysicalActivityLevel

        self.SleepQuality = SleepQuality

        self.Age = Age

        self.BMR = BMR

        self.CaloricSurplusOrDeficit = CaloricSurplusOrDeficit

        self.DailyCaloriesConsumed = DailyCaloriesConsumed

        self.Duration = Duration

        self.CurrentWeight = CurrentWeight

        self.StressLevel = StressLevel
        self.WeightChange = WeightChange

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "Gender": [self.Gender],
                "PhysicalActivityLevel": [self.PhysicalActivityLevel],
                "SleepQuality": [self.SleepQuality],
                "Age": [self.Age],
                "DailyCaloriesConsumed": [self.DailyCaloriesConsumed],
                "BMR": [self.BMR],
                "CaloricSurplusOrDeficit": [self.CaloricSurplusOrDeficit],
                "Duration": [self.Duration],
                "CurrentWeight": [self.CurrentWeight],
                "StressLevel": [self.StressLevel],
                "WeightChange": [self.WeightChange]
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)
        