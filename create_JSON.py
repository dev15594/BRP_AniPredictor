import pandas as pd 
import numpy as np 
import os 
import json
import ast

class CreateJSON:
    def __init__(self) -> None:
        pass
    def get_category_key(self, category, cat_dict):
        for key, val in cat_dict.items():
            if val == category:
                return str(key)
        return None

    def run(self, predictions_path, save_path, model_choice):
        pred_df = pd.read_csv(predictions_path)
        grouped_df = pred_df.groupby(['Filepath'])

        categories = ["Animal", "Human", "Vehicle"]
        categories += ["GIB", "Goat_Sheep", "Hare", "Raptor", "Small Bird", "Small Carnivore", "Ungulate", "Wild Pig"]
        categories += ["Camel", "Chinkara", "Nilgai", "Cattle"]
        categories += ["Dog", "Desert Cat", "Fox"]
        
        categories_dict = {}
        for i, item in enumerate(categories):
            categories_dict[f'{i}'] = item

        info = {"detection_completion_time": "2024-03-06 18:32:15",
        "format_version": "1.2",
        "detector": "md_v5a.0.0.pt",
        "detector_metadata": {
        "megadetector_version": "v5a.0.0",
        "typical_detection_threshold": 0.2,
        "conservative_detection_threshold": 0.7}}



        images = []
        for item in grouped_df:
            filepath = item[0]
            df = item[1]
            # print(type(df))
            detections = []
            for i in range(len(df)):
                max_conf = 0
                megadetector_pred = df["Detection_Confidence"].iloc[i]
                
                if model_choice == 2:
                    species_pred = df["Species_pred_prob"].iloc[i]
                    if not np.isnan(species_pred):
                        category = df["Species_pred"].iloc[i]
                    else:
                        species_pred = df["Order_pred_prob"].iloc[i]
                        category = df["Order_pred"].iloc[i]
                        
                    conf = species_pred
                elif model_choice in [1,3]:
                    order_pred = df["Order_pred_prob"].iloc[i]
                    category = df["Order_pred"].iloc[i]
                    conf = order_pred
                else:
                    category = df["Category"].iloc[i]
                    conf = megadetector_pred

                bbox = df["Detection_bbox"].iloc[i]
                bbox = ast.literal_eval(bbox)
                max_conf = max(max_conf, conf)
                category = self.get_category_key(category, categories_dict)
                animal_dict = {"category" : category, 
                            "conf" : conf,
                            "bbox" : bbox}
                # print(animal_dict)
                detections.append(animal_dict)

            # print(detections)
            image_dict = {"file" : filepath[0],
                        "max_detection_conf" : max_conf,
                        "detections" : detections}

            images.append(image_dict)


        json_dict = {"images" : images,
                "detections_categories" : categories_dict,
                "info" : info}

        with open(save_path, "w") as f: 
            json.dump(json_dict, f, indent = 4, separators=(',', ': '))
