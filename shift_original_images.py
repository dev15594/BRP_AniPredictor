from dependencies import *

class ShiftOriginals():
    def __init__(self) -> None:
        pass
    
    def get_df_dirs(self, camera_folder):
        df_dirs = []
        sub_dirs = os.listdir(camera_folder)
        for dir in sub_dirs:
            dir_path = os.path.join(camera_folder, dir)
            df_dir = os.path.join(dir_path, "predictions.csv")
            if os.path.exists(df_dir):
                df_dirs.append(df_dir)
        return df_dirs
    
    def run(self, dst_dir, animal_list):
        df_dirs = self.get_df_dirs(dst_dir)

        for df_dir in df_dirs:
            # count = 0
            parent_path = os.path.dirname(df_dir)
            for animal in animal_list:
                new_animal_folder = os.path.join(parent_path, animal)
                if not os.path.exists(new_animal_folder):
                    os.makedirs(new_animal_folder)
                cropped_animal_dir = os.path.join(parent_path, "Cropped_images", animal)
                if os.path.exists(cropped_animal_dir):
                    images = os.listdir(cropped_animal_dir)
                    for image in images:
                        image_name = "_".join(image.split("_")[:-1])
                        img_path = os.path.join(parent_path, f"{image_name}.jpg")
                        new_img_path = os.path.join(new_animal_folder, f"{image_name}.jpg")
                        # print(img_path, new_img_path)
                        if os.path.exists(img_path):
                            # print(img_path, new_img_path)
                            # count += 1
                            if not os.path.exists(new_img_path):
                                shutil.copy(src=img_path,
                                            dst=new_img_path)
                            # print(img_path, new_img_path)
                            continue
                        else:
                            print(f"IMG NOT FOUND - {img_path}")
        return
    