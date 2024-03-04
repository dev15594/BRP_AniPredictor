from tkinter import *
import tkinter
import customtkinter
from tkinter import filedialog
from pathlib import Path
import os 
import shutil 
import tensorflow as tf
import cv2
from subprocess import Popen, PIPE 
import sys 
import re
import subprocess
import time
import pandas as pd
import json
import numpy as np
from PIL import Image

customtkinter.set_appearance_mode("dark")  # Modes: "System" (standard), "Dark", "Light"
customtkinter.set_default_color_theme("green")  # Themes: "blue" (standard), "green", "dark-blue"
    
class App(customtkinter.CTk):

    WIDTH = 730
    HEIGHT = 750

    def __init__(self):
        super().__init__()

        self.title("AniPredictor.py")
        self.geometry(f"{App.WIDTH}x{App.HEIGHT}")
        self.eval('tk::PlaceWindow . center')
        self.protocol("WM_DELETE_WINDOW", self.on_closing)

        self.frame = customtkinter.CTkFrame(master=self)
        self.frame.grid(row=0, column=0, padx = 20, pady = 20, sticky=E+W+N+S)
        self.columnconfigure(0, weight = 1)
        self.rowconfigure(0, weight = 1)

        self.model_frame = customtkinter.CTkFrame(master = self.frame)
        self.model_frame.grid(row=8, column=0, padx = 20, pady = 20, columnspan = 6)

        # self.frame.columnconfigure(0, weight = 0, minsize = 20)
        # self.frame.columnconfigure(1, weight = 0)
        # self.frame.columnconfigure(2, weight = 0)
        # self.frame.columnconfigure(3, weight = 0)
        
        self.img_path = StringVar()
        self.data_type = IntVar(value=0)
        self.frames = StringVar(value= "0")
        self.model_choice = IntVar(value = 0)

        self.heading = customtkinter.CTkLabel(master=self.frame, text = "AniPredictor.py", font=("Roboto Medium", -25))
        self.heading.grid(row = 0, column = 0, columnspan = 6, pady = 20)
        
        ##------------------------------------------------------- Check Environment -------------------------------------------------------

        self.check_env_btn = customtkinter.CTkButton(master=self.frame,
                                                text="Check Environment",
                                                font=("Roboto Medium", -15), 
                                                command = self.check_env)
        self.check_env_btn.grid(row=1, column=0, columnspan = 6, sticky = E+W, padx = 200)

        self.check_env_response = customtkinter.CTkLabel(master=self.frame, text = "")
        self.check_env_response.grid(row = 2, column = 0, columnspan = 6)

        ##----------------------------------------------------------- Photo or video -----------------------------------------------------------

        self.radio_label = customtkinter.CTkLabel(master=self.frame, text="Select Data Type", font=("Roboto Medium", -18))
        self.radio_label.grid(row = 3, column = 2, pady = 15, sticky = E)

        self.radio_btn_1 = customtkinter.CTkRadioButton(master=self.frame,
                                                        variable=self.data_type,
                                                        value=0,
                                                        text = "Photos",
                                                        command = self.photos)
        self.radio_btn_1.grid(row=3, column=3, sticky=E, pady = 8)

        self.radio_btn_2 = customtkinter.CTkRadioButton(master=self.frame,
                                                        variable=self.data_type,
                                                        value=1,
                                                        text = "Videos",
                                                        command = self.videos)
        self.radio_btn_2.grid(row=3, column=4, sticky=W, pady = 8)

        ## ------------------------------------------------------------- Path ---------------------------------------------------------

        self.img_label = customtkinter.CTkLabel(master=self.frame, text = "Enter Path to Data : ",font=("Roboto Medium", -18))
        self.img_label.grid(row = 6, column = 0, padx = 20, pady = 15, sticky = E, columnspan = 2)

        self.img_entry = customtkinter.CTkEntry(master=self.frame,
                                                placeholder_text="PATH",
                                                textvariable=self.img_path)

        self.img_entry.grid(row=6, column=2, sticky=E+W, pady = 15, columnspan = 2)

        self.img_btn = customtkinter.CTkButton(master=self.frame,
                                                text="Browse",
                                                font=("Roboto Medium", -12),
                                                command = lambda: self.browse())
        self.img_btn.grid(row=6, column = 4, pady = 15, columnspan = 2)

        ##----------------------------------------------------------- Select Model -----------------------------------------------------------

        self.model_label = customtkinter.CTkLabel(master=self.model_frame, text = "Select the Desired Model : ",font=("Roboto Medium", -18))
        self.model_label.grid(row = 0, column = 0, padx = 25, pady = 10, sticky = E, columnspan = 2)

        self.megadetector_model = customtkinter.CTkRadioButton(master=self.model_frame,
                                                               variable=self.model_choice,
                                                               value=0,
                                                               text = "Megadetector (Animal, Vehicle, Human)",
                                                               font=("Roboto Medium", -18))
        self.megadetector_model.grid(row=0, column=2, sticky=E+W, columnspan = 2, padx = 10, pady = 10)

        self.order_level_model = customtkinter.CTkRadioButton(master=self.model_frame,
                                                               variable=self.model_choice,
                                                               value=1,
                                                               text = "Order Level (9 classes)",
                                                               font=("Roboto Medium", -18))
        self.order_level_model.grid(row=1, column=2, sticky=E+W, columnspan = 2, padx = 10, pady = 10)

        self.order_level_model = customtkinter.CTkRadioButton(master=self.model_frame,
                                                               variable=self.model_choice,
                                                               value=2,
                                                               text = "Complete (14 classes)",
                                                               font=("Roboto Medium", -18))
        self.order_level_model.grid(row=2, column=2, sticky=E+W, columnspan = 2, padx = 10, pady = 10)

        self.order_level_model = customtkinter.CTkRadioButton(master=self.model_frame,
                                                               variable=self.model_choice,
                                                               value=3,
                                                               text = "GIB Identification",
                                                               font=("Roboto Medium", -18))
        self.order_level_model.grid(row=3, column=2, sticky=E+W, columnspan = 2, padx = 10, pady = 10)

        ##------------------------------------------------------ Predict Button ----------------------------------------------------------

        self.predict_btn = customtkinter.CTkButton(master=self.frame,
                                                text="Predict",
                                                font=("Roboto Medium", -18),
                                                command = lambda: self.predict(Path(self.img_path.get()),
                                                                               int(self.data_type.get()),
                                                                               int(self.model_choice.get()),
                                                                               int(self.frames.get())))
        self.predict_btn.grid(row=13, column = 2, pady=20, padx = 30, columnspan = 2, sticky = E+W)

        ##--------------------------------------------------- Responsive Messages --------------------------------------------------------

        self.output_label = customtkinter.CTkLabel(master=self.frame, text = "", font=("Roboto Medium", -18))
        self.output_label.grid(row = 14, column = 0, columnspan = 3)

        self.progress_bar_label = customtkinter.CTkLabel(master=self.frame, text = "", font=("Roboto Medium", -15))
        self.progress_bar_label.grid(row = 15, column = 0, columnspan = 3)


    ## ----------------------------------------------------------------------- End of UI ----------------------------------------------------------

    ## ----------------------------------------------------------------------- Functions ----------------------------------------------------------
    
    def photos(self):
        if self.data_type.get() == 0:
            self.frames_label.grid_forget()
            self.frames_dropdown.grid_forget()
        return

    def videos(self):
        if self.data_type.get() == 1:
            self.frames_label = customtkinter.CTkLabel(master=self.frame, text = "Number of frames in 1 minute : ", font=("Roboto Medium", -18))
            self.frames_label.grid(row = 5, column = 1, pady = 5, columnspan = 2)

            self.frames_dropdown = customtkinter.CTkOptionMenu(master=self.frame,
                                                               values=["1", "10", "20", "30", "60", "120"],
                                                               variable=self.frames,
                                                               font = ("Roboto Medium", -12))
            self.frames_dropdown.grid(row = 5, column = 3, columnspan = 2, sticky = E+W, padx = 40)
        else:
            self.frames_label.grid_forget()
            self.frames_dropdown.grid_forget()
    
    def browse(self):
        self.filename = filedialog.askdirectory()
        print(self.filename)
        self.img_path.set(self.filename)
        return
        
    def check_env(self):
        if len(tf.config.list_physical_devices("GPU"))>0:
            self.check_env_response.configure(text = "GPU Available :))")
        else:
            self.check_env_response.configure(text = "GPU Unavailable :((")
        return 
    
    def check_existing_file(self, img_dir):
        output_file_path = os.path.join(img_dir, "output.json")
        if os.path.exists(output_file_path):
            return 1
        return 0

    ## --------------------------------------------------------Predict function -----------------------------------------------------

    def run_models(self, data_dir, model_choice, order_level_model, ungulates_model, small_carnivores_model, gib_model):
        s2 = time.time()
        num_images = 0
        for _,_,files in os.walk(data_dir):
            num_images += len(files) 
        # json_dir = r"D:\WII\Experiment\output.json"
        json_dir = self.megadetector(data_dir, num_images)
        df = self.get_detection_df(data_dir, json_dir)

        if model_choice == 0:
            self.create_output_files_model1(data_dir, df)
        elif model_choice == 1:
            df = self.create_output_files_model2(data_dir, df, order_level_model)
            self.shift_original_images_model2(data_dir, df)
        elif model_choice == 2:
            df = self.create_output_files_model2(data_dir, df, order_level_model)
            new_df = self.create_output_files_model3(data_dir, df, ungulates_model, small_carnivores_model)
            # self.shift_original_images_model3(data_dir, new_df)
        else:
            new_df = self.create_output_files_gib_model(data_dir, df, gib_model)
            self.shift_gib_images(data_dir, new_df)


        e2 = time.time()
        print(f"Time taken to predict {num_images} images : {round(e2-s2, 2)} secs")
        self.output_label.configure(text = "")
        self.output_label.configure(text = f"Classification Complete...\nTime taken : {round(e2-s2, 2)} secs\nData saved in detections.csv")

        return

    def predict(self, data_dir, data_type, model_choice, num_frames):
        print(f"Data dir : {data_dir}")
        print(f"Data type : {data_type}")
        print(f"Model Choice : {model_choice}")
        print(f"Number of frames : {num_frames}")

        model_path = r"Models\Refined_Hierarchical.ckpt"
        ungulates_model_path = r"Models\Efficient_Net_Ungulates_3.ckpt"
        small_carnivores_model_path = r"Models\Efficient_Net_Small_Carnivores_1.ckpt"
        gib_model_path = r"Models\GIB_Only.ckpt"

        self.output_label.configure(text = f"Loading Models..")
        self.progress_bar_label.configure(text = "")
        self.update()

        if model_choice == 3:
            gib_model = tf.keras.models.load_model(gib_model_path)
        elif model_choice == 2:
            order_level_model = tf.keras.models.load_model(model_path)
            ungulates_model = tf.keras.models.load_model(ungulates_model_path)
            small_carnivores_model = tf.keras.models.load_model(small_carnivores_model_path)
        elif model_choice == 1:
            order_level_model = tf.keras.models.load_model(model_path)

        if data_type == 1:
            videos = os.listdir(data_dir)
            for video in videos:
                video_name = video.split(".")[0]
                video_path = os.path.join(data_dir, video)
                frames_dir = self.split_video_to_frames(data_dir, video_path, num_frames, video_name)
                if model_choice == 3:
                    self.run_models(frames_dir, model_choice, "", "", "", gib_model=gib_model)
                elif model_choice == 0:
                    self.run_models(frames_dir, model_choice, order_level_model, ungulates_model, small_carnivores_model, gib_model="")
                else:
                    self.run_models(frames_dir, model_choice, "", "", "", "", "")

        if data_type == 0:
            if model_choice == 3:
                self.run_models(data_dir, model_choice, "", "", "", gib_model=gib_model)
            elif not model_choice == 0:
                self.run_models(data_dir, model_choice, order_level_model, ungulates_model, small_carnivores_model, gib_model="")
            else:
                self.run_models(data_dir, model_choice, "", "", "", "")
        return
    
    ## --------------------------------------------------- Megadetector functions -----------------------------------------------------
    
    def megadetector(self, img_dir, num_images):
        print("Megadetector model")
        self.output_label.configure(text = "")
        self.progress_bar_label.configure(text = "")
        self.output_label.configure(text = f"Searching for objects...")
        self.update()

        
        # local_detector_path = os.path.join(r"D:\WII\GIT\BRP_AniPredictor\cameratraps\detection\run_detector_batch.py")
        # megadetector_path = os.path.join(r"D:\WII\GIT\BRP_AniPredictor\md_v5a.0.0.pt")

        local_detector_path = os.path.join(os.getcwd(), r"cameratraps\\detection\\run_detector_batch.py")
        megadetector_path = os.path.join(os.getcwd(), "md_v5a.0.0.pt")
        json_dir = os.path.join(img_dir, "output.json")

        if self.check_existing_file(img_dir) == 1:
            # self.output_label.configure(text = f"Megadetector output file already exists.. Going for species classification")
            print("Megadetector output file already exists.. Going for species classification")
            return json_dir
        
        print(local_detector_path, megadetector_path, json_dir)

        command = [sys.executable,
                   local_detector_path,
                   megadetector_path,
                   img_dir,
                   json_dir,
                   "--recursive"]

        with Popen(command,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=1, shell=True,
                universal_newlines=True) as p:
            for line in p.stdout:
                if line.startswith("Loaded model in"):
                    self.output_label.configure(text = "")
                    self.output_label.configure(text = f"Predicting objects in {num_images} images...\n")
                    self.update()
                
                elif "%" in line[0:4]:
                    percentage = re.search("\d*%", line[0:4])[0][:-1]
                    self.progress_bar_label.configure(text = f"{percentage}%")
                    self.update()
                    
                print(line)
        self.output_label.configure(text = "")
        self.output_label.configure(text = "Bounding Boxes Created...")
        self.progress_bar_label.configure(text = "")
        self.update()

        return json_dir
    
    ## ---------------------------------------------------- Get detections function ---------------------------------------------------

    def get_detection_df(self, img_path, json_dir):

        self.output_label.configure(text = "")
        self.output_label.configure(text = "Generating detections.csv...")
        self.update()
        
        with open(json_dir,'r+') as f:
            data = json.load(f)
            df = pd.DataFrame(data["images"])

        file_paths = []
        file_names = []
        categories = []
        d_confs = []
        d_bboxs = []
        d_nums = []
        for i in range(len(df)):
            percentage = int(((i+1)/len(df)) * 100)
            self.progress_bar_label.configure(text = "")
            self.progress_bar_label.configure(text = f"{percentage}%")
            self.update()
            # print(f"{percentage}%")

            filepath = df.iloc[i]["file"]
            filename = (filepath.split("\\")[-1]).split(".")[0] 
            detections = df.iloc[i]["detections"]
            for j, detection in enumerate(detections):
                d_nums.append(j+1)
                file_paths.append(filepath)
                file_names.append(filename)
                d_confs.append(detection["conf"])
                d_bboxs.append(detection["bbox"])

                area = detection["bbox"][2] * detection["bbox"][3]
                if area <= 0.001:
                    categories.append("Small Object")
                elif detection["category"] == '1':
                    categories.append("Animal")
                elif detection["category"] == '2':
                    categories.append("Person")
                else:
                    categories.append("Vehicle")

        new_df = pd.DataFrame({"Filepath" : file_paths,
                            "Filename" : file_names,
                            "Detection number" : d_nums,
                            "Category" : categories,
                            "Detection Confidence" : d_confs,
                            "Detection bbox" : d_bboxs})

        new_df = new_df[new_df["Detection Confidence"] > 0.1]
        new_df = new_df[new_df["Category"] != "Small Object"]
        new_df.reset_index(inplace = True)

        small_obj_df = new_df[new_df["Category"] == "Small Object"]
        if len(small_obj_df) > 0:
            small_obj_df.reset_index(inplace = True)
            small_obj_df_path = os.path.join(img_path, "small_objects.csv")
            small_obj_df.to_csv(small_obj_df_path, index = False)
            
        df_path = os.path.join(img_path, "detections.csv")
        new_df.to_csv(df_path, index = False)
        return new_df
    
    ## -------------------------------------------------------- Model 1 functions ---------------------------------------------------

    def create_output_files_model1(self, img_path, df):
        print("Shifting original images based on detections...")
        self.shift_original_images_model1(img_path, df)

        json_obj = df.to_json(orient='table',double_precision=4, indent = 3, index = False)

        with open(os.path.join(img_path, "detections.json"), 'w') as f:
            f.write(json_obj)
        return

    def shift_original_images_model1(self, img_path, df):
        for i in range(len(df)):
            filepath = Path(df.iloc[i]["Filepath"])
            img_folder = os.path.dirname(filepath)
            filename = filepath.name
            if df.iloc[i]["Detection Confidence"] >= 0.8:
                species = df.iloc[i]["Category"]
            else:
                species = "Others"
                
            cat_dir = os.path.join(img_folder, species)
            if not os.path.exists(cat_dir):
                os.makedirs(cat_dir)
            new_img_path = os.path.join(cat_dir, filename)
            shutil.copy(src=filepath,
                        dst=new_img_path) 
        return
    
    ## -------------------------------------------------------- Model 2 functions ---------------------------------------------------

    def crop_img(self, img_dir, bbox):
        img = Image.open(img_dir)
        x,y,w,h = tuple(i for i in bbox)
        mul_x = img.size[0]
        mul_y = img.size[1]
        w = w * mul_x
        h = h * mul_y
        x1 = x * mul_x
        x2 = x * mul_x + w
        y1 = y * mul_y
        y2 = y * mul_y + h
        cropped = img.crop((x1,y1,x2, y2))
        return cropped

    def create_output_files_model2(self, img_path, df, order_level_model):
        print("Order Level Model")
        # cropped_dir = os.path.join(img_path, "Cropped_images")
        # if not os.path.exists(cropped_dir):
        #     os.makedirs(cropped_dir)

        IMG_SIZE = (224,224)

        order_level_class_names = ["GIB", "Goat_Sheep", "Hare", "Human", "Raptor", "Small Bird", "Small Carnivore", "Ungulate", "Vehicle", "Wild Pig"]
        order_level_class_names.sort()

        order_level_categories = []
        pred_probs = []
        cropped_paths = []
        # groups = df.groupby(by = "Filepath")
        # for group in groups: 
        #     filepath = Path(group[0])
        #     img_dir = filepath.parent
        #     filename = filepath.name
        #     for i in range(len(group[1])):
        for i in range(len(df)):

            filepath = Path(df.iloc[i]["Filepath"])
            img_folder = os.path.dirname(filepath)
            filename = filepath.name

            d_num = df.iloc[i]["Detection number"]
            conf = df.iloc[i]["Detection Confidence"]
            bbox = df.iloc[i]["Detection bbox"]
            category = df.iloc[i]["Category"]
            # d_num = group[1].iloc[i]["Detection number"]
            # conf = group[1].iloc[i]["Detection confidence"]
            # bbox = group[1].iloc[i]["Bounding box coordinates"]
            
            
            # print(conf, "-", bbox)
            cropped_img = self.crop_img(filepath, bbox)
            cropped_img_1 = np.array(cropped_img)
            cropped_img_1 = tf.image.resize(cropped_img_1, 
                                        size= IMG_SIZE,
                                        method = "area")
            cropped_img_1 = tf.expand_dims(cropped_img_1, axis=0)
            pred = order_level_model.predict(cropped_img_1)
            pred = np.squeeze(pred)
            # print(pred)
            order_level_pred_prob = round(max(pred), 2)
            pred_probs.append(order_level_pred_prob)
            if order_level_pred_prob >= 0.8:
                pred_class = order_level_class_names[np.argmax(pred)]
            else:
                pred_class = "Others"
            
            order_level_categories.append(pred_class)

            
            cropped_dir = os.path.join(img_folder, "Cropped_images", pred_class)
            if not os.path.exists(cropped_dir):
                os.makedirs(cropped_dir)

            cropped_img_path = os.path.join(cropped_dir, f"{d_num}_{conf}_{order_level_pred_prob}_{filename}")
            cropped_paths.append(cropped_img_path)
            cropped_img.save(cropped_img_path)
            


        # print(len(pred_probs), len(order_level_categories), len(cropped_paths))
        # print(len(df))

        df["Order pevel pred"] = pred_probs
        df["Order level class"] = order_level_categories
        df["Cropped image path"] = cropped_paths
        
        df_save_dir = os.path.join(img_path, "detections_1.csv")
        df.to_csv(df_save_dir, index = False)
    
        return df
    
    def shift_original_images_model2(self, img_dir, df):
        all_files_list = []
        for i in range(len(df)):
            order_level_class = df.iloc[i]["Order Level Class"]
            filename = df.iloc[i]["Filename"]
            filepath = Path(df.iloc[i]["Filepath"])
            img_folder = os.path.dirname(filepath)
            
            if not filepath in all_files_list:
                all_files_list.append(filepath)

            animal_folder = os.path.join(img_folder, order_level_class)
            if not os.path.exists(animal_folder):
                os.makedirs(animal_folder)
            
            new_img_path = os.path.join(animal_folder, f"{filename}.jpg")
            if not os.path.exists(new_img_path):
                shutil.copy(src = Path(filepath),
                            dst= new_img_path)

        for image in all_files_list:
            os.remove(image)    
        
        return

    ## -------------------------------------------------------- Model 3 functions ---------------------------------------------------
    def lower_level_predict(self, img_path, model):

        IMG_SIZE = (224,224)
        img = Image.open(img_path)
        img = np.array(img)
        img = tf.image.resize(img, 
                              size= IMG_SIZE,
                              method = "area")
        img = tf.expand_dims(img, axis=0)
        pred = model.predict(img)
        pred = np.squeeze(pred)

        return pred
    
    def create_output_files_model3(self, img_path, df, ungulates_model, small_carnivores_model):

        ungulates_class_names = ["Camel", "Chinkara", "Nilgai", "Cattle"]
        ungulates_class_names.sort()

        small_carnivores_class_names = ["Dog", "Desert Cat", "Fox"]
        small_carnivores_class_names.sort()
        
        lower_level_preds = []
        lower_level_classes = []
        
        for i in range(len(df)):
            order_level_class = df.iloc[i]["Order level class"]
            cropped_img_path = df.iloc[i]["Cropped image path"]
            

            self.output_label.configure(text = "")
            self.output_label.configure(text = "Classifying animals...")
            self.update()

            if order_level_class == "Ungulate":
                pred = self.lower_level_predict(cropped_img_path, ungulates_model)
                max_pred = round(max(pred), 2)
                pred_class = ungulates_class_names[np.argmax(pred)]
            elif order_level_class == "Small Carnivore":
                pred = self.lower_level_predict(cropped_img_path, small_carnivores_model)
                max_pred = round(max(pred), 2)
                pred_class = small_carnivores_class_names[np.argmax(pred)]
            else:
                pred_class = order_level_class
                max_pred = 0
            
            lower_level_preds.append(max_pred)
            lower_level_classes.append(pred_class)

        # print(lower_level_preds)
        # print(lower_level_classes)

        df["Lower level pred"] = lower_level_preds
        df["Lower level class"] = lower_level_classes

        df_save_dir = os.path.join(img_path, "detections_2.csv")
        df.to_csv(df_save_dir, index = False)

        return df
    
    def shift_original_images_model3(self, img_dir, df):
        all_files_list = []
        for i in range(len(df)):
            order_level_class = df.iloc[i]["Lower level class"]
            filename = df.iloc[i]["Filename"]
            filepath = Path(df.iloc[i]["Filepath"])
            img_folder = os.path.dirname(filepath)

            if not filepath in all_files_list:
                all_files_list.append(filepath)

            animal_folder = os.path.join(img_folder, order_level_class)
            if not os.path.exists(animal_folder):
                os.makedirs(animal_folder)
            
            new_img_path = os.path.join(animal_folder, f"{filename}.jpg")
            if not os.path.exists(new_img_path):
                shutil.copy(src = Path(filepath),
                            dst= new_img_path)
        
        for image in all_files_list:
            os.remove(image)

        return
        
    ## ------------------------------------------------------- GIB only Model -------------------------------------------------------

    def create_output_files_gib_model(self, img_path, df, gib_model):
        print("GIB Only Model")
        # cropped_dir = os.path.join(img_path, "Cropped_images")
        # if not os.path.exists(cropped_dir):
        #     os.makedirs(cropped_dir)

        IMG_SIZE = (224,224)

        class_names = ["GIB", "Others"]
        class_names.sort()

        categories = []
        pred_probs = []
        cropped_paths = []
        # groups = df.groupby(by = "Filepath")
        # for group in groups: 
        #     filepath = Path(group[0])
        #     img_dir = filepath.parent
        #     filename = filepath.name
        #     for i in range(len(group[1])):
        for i in range(len(df)):

            filepath = Path(df.iloc[i]["Filepath"])
            img_folder = os.path.dirname(filepath)
            filename = filepath.name

            d_num = df.iloc[i]["Detection number"]
            conf = df.iloc[i]["Detection Confidence"]
            bbox = df.iloc[i]["Detection bbox"]
            # d_num = group[1].iloc[i]["Detection number"]
            # conf = group[1].iloc[i]["Detection confidence"]
            # bbox = group[1].iloc[i]["Bounding box coordinates"]
            
            
            # print(conf, "-", bbox)
            cropped_img = self.crop_img(filepath, bbox)
            cropped_img_1 = np.array(cropped_img)
            cropped_img_1 = tf.image.resize(cropped_img_1, 
                                        size= IMG_SIZE,
                                        method = "area")
            cropped_img_1 = tf.expand_dims(cropped_img_1, axis=0)
            pred = gib_model.predict(cropped_img_1)
            pred = np.squeeze(pred)
            # print(pred)
            pred_probs.append(pred)
            pred_class = class_names[int(np.round(pred))]
            

            categories.append(pred_class)
            cropped_dir = os.path.join(img_folder, "Cropped_images", pred_class)
            if not os.path.exists(cropped_dir):
                os.makedirs(cropped_dir)

            cropped_img_path = os.path.join(cropped_dir, f"{d_num}_{conf}_{np.round(pred)}_{filename}")
            cropped_paths.append(cropped_img_path)
            cropped_img.save(cropped_img_path)
            


        # print(len(pred_probs), len(categories), len(cropped_paths))
        # print(len(df))

        df["Pred Prob"] = pred_probs
        df["Class"] = categories
        df["Cropped image path"] = cropped_paths
        
        df_save_dir = os.path.join(img_path, "gib_detections.csv")
        df.to_csv(df_save_dir, index = False)
    
        return df

    def shift_gib_images(self, img_dir, df):
        all_files_list = []
        for i in range(len(df)):
            class_name = df.iloc[i]["Class"]
            filename = df.iloc[i]["Filename"]
            filepath = Path(df.iloc[i]["Filepath"])
            img_folder = os.path.dirname(filepath)

            if not filepath in all_files_list:
                all_files_list.append(filepath)

            animal_folder = os.path.join(img_folder, class_name)
            if not os.path.exists(animal_folder):
                os.makedirs(animal_folder)
            
            new_img_path = os.path.join(animal_folder, f"{filename}.jpg")
            if not os.path.exists(new_img_path):
                shutil.copy(src = Path(filepath),
                            dst= new_img_path)

        for image in all_files_list:
            os.remove(image)    
        
        return

    ## ------------------------------------------------------- Video functions -------------------------------------------------------

    def split_video_to_frames(self, folder, path, frames, video_name):
        print("Function start")

        frames_dir = os.path.join(folder, video_name)
        if not os.path.exists(frames_dir):
            os.makedirs(frames_dir)

        video = cv2.VideoCapture(path)
        fps = int(round(video.get(cv2.CAP_PROP_FPS)))
        print(fps)
        idx = 0
        count = 1
        interval = (fps * 60)/frames
        while True:
            ret, frame = video.read()
            if ret == False:
                video.release()
                break
            if (idx %  interval == 0):
                print(count)
                cv2.imwrite(f"{frames_dir}\\{count}.jpg", frame)
                count += 1
            idx += 1
        print("Frames extracted...")
        return frames_dir

    def on_closing(self, event=0):
        self.destroy()
    
if __name__ == "__main__":
    app = App()
    app.mainloop()