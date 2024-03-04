from tkinter import *
import tkinter
import customtkinter
from tkinter import filedialog
from functions import *
import torch
from tqdm.notebook import tqdm

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
        self.areModelsLoaded = BooleanVar(value=False)
        self.log = {}


        self.heading = customtkinter.CTkLabel(master=self.frame, text = "AniPredictorV2", font=("Roboto Medium", -25))
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
        gpus = tf.config.experimental.list_physical_devices('GPU')

        if gpus:
            print("GPU available")
            self.check_env_response.configure(text = "GPU Available :))")
            self.gpu_name = torch.cuda.get_device_name()
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
            except:
                pass
        else:
            print("No GPUs available.")
            self.check_env_response.configure(text = "GPU Not Available :((")
    
    def load_models(self):

        print("Loading Models...")
        self.order_level_class_names = ["GIB", "Goat_Sheep", "Hare", "Human", "Raptor", "Small Bird", "Small Carnivore", "Ungulate", "Vehicle", "Wild Pig"]
        self.order_level_class_names.sort()
        self.ungulate_class_names = ["Camel", "Chinkara", "Nilgai", "Cattle"]
        self.ungulate_class_names.sort() 
        self.small_carnivores_class_names = ["Dog", "Desert Cat", "Fox"]
        self.small_carnivores_class_names.sort()
        
        model_load_start=time.time()
        if self.areModelsLoaded == False:
            order_level_model_path = os.path.join(os.getcwd(), r"Models\Refined_Hierarchical.ckpt")
            order_level_model = tf.keras.models.load_model(order_level_model_path)
            
            ungulate_model_path = os.path.join(os.getcwd(), r"Models\Efficient_Net_Ungulates_3.ckpt")
            ungulate_model = tf.keras.models.load_model(ungulate_model_path)   
            
            small_carnivore_model_path = os.path.join(os.getcwd(), r"Models\Efficient_Net_Small_Carnivores_1.ckpt")
            small_carnivore_model= tf.keras.models.load_model(small_carnivore_model_path)
            
            self.areModelsLoaded = True
            
        model_load_end = time.time()
        model_load_time = str(timedelta(seconds=round(model_load_end - model_load_start)))
        self.log.update({"Species Model Load Time" : model_load_time})
        print(model_load_time)

        return order_level_model, ungulate_model, small_carnivore_model
    
    def predict(self, data_dir, data_type, model_choice, num_frames):
        now = datetime.now()
        order_level_model, ungulate_model, small_carnivore_model = self.load_models()

        self.log.update({"Run timestamp" : str(now)})
        # log.update({"GPU" : gpus})
        self.log.update({"GPU Available for Classification : " : self.gpu_name})
        # log.update({"CPU" : cpu})
        num_images = 0
        for root,dirs,files in os.walk(data_dir):
            if not root == "Cropped_images":
                num_images += len(files)
                for f in files:
                    if not f.endswith(".jpg"):
                        num_images -= 1
        self.log.update({"Num images" : num_images})
        print(num_images)
        
        ## RUN MEGADETECTOR AND CREATE DETECTIONS.DF
        
        megadetector_start = time.time()
        json_dir, megadetector_log = megadetector(data_dir, num_images)
        if not megadetector_log == {}:
            self.log.update(megadetector_log)
        else:
            megadetector_end = time.time()
            megadetector_time = str(timedelta(seconds=round(megadetector_end - megadetector_start)))
            self.log.update({"Megadetector time" : megadetector_time})
            self.log.update({"Megadetector Filename" : os.path.basename(json_dir)})

        detections_dir = os.path.join(data_dir, "detections.csv")
        if not os.path.exists(detections_dir):
            df_detections = get_detection_df(data_dir, json_dir)
        else:
            print("Detections.csv exists...")
            df_detections = pd.read_csv(detections_dir)
        
        
        ## CROP IMAGES
        
        cropping_start = time.time() 
        cropped_images = os.path.join(data_dir,r"Cropped_images\*")
        cropped_dir = clean_path("\\".join(cropped_images.split("\\")[:-1]))
        if not os.path.exists(cropped_dir):
            print("Cropping Images")
            try:
                df_crop=crop_images_batch2(df_detections,512)
            except:
                df_crop = df_detections
                df_crop["Cropped_image_name"] = df_crop["Filename"] + "_" + df_crop["Detection_number"].astype(str) + ".jpg"
                #df_crop=crop_images_batch2(df_detections,512)
                print(f"Cropping exception occured")
            df_crop["Cropped_image_path"] = (cropped_dir + "\\" + df_crop["Cropped_image_name"]).apply(clean_path)

        else:
            print("Images already cropped...")
        cropping_end = time.time()
        cropping_time = str(timedelta(seconds=round(cropping_end - cropping_start)))
        self.log.update({"Cropping Time" : cropping_time})
        self.log.update({"Number of Detections" : len(df_detections)})
        
        ## ORDER LEVEL PREDICTIONS
        
        order_level_start = time.time()
        print("Predicting Order Level Classes...")
        df_temp, num_cropped = predict_lower_level_species(data_dir, 
                                                           r"Cropped_images\*", 
                                                           self.order_level_class_names,
                                                           order_level_model,
                                                           level = "Order")
        
        order_level_end = time.time()
        df_order = pd.merge(df_crop, df_temp, on='Cropped_image_name', how='left')
        df_order["Order_pred"] = df_order["Order_pred"].fillna("Error")
        df_order["Order_dir"] = (cropped_dir + "\\" + df_order["Order_pred"]).apply(clean_path)
        df_order["Order_level_path"] = (df_order["Order_dir"] + "\\" + df_order["Cropped_image_name"]).apply(clean_path)
        
        unique_directories = set(df_order['Order_dir'])
        for directory in unique_directories:
            os.makedirs(directory, exist_ok=True)
        
        print("Moving Order Level Images...")
        #copy_images_batch(df_order["Cropped_image_path"], df_order["Order_level_path"])
        #delete_images_batch(df_order["Cropped_image_path"])
        move_images_batch(df_order["Cropped_image_path"], df_order["Order_level_path"])
        
        order_shift_end = time.time()
        order_pred_time = str(timedelta(seconds=round(order_level_end - order_level_start)))
        order_shift_time = str(timedelta(seconds=round(order_shift_end - order_level_end)))
        self.log.update({"Order Level Prediction Time" : order_shift_time})
        self.log.update({"Order Level Shifting Time" : order_shift_time})
        
        
        ## SMALL CARNIVORES PREDICT
        small_carnivores_start = time.time()
        if os.path.exists(os.path.join(cropped_dir,r"Small_Carnivore")):
            print("Predicting Small Carnivores...")
            df_small_carnivore, num_small_carnivores = predict_lower_level_species(cropped_dir, 
                                                                                   r"Small Carnivore\*", 
                                                                                   self.small_carnivores_class_names,
                                                                                   small_carnivore_model,
                                                                                   level = "Species")
            
            small_carnivores_end = time.time()
            small_carnivore_time = str(timedelta(seconds=round(small_carnivores_end - small_carnivores_start)))
            self.log.update({"Number of Small Carnivores Images" : num_small_carnivores})
            self.log.update({"Small Carnivore Model Pred Time" : small_carnivore_time})
        else:
            df_small_carnivore = pd.DataFrame(columns=['Cropped_image_name','Species_pred','Species_pred_prob'])
       
        ## UNGULATES PREDICT

        if os.path.exists(os.path.join(cropped_dir,r"Ungulate")):
            ungulate_start = time.time()
            print("Predicting Ungulates...")
            df_ungulate, num_ungulates = predict_lower_level_species(cropped_dir, 
                                                                     r"Ungulate\*", 
                                                                     self.ungulate_class_names,
                                                                     ungulate_model,
                                                                     level = "Species")
            
            ungulate_end = time.time()
            ungulate_time = str(timedelta(seconds=round(ungulate_end - ungulate_start)))
            self.log.update({"Number of Ungulates Images" : num_ungulates})
            self.log.update({"Ungulate Model Pred Time" : ungulate_time})
        else:
            df_ungulate = pd.DataFrame(columns=['Cropped_image_name','Species_pred','Species_pred_prob'])
        
        species_shift_start = time.time()
        df_species = pd.concat([df_small_carnivore,df_ungulate])
        df_species["Species_dir"] = (cropped_dir + "\\" + df_species["Species_pred"]).apply(clean_path)
        df_species["Species_level_path"] = (df_species["Species_dir"] + "\\" + df_species["Cropped_image_name"]).apply(clean_path)
        
        df_move = pd.merge(df_species, df_order, on='Cropped_image_name', how='left')
        df_move = df_move[df_move["Order_level_path"] != df_move["Species_level_path"]]
        unique_directories = set(df_move['Species_dir'])
        for directory in unique_directories:
            os.makedirs(directory, exist_ok=True)
        
        print("Moving Ungulates and Small Carnivores...")
        #copy_images_batch(df_move["Order_level_path"], df_move["Species_level_path"])
        #delete_images_batch(df_move["Order_level_path"])
        move_images_batch(df_move["Order_level_path"], df_move["Species_level_path"])
        
        species_shift_end = time.time()
        species_shift_time = str(timedelta(seconds=round(species_shift_end - species_shift_start)))
        species_level_time = str(timedelta(seconds=round(species_shift_end - small_carnivores_start)))
        self.log.update({"Species Level Shift Imgs Time" : species_shift_time})
        self.log.update({"Species Level Predict and Shift" : species_level_time})
        
        ## SAVE FINAL PREDICTIONS.CSV
        
        df_final = pd.merge(df_order, df_species, on='Cropped_image_name', how='left')
        df_final.drop(columns=['Order_dir', 'Order_level_path','Cropped_image_path'], inplace=True)
        df_final_path = os.path.join(data_dir, "predictions.csv")
        df_final.to_csv(df_final_path, index=False)
        
        ## SAVE LOG FILE
        print("Saving Logs")
        log_file_name = "_".join(data_dir.split("\\")[-3:])
        log_file_path = os.path.join(data_dir, f"{log_file_name}_log.json")
        with open(log_file_path, "w") as f:
            json.dump(self.log, f, indent=2)

        return
    

    def on_closing(self, event=0):
        self.destroy()
        return


if __name__ == "__main__":
    app = App()
    app.mainloop()