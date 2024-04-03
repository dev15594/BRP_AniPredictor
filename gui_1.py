# from dependencies import *
# from gui_functs import *
from find_corrupt_files import *
from geotag import *
from rename_and_copy import *
from aniPredictor import *
from shift_original_images import *
from aniPred_helper import *
from create_JSON import *

# from Megadetector import *
customtkinter.set_appearance_mode("dark")  # Modes: "System" (standard), "Dark", "Light"
customtkinter.set_default_color_theme("green")  # Themes: "blue" (standard), "green", "dark-blue"

class ScrollableCheckBoxFrame(customtkinter.CTkScrollableFrame):
    def __init__(self, master, item_list, command=None, **kwargs):
        super().__init__(master, **kwargs)

        self.command = command
        self.checkbox_list = []
        for i, item in enumerate(item_list):
            self.add_item(item)

    def add_item(self, item):
        checkbox = customtkinter.CTkCheckBox(self, text=item)
        if self.command is not None:
            checkbox.configure(command=self.command)
        checkbox.grid(row=len(self.checkbox_list), column=0, pady=(0, 10))
        self.checkbox_list.append(checkbox)

    def remove_item(self, item):
        for checkbox in self.checkbox_list:
            if item == checkbox.cget("text"):
                checkbox.destroy()
                self.checkbox_list.remove(checkbox)
                return

    def get_checked_items(self):
        return [checkbox.cget("text") for checkbox in self.checkbox_list if checkbox.get() == 1]
    
    
class App(customtkinter.CTk):
    def __init__(self):
        super().__init__()
        # guiFuncts = GuiFunctions()

        self.title("AniPredictor.py")
        # self.geometry(f"{App.WIDTH}x{App.HEIGHT}")
        self.eval('tk::PlaceWindow . center')
        self.protocol("WM_DELETE_WINDOW", self.on_closing)

        self.frame = customtkinter.CTkFrame(master=self)
        # self.WIDTH = self.frame.winfo_screenwidth()
        # self.HEIGHT = self.frame.winfo_screenheight()

        self.frame.grid(row=0, column=0, padx = 20, pady = 20, sticky=E+W+N+S)
        self.columnconfigure(0, weight = 1)
        self.rowconfigure(0, weight = 1)

        self.copy_frame = customtkinter.CTkFrame(master = self.frame)
        self.copy_frame.grid(row=8, column=0, padx = 5, pady = 5, columnspan = 6)

        self.geotag_frame = customtkinter.CTkFrame(master = self.frame)
        self.geotag_frame.grid(row=12, column=0, padx = 5, pady = 5, columnspan = 6)

        self.model_frame = customtkinter.CTkFrame(master = self.frame)
        self.model_frame.grid(row=13, column=0, padx = 20, pady = 20, columnspan = 3)

        self.shift_images_frame = customtkinter.CTkFrame(master=self.frame)
        self.shift_images_frame.grid(row=13, column=3, pady=20, columnspan=3)
        

        self.log_frame = customtkinter.CTkFrame(master=self)
        self.log_frame.grid(row=0, column=6, padx=80, pady=150, columnspan=6, sticky=E+W+N+S)

        # self.frame.columnconfigure(0, weight = 0, minsize = 20)
        # self.frame.columnconfigure(1, weight = 0)
        # self.frame.columnconfigure(2, weight = 0)
        # self.frame.columnconfigure(3, weight = 0)
        
        self.img_path = StringVar()
        self.data_type = IntVar(value=0)
        self.frames = StringVar(value= "0")
        self.model_choice = IntVar(value = 0)
        self.dst_path = StringVar()
        self.latitude = StringVar()
        self.longitude = StringVar()
        self.camera = StringVar()
        self.station = StringVar()
        self.run_model = BooleanVar(value=True)
        self.doCopyAndRenaming = BooleanVar(value=True)
        self.doGeotagging = BooleanVar(value=True)
        self.doShifting = BooleanVar(value = False)
        self.heading = customtkinter.CTkLabel(master=self.frame, text = "AniPredictor.py", font=("Roboto Medium", -25))
        self.heading.grid(row = 0, column = 0, columnspan = 6, pady = 20)
        self.runModel = BooleanVar(value = False)

        
        self.species_list = ["Animal", "Human", "Vehicle"]
        
        self.order_level_class_names = ["GIB", "Goat_Sheep", "Hare", "Human", "Raptor", "Small Bird", "Small Carnivore", "Ungulate", "Vehicle", "Wild Pig"]
        self.order_level_class_names.sort()
        self.ungulate_class_names = ["Camel", "Chinkara", "Nilgai", "Cattle"]
        self.ungulate_class_names.sort() 
        self.small_carnivores_class_names = ["Dog", "Desert Cat", "Fox"]
        self.small_carnivores_class_names.sort()

        self.order_level_model_path = os.path.join(os.getcwd(), r"Models\Refined_Hierarchical.ckpt")
        self.ungulate_model_path = os.path.join(os.getcwd(), r"Models\Efficient_Net_Ungulates_3.ckpt")
        self.small_carnivore_model_path = os.path.join(os.getcwd(), r"Models\Efficient_Net_Small_Carnivores_1.ckpt")

        
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

        self.img_label = customtkinter.CTkLabel(master=self.frame, text = "Enter Source Path : ",font=("Roboto Medium", -18))
        self.img_label.grid(row = 6, column = 0, padx = 20, pady = 15, sticky = E, columnspan = 2)

        self.img_entry = customtkinter.CTkEntry(master=self.frame,
                                                placeholder_text="PATH",
                                                textvariable=self.img_path)

        self.img_entry.grid(row=6, column=2, sticky=E+W, pady = 15, columnspan = 2)

        self.img_btn = customtkinter.CTkButton(master=self.frame,
                                                text="Browse",
                                                font=("Roboto Medium", -12),
                                                command = lambda: self.browse("src"))
        self.img_btn.grid(row=6, column = 4, pady = 15, columnspan = 2)

        ##--------------------------------------------------- Copy and Rename Option --------------------------------------------------------

        self.copy_checkbox =    customtkinter.CTkCheckBox(master=self.frame,
                                                          variable=self.doCopyAndRenaming,
                                                          text = "Copy and Rename Files",
                                                          command = self.copyAndRename)
        self.copy_checkbox.grid(row=7, column=0, columnspan=6, sticky=E+W, pady = 1, padx=200)

        self.copyAndRenameFields()

        ##--------------------------------------------------- Geotagging --------------------------------------------------------------------

        self.geotag_checkbox =  customtkinter.CTkCheckBox(master=self.frame,
                                                          variable=self.doGeotagging,
                                                          text = "Geotag Files",
                                                          command = self.geotag)
        self.geotag_checkbox.grid(row=11, column=0, columnspan=6, pady=1, sticky=E+W, padx=200)

        self.geotaggingFields()

        ##----------------------------------------------------------- Select Model -----------------------------------------------------------

        # self.model_label = customtkinter.CTkLabel(master=self.model_frame, text = "Select the Desired 
        self.model_label = customtkinter.CTkLabel(master=self.model_frame,
                                                  text = "Select the Desired Model : ",
                                                  font=("Roboto Medium", -18, "bold"))
        self.model_label.grid(row = 0, column = 0, padx = 10, pady = 10, sticky = E, columnspan = 3)

        self.megadetector_model = customtkinter.CTkRadioButton(master=self.model_frame,
                                                               variable=self.model_choice,
                                                               value=0,
                                                               text = "Megadetector (Animal, Vehicle, Human)",
                                                               font=("Roboto Medium", -18),
                                                               command = lambda: self.set_species_list("megadetector"))
        self.megadetector_model.grid(row=1, column=2, sticky=E+W, columnspan = 3, padx = 10, pady = 10)

        self.order_level_model = customtkinter.CTkRadioButton(master=self.model_frame,
                                                               variable=self.model_choice,
                                                               value=1,
                                                               text = "Order Level (9 classes)",
                                                               font=("Roboto Medium", -18),
                                                               command = lambda: self.set_species_list("order"))
        self.order_level_model.grid(row=2, column=2, sticky=E+W, columnspan = 3, padx = 10, pady = 10)

        self.order_level_model = customtkinter.CTkRadioButton(master=self.model_frame,
                                                               variable=self.model_choice,
                                                               value=2,
                                                               text = "Complete (14 classes)",
                                                               font=("Roboto Medium", -18),
                                                               command = lambda: self.set_species_list("species"))
        self.order_level_model.grid(row=3, column=2, sticky=E+W, columnspan = 3, padx = 10, pady = 10)

        self.order_level_model = customtkinter.CTkRadioButton(master=self.model_frame,
                                                               variable=self.model_choice,
                                                               value=3,
                                                               text = "GIB Identification",
                                                               font=("Roboto Medium", -18),
                                                               command = lambda: self.set_species_list("GIB"))
        self.order_level_model.grid(row=4, column=2, sticky=E+W, columnspan = 3, padx = 10, pady = 10)

        ##------------------------------------------------------ Shift Original Imaages ---------------------------------------------------

        # self.model_label = customtkinter.CTkLabel(master=self.model_frame, text = "Select the Desired Model : ",font=("Roboto Medium", -18, "bold"))
        # self.model_label.grid(row = 0, column = 0, padx = 10, pady = 10, sticky = E, columnspan = 3)

        self.shift_original_checkbox =  customtkinter.CTkCheckBox(master=self.shift_images_frame,
                                                                  variable=self.doShifting,
                                                                  text = "Shift Original Images",
                                                                  font=("Roboto Medium", -18, "bold"),
                                                                  command = self.shiftImages)
        
        self.shift_original_checkbox.grid(row = 0, column = 0, padx = 10, pady = 10, sticky = W)
        
        self.shift_label = customtkinter.CTkLabel(master=self.shift_images_frame, text = "",font=("Roboto Medium", -18, "bold"))
        self.shift_label.grid(row = 1, column = 0, padx = 10, pady = 10, sticky=W)
        
        ##------------------------------------------------------ Predict Button ----------------------------------------------------------

        self.predict_btn = customtkinter.CTkButton(master=self.frame,
                                                text="RUN",
                                                font=("Roboto Medium", -18),
                                                command = lambda: self.run())
                                                # command = lambda: self.run(Path(self.img_path.get()),
                                                #                                int(self.data_type.get()),
                                                #                                int(self.model_choice.get()),
                                                #                                int(self.frames.get())))
        self.predict_btn.grid(row=17, column = 2, pady=20, padx = 30, columnspan = 2, sticky = E+W)


        ##--------------------------------------------------- Log Messages --------------------------------------------------------

        self.logHeading = customtkinter.CTkLabel(master=self.log_frame, text="LOG", font=("Roboto Medium", -25))
        self.logHeading.grid(row=0, column=0, columnspan=6, sticky=E+W+N+S, padx=250, pady=20)
        
        self.output_label = customtkinter.CTkLabel(master=self.log_frame, text = "", font=("Roboto Medium", -18))
        self.output_label.grid(row = 1, column = 0, columnspan = 6, padx=15, sticky=E+W+N+S)
        
        self.output_label_2 = customtkinter.CTkLabel(master=self.log_frame, text = "", font=("Roboto Medium", -18))
        self.output_label_2.grid(row = 2, column = 0, columnspan = 6, padx=15, sticky=E+W+N+S)
         
        # self.progress_bar_label = customtkinter.CTkLabel(master=self.frame, text = "", font=("Roboto Medium", -15))
        # self.progress_bar_label.grid(row = 2, column = 0, columnspan = 3)

    ## ----------------------------------------------------------------------- End of UI --------------------------------------------------
    
    ## ----------------------------------------------------------------------- Predict Function -------------------------------------------

    def print_log(self):
        self.output_label.configure(text = "")
        self.output_label.configure(text = self.log_msg)
        self.update()
        return
    
    def print_finished_log(self):
        self.output_label_2.configure(text = "")
        self.output_label_2.configure(text = self.finished_log_msg)
        self.update()
        return
    
    def aniPredictor_run(self, dst_path, model_choice):

        try:    
            print(f"Models loaded : {areModelsLoaded}")
        except:
            areModelsLoaded = False
        log = {}
        now = datetime.now()

        ## CHECK FOR GPU
        gpus = tf.config.experimental.list_physical_devices('GPU')
        cpu = tf.config.experimental.list_physical_devices("CPU")

        if gpus:
            print("GPU available")
            self.log_msg += "GPU Available \n"
            self.print_log()
            gpu_name = torch.cuda.get_device_name()
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)  
                    
            except:
                pass
        else:
            print("No GPUs available.")
            self.log_msg += "GPU Not Available \n"
            self.print_log()

        ## GET SUBDIRECTORIES
        self.log_msg += "Checking for Subdirectories Subdirs \n"
        self.print_log()

        sub_dirs = []
        for dirpath, dirnames, filenames in os.walk(dst_path):
            if not "Cropped_images" in dirpath and not self.dst_path == dirpath: # 'not "Cropped_images" in dirnames and' at the start if required
                if dirpath.split("\\")[-1] not in self.order_level_class_names:
                    sub_dirs.append(dirpath)
                
        if sub_dirs == []:
            sub_dirs.append(dst_path)

        sub_dirs = sub_dirs[1:]
        print("Printing Subdirs")
        
        for i in sub_dirs:
            print(i)

        ## LOAD MODELS
        print("Loading Models...")
        
        
        model_load_start=time.time()

        

        if areModelsLoaded == False:
            self.log_msg += "Loading Models \n"
            self.print_log()
            order_level_model = tf.keras.models.load_model(self.order_level_model_path)
            ungulate_model = tf.keras.models.load_model(self.ungulate_model_path)   
            small_carnivore_model= tf.keras.models.load_model(self.small_carnivore_model_path)
            areModelsLoaded = True
            
        model_load_end = time.time()
        model_load_time = str(timedelta(seconds=round(model_load_end - model_load_start)))
        log.update({"Species Model Load Time" : model_load_time})
        
        self.log_msg += f"Model Load Time : {model_load_time} \n"
        self.print_log()
        
        print(model_load_time)
        
        for data_dir in sub_dirs:
            print()
            print(f"\n File : \n{data_dir}")
            self.log_msg += f"\n File : \n{data_dir} \n"
            self.print_log()

            ## CREATE LOGS
            
            log.update({"Run timestamp" : str(now)})
            # log.update({"GPU" : gpus})
            log.update({"GPU Available for Classification : " : gpu_name})
            # log.update({"CPU" : cpu})
            num_images = 0
            # len(os.listdir(data_dir))
            for _,_,files in os.walk(data_dir):
                num_images += len(files) 
            # for root,dirs,files in os.walk(data_dir):
            #     if not root == "Cropped_images":
            #         num_images += len(files)
            #         for f in files:
            #             if not f.endswith(".jpg"):
            #                 num_images -= 1
            log.update({"Num images" : num_images})
            print(num_images)

            self.log_msg += f"Number of Images : {num_images} \n"
            self.print_log()

            ## RUN MEGADETECTOR AND CREATE DETECTIONS.DF
            self.log_msg += "Running Megadetector \n"
            self.print_log()

            megadetector_start = time.time()
            json_dir, megadetector_log = megadetector(data_dir, num_images)
            if not megadetector_log == {}:
                log.update(megadetector_log)
            else:
                megadetector_end = time.time()
                megadetector_time = str(timedelta(seconds=round(megadetector_end - megadetector_start)))
                log.update({"Megadetector time" : megadetector_time})
                log.update({"Megadetector Filename" : os.path.basename(json_dir)})
            
            df_detections = get_detection_df(data_dir, json_dir)
            df_final = df_detections
            
            
            ## CROP I{MAGES
            
            self.log_msg += "Cropping Detections \n"
            self.print_log()

            cropping_start = time.time() 
            cropped_images = os.path.join(data_dir,r"Cropped_images\*")
            cropped_dir = clean_path("\\".join(cropped_images.split("\\")[:-1]))
            if not os.path.exists(cropped_dir):
                print("Cropping Images")
                df_crop = df_detections.copy()
                df_crop["Cropped_image_directory"] = cropped_dir
                df_crop["Cropped_image_name"] = df_crop["Filename"] + "_" + df_crop["Detection_number"].astype(str) + ".jpg"
                df_crop["Cropped_image_path"] = (cropped_dir + "\\" + df_crop["Cropped_image_name"]).apply(clean_path)
                try:
                    df_crop=crop_images_batch_gpu(df_crop,512)
                except:
                    df_crop = crop_images_batch(df_crop,512)
                    print(f"Cropping exception occured")
            else:
                df_crop = df_detections.copy()
                df_crop["Cropped_image_directory"] = cropped_dir
                df_crop["Cropped_image_name"] = df_crop["Filename"] + "_" + df_crop["Detection_number"].astype(str) + ".jpg"
                df_crop["Cropped_image_path"] = (cropped_dir + "\\" + df_crop["Cropped_image_name"]).apply(clean_path)
                print("Images already cropped...")
            cropping_end = time.time()
            cropping_time = str(timedelta(seconds=round(cropping_end - cropping_start)))
            log.update({"Cropping Time" : cropping_time})
            log.update({"Number of Detections" : len(df_detections)})
            
            # df_crop_path = os.path.join(data_dir, "cropped.csv")
            # df_crop.to_csv(df_crop_path, index=False)
            
            if model_choice == 0:
                
                df_crop["Md_dir"] = (cropped_dir + "\\" + df_crop["Category"]).apply(clean_path)
                df_crop["Md_level_path"] = (df_crop["Md_dir"] + "\\" + df_crop["Cropped_image_name"]).apply(clean_path)
                
                unique_directories = set(df_crop['Md_dir'])
                for directory in unique_directories:
                    os.makedirs(directory, exist_ok=True)
                
                print("Moving Megadetector Level Images")
                self.log_msg += "Moving Megadetector Level Images \n"
                self.print_log()
                move_images_batch(df_crop["Cropped_image_path"], df_crop["Md_level_path"]) 
                df_final = df_crop
                    
            elif model_choice > 0:
                ## ORDER LEVEL PREDICTIONS
                
                if os.path.exists(os.path.join(cropped_dir,r"Ungulate")) or os.path.exists(os.path.join(cropped_dir,r"Small Carnivore")) or os.path.exists(os.path.join(cropped_dir,r"Others")):
                    self.log_msg += "Starting Order Level Predictions... \n"
                    self.print_log()
                    df_order = df_crop.copy()
                    df_order['Order_dir'] = np.nan
                    df_order['Order_level_path'] = np.nan
                    print("Order level classification already complete")
                else:
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
                    
                    print("Moving Order Level Images")
                    move_images_batch(df_order["Cropped_image_path"], df_order["Order_level_path"]) 
                    
                    order_shift_end = time.time()
                    order_pred_time = str(timedelta(seconds=round(order_level_end - order_level_start)))
                    order_shift_time = str(timedelta(seconds=round(order_shift_end - order_level_end)))
                    log.update({"Order Level Prediction Time" : order_shift_time})
                    log.update({"Order Level Shifting Time" : order_shift_time})
                    
                    self.log_msg += "Order Level Prediction Complete \n"
                    self.print_log()
                    df_final = df_order

                if model_choice == 3:
                    pass
                elif model_choice > 1:
                    ## SMALL CARNIVORES PREDICT
                    small_carnivores_start = time.time()
                    if os.path.exists(os.path.join(cropped_dir,r"Small Carnivore")) and len(os.listdir(os.path.join(cropped_dir,r"Small Carnivore"))) > 0:
                        print("Predicting Small Carnivores...")

                        self.log_msg += "Predicting Small Carnivores \n"
                        self.print_log()

                        df_small_carnivore, num_small_carnivores = predict_lower_level_species(cropped_dir, 
                                                                                            r"Small Carnivore\*", 
                                                                                            self.small_carnivores_class_names,
                                                                                            small_carnivore_model,
                                                                                            level = "Species")
                        df_small_carnivore["Order_pred"] = r"Small Carnivore"
                        df_small_carnivore["Order_dir"] = (cropped_dir + "\\" + df_small_carnivore["Order_pred"]).apply(clean_path)
                        df_small_carnivore["Order_level_path"] = (df_small_carnivore["Order_dir"] + "\\" + df_small_carnivore["Cropped_image_name"]).apply(clean_path)
                        small_carnivores_end = time.time()
                        small_carnivore_time = str(timedelta(seconds=round(small_carnivores_end - small_carnivores_start)))
                        log.update({"Number of Small Carnivores Images" : num_small_carnivores})
                        log.update({"Small Carnivore Model Pred Time" : small_carnivore_time})
                    else:
                        df_small_carnivore = pd.DataFrame(columns=['Cropped_image_name','Species_pred','Species_pred_prob'])
                
                    ## UNGULATES PREDICT

                    if os.path.exists(os.path.join(cropped_dir,r"Ungulate")) and len(os.listdir(os.path.join(cropped_dir,r"Ungulate"))) > 0 and not os.path.exists(os.path.join(cropped_dir,r"Cattle")):
                        ungulate_start = time.time()
                        print("Predicting Ungulates...")
                        self.log_msg += "Predicting Ungulates \n"
                        self.print_log()

                        df_ungulate, num_ungulates = predict_lower_level_species(cropped_dir, 
                                                                                r"Ungulate\*", 
                                                                                self.ungulate_class_names,
                                                                                ungulate_model,
                                                                                level = "Species")
                        df_ungulate["Order_pred"] = r"Ungulate"
                        df_ungulate["Order_dir"] = (cropped_dir + "\\" + df_ungulate["Order_pred"]).apply(clean_path)
                        df_ungulate["Order_level_path"] = (df_ungulate["Order_dir"] + "\\" + df_ungulate["Cropped_image_name"]).apply(clean_path)
                        ungulate_end = time.time()
                        ungulate_time = str(timedelta(seconds=round(ungulate_end - ungulate_start)))
                        log.update({"Number of Ungulates Images" : num_ungulates})
                        log.update({"Ungulate Model Pred Time" : ungulate_time})
                    else:
                        df_ungulate = pd.DataFrame(columns=['Cropped_image_name','Species_pred','Species_pred_prob'])
                    
                    species_shift_start = time.time()
                    df_species = pd.concat([df_small_carnivore,df_ungulate])
                    df_species["Species_dir"] = (cropped_dir + "\\" + df_species["Species_pred"]).apply(clean_path)
                    df_species["Species_level_path"] = (df_species["Species_dir"] + "\\" + df_species["Cropped_image_name"]).apply(clean_path)
                    
                    #df_move = pd.merge(df_species, df_order, on='Cropped_image_name', how='left')
                    try:
                        df_move = df_species[df_species["Order_level_path"] != df_species["Species_level_path"]]
                        unique_directories = set(df_move['Species_dir'])
                        for directory in unique_directories:
                            os.makedirs(directory, exist_ok=True)
                        print("Moving Ungulates and Small Carnivores...")

                        self.log_msg += "Moving Ungulates and Small Carnivores\n"
                        self.print_log()

                        #copy_images_batch(df_move["Order_level_path"], df_move["Species_level_path"])
                        #delete_images_batch(df_move["Order_level_path"])
                        move_images_batch(df_move["Order_level_path"], df_move["Species_level_path"])
                    except:
                        print("No files to move")
                        
                    species_shift_end = time.time()
                    species_shift_time = str(timedelta(seconds=round(species_shift_end - species_shift_start)))
                    species_level_time = str(timedelta(seconds=round(species_shift_end - small_carnivores_start)))
                    log.update({"Species Level Shift Imgs Time" : species_shift_time})
                    log.update({"Species Level Predict and Shift" : species_level_time})
                    df_final = df_species
                
                    # SAVE FINAL PREDICTIONS.CSV

                    if os.path.exists(os.path.join(data_dir,r"predictions.csv")):
                        predictions = pd.read_csv(os.path.join(data_dir,r"predictions.csv"))
                        preds = predictions.copy().dropna()
                        preds_all = pd.concat([preds,df_species[~df_species["Cropped_image_name"].isin(preds["Cropped_image_name"].tolist())]])
                        preds_all = preds_all[["Cropped_image_name","Species_pred","Species_pred_prob","Species_dir","Species_level_path"]]
                        predictions.drop(columns = ["Species_pred","Species_pred_prob","Species_dir","Species_level_path"], inplace = True)
                        df_final = pd.merge(predictions, preds_all, how='left', on = "Cropped_image_name")
                    else:
                        try:
                            df_final = pd.merge(df_order, df_species.drop(columns=['Order_pred','Order_dir', 'Order_level_path']), on='Cropped_image_name', how='left')
                        except:
                            df_final = pd.merge(df_order, df_species, on='Cropped_image_name', how='left')
                        df_final.drop(columns=['Order_dir', 'Order_level_path','Cropped_image_path'], inplace=True)
                        
            df_final_path = os.path.join(data_dir, "predictions.csv")
            df_final.to_csv(df_final_path, index=False)
            
            ## SAVE LOG FILE
            self.log_msg += "Saving Logs \n"
            self.print_log()
            print("Saving Logs")
            log_file_name = "_".join(data_dir.split("\\")[-3:])
            log_file_path = os.path.join(data_dir, f"{log_file_name}_log.json")
            with open(log_file_path, "w") as f:
                json.dump(log, f, indent=2)
            gc.collect()
            
            self.log_msg += "Converting predictions to json\n"
            self.print_log()
            
            create_json_obj = CreateJSON()
            json_save_path = os.path.join(data_dir, "predictions.json")
            create_json_obj.run(df_final_path, json_save_path, model_choice)
            
            self.finished_log_msg += f"{data_dir} \n"
            self.print_finished_log()
        
        return
        
    def run(self):
        src_path = Path(self.img_path.get())
        dst_path = Path(self.dst_path.get())
        data_type = int(self.data_type.get())
        model_choice = int(self.model_choice.get())
        num_frames = int(self.frames.get())
        camera = self.camera.get()
        station = self.station.get()
        lat = self.latitude.get()
        long = self.longitude.get()
        copyAndRename = self.doCopyAndRenaming.get()
        geotag = self.doGeotagging.get()
        shiftOriginals = self.doShifting.get()
        species_list = []
        run_model = self.runModel.get()

        
        self.finished_log_msg = "\n\n DIRECTORIES PROCESSED : \n\n"
        self.print_finished_log()
        
        # Delete Corrupt Files 
        self.log_msg = "Checking for Corrupt Files \n"
        self.print_log()
        corruptFileObj = Corrupt_Files()
        corrupt_files = corruptFileObj.list_corrupt_files_in_directory(src_path)
        print(f"Found {len(corrupt_files)} corrupt files..")
        corruptFileObj.delete_corrupt_files(corrupt_files, src_path)

        # Copy and Rename 
        if copyAndRename:
            print("Copying and Renaming Files")
            self.log_msg += "Copying and Renaming Files \n"
            self.print_log()
            copyObj = RenameAndCopy(dest_dir=dst_path, camera=camera, station=station)
            self.unique_directories = copyObj.run(input_dir=src_path)
        
        # Geotag Images
        if geotag:
            print("Geotagging Images...")
            self.log_msg += "Geotagging Images \n"
            self.print_log()
            geotagObj = Geotag(lat=lat, long=long)
            geotagObj.run(self.unique_directories)  
        
        # Run Model
        run_model = 1
        if run_model:
            print("Predicting Species..")
            
            self.aniPredictor_run(dst_path, model_choice)
            # aniPredictorObj = AniPredictor(dst_path)
            # aniPredictorObj.run()  

        #Shift Original Images
        if shiftOriginals:
            print("Shifting Original Images...")
            self.log_msg += "Shifting Original Images \n"
            self.print_log()
            selected_species = self.selected_species
            print(selected_species)
            shiftImagesObj = ShiftOriginals()
            shiftImagesObj.run(dst_path, selected_species)

        self.log_msg += "\n RUN COMPLETE \n"
        self.print_log()
        
        return
    ## ----------------------------------------------------------------------- Other Functions --------------------------------------------


    def checkbox_frame_event(self):
        self.selected_species = self.scrollable_checkbox_frame.get_checked_items()
        # print(f"checkbox frame modified: {self.selected_species}")
        return

    def check_env(self):
        if len(tf.config.list_physical_devices("GPU"))>0:
            self.check_env_response.configure(text = "GPU Available :))")
        else:
            self.check_env_response.configure(text = "GPU Unavailable :((")
        return 

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

        return
    
    def copyAndRenameFields(self):

        self.copy_label = customtkinter.CTkLabel(master=self.copy_frame, text = "Camera :   ",font=("Roboto Medium", -15))
        self.copy_label.grid(row = 8, column = 1, padx = 20,  pady=10, sticky = E, columnspan=1)

        self.copy_entry = customtkinter.CTkEntry(master=self.copy_frame,
                                                placeholder_text="NAME",
                                                textvariable=self.camera)
        self.copy_entry.grid(row=8, column=2, sticky=E+W, columnspan = 1, pady=10)

        self.copy_label = customtkinter.CTkLabel(master=self.copy_frame, text = "    Station :   ",font=("Roboto Medium", -15))
        self.copy_label.grid(row = 8, column = 3, sticky = W, columnspan=1, pady=10)

        self.copy_entry = customtkinter.CTkEntry(master=self.copy_frame,
                                                placeholder_text="NAME",
                                                textvariable=self.station)
        self.copy_entry.grid(row=8, column=4, sticky=W, columnspan = 1, padx = 10, pady=10)

        self.copy_label = customtkinter.CTkLabel(master=self.copy_frame, text = "Enter Destination Path : ",font=("Roboto Medium", -15))
        self.copy_label.grid(row = 9, column = 1, padx = 20, sticky = E, columnspan = 1, pady=10)

        self.copy_entry = customtkinter.CTkEntry(master=self.copy_frame,
                                                placeholder_text="PATH",
                                                textvariable=self.dst_path)
        self.copy_entry.grid(row=9, column=2, sticky=E+W, padx=3, columnspan = 2, pady=10)

        self.copy_btn = customtkinter.CTkButton(master=self.copy_frame,
                                                text="Browse",
                                                font=("Roboto Medium", -12),
                                                command = lambda: self.browse("dst"))
        self.copy_btn.grid(row=9, column = 4, columnspan = 1, sticky=W, padx=20, pady=10)

        return
    
    def geotaggingFields(self):

        self.copy_label = customtkinter.CTkLabel(master=self.geotag_frame, text = "Latitude :   ",font=("Roboto Medium", -15))
        self.copy_label.grid(row = 0, column = 1, padx = 20,  pady=10, sticky = E, columnspan=1)

        self.copy_entry = customtkinter.CTkEntry(master=self.geotag_frame,
                                                placeholder_text="COORDINATES",
                                                textvariable=self.latitude)
        self.copy_entry.grid(row=0, column=2, sticky=E+W, columnspan = 1, pady=10)

        self.copy_label = customtkinter.CTkLabel(master=self.geotag_frame, text = "    Longitude :   ",font=("Roboto Medium", -15))
        self.copy_label.grid(row = 0, column = 3, sticky = W, columnspan=1, pady=10)

        self.copy_entry = customtkinter.CTkEntry(master=self.geotag_frame,
                                                placeholder_text="COORDINATES",
                                                textvariable=self.longitude)
        self.copy_entry.grid(row=0, column=4, sticky=W, columnspan = 1, padx = 10, pady=10)

        return
    
    def shiftImagesFields(self):
        # self.species_list = ["GIB", "Ungulates", "Small Carnivores", "Hare", "Small Birds", "Raptors", "Wild Pig", "Human", "Vehicle"]
        self.shift_label.configure(text = "Select Species", font = ("Roboto Medium", -18, "bold"))
        self.shift_label.grid(row = 1, column = 0, padx = 10, pady = 10, sticky=W)
        
        self.scrollable_checkbox_frame = ScrollableCheckBoxFrame(master=self.shift_images_frame, command=self.checkbox_frame_event,
                                                                 item_list = self.species_list)
        self.scrollable_checkbox_frame.grid(row=2, column=0, padx=15, sticky=W)
        # self.scrollable_checkbox_frame.add_item("new item")
        
        
        return
    
    def set_species_list(self, tag):
        if tag == "megadetector":
            self.species_list = ["Animal", "Human", "Vehicle"]
            print(self.species_list)
        elif tag == "order":
            self.species_list = self.order_level_class_names
            print(self.species_list)
        elif tag == "species":
            self.species_list = self.order_level_class_names + self.ungulate_class_names + self.small_carnivores_class_names
            print(self.species_list)
        else:
            self.species_list = ["GIB"]
            print(self.species_list)
            
        if self.doShifting == True:
            self.scrollable_checkbox_frame = ScrollableCheckBoxFrame(master=self.shift_images_frame, command=self.checkbox_frame_event,
                                                                 item_list = self.species_list)
            self.scrollable_checkbox_frame.grid(row=2, column=0, padx=15, sticky=W)
            
        return
    
    def browse(self, type):
        self.filename = filedialog.askdirectory()
        print(self.filename)
        if type == "src":
            self.img_path.set(self.filename)
        else:
            self.dst_path.set(self.filename)
        return
    
    def geotag(self):
        if self.doGeotagging.get() == True:
            print("ON")
            self.geotag_frame = customtkinter.CTkFrame(master = self.frame)
            self.geotag_frame.grid(row=12, column=0, padx = 5, pady = 5, columnspan = 6)
            self.geotaggingFields()
        else:
            print("OFF")
            self.geotag_frame.grid_forget()

        print(self.doGeotagging.get())
        return

    def copyAndRename(self):
        if self.doCopyAndRenaming.get() == True:
            self.copy_frame = customtkinter.CTkFrame(master = self.frame)
            self.copy_frame.grid(row=8, column=0, padx = 5, pady = 5, columnspan = 6)
            self.copyAndRenameFields()
            print("ON")
        else:
            print("OFF")
            self.copy_frame.grid_forget()

        print(self.doCopyAndRenaming.get())
        return

    def shiftImages(self):
        if self.doShifting.get() == True:
            print("ON")
            # self.shift_images_frame = customtkinter.CTkFrame(master = self.frame)
            # self.shift_images_frame.grid(row=14, column=3, pady=20, columnspan=3)
            self.shiftImagesFields()
        else:
            print("OFF")
            self.shift_label.configure(text = "", font = ("Roboto Medium", -18, "bold"))
            self.scrollable_checkbox_frame.grid_forget()
            # self.shift_images_frame.grid_forget()

        print(self.doShifting.get())
        return
    
    def on_closing(self, event=0):
        self.destroy()
    
if __name__ == "__main__":
    app = App()
    # app.attributes("-fullscreen", "True")
    app.mainloop()