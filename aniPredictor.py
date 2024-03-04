from dependencies import *
from aniPred_helper import *

class AniPredictor():
    def __init__(self, dest_dir) -> None:
        self.dest_dir = dest_dir

    def megadetector(self, img_dir):
        print("Megadetector model")
        log = {}
        local_detector_path = os.path.join(os.getcwd(), r"cameratraps\\detection\\run_detector_batch.py")
        megadetector_path = os.path.join(os.getcwd(), "md_v5a.0.0.pt")
        output_file_name = "_".join(img_dir.split("\\")[-3:])
        json_dir = os.path.join(img_dir, f"{output_file_name}_megadetector.json")

        if os.path.exists(json_dir):
            print("Megadetector output file already exists.. Going for species classification")
            log_dir = os.path.join(img_dir, f"{output_file_name}_log.json")
            if os.path.exists(log_dir):
                with open(log_dir, 'r') as f:
                    log = json.load(f)
            return json_dir, log
        
        print(f"Saving detections at {json_dir}...")
        
        command = [sys.executable,
                local_detector_path,
                megadetector_path,
                img_dir,
                json_dir,
                "--recursive"]
        
        prev_percentage = 0
        with Popen(command,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=1, shell=True,
                universal_newlines=True) as p:
            for line in p.stdout:
                if line.startswith("Loaded model in"):
                    print(line)
                
                elif "%" in line[0:4]:
                    percentage = re.search("\d*%", line[0:4])[0][:-1]
                    if percentage > prev_percentage:
                        prev_percentage = percentage
                        print(percentage)
                    

        # command = [sys.executable,
        #             local_detector_path,
        #             megadetector_path,
        #             img_dir,
        #             json_dir]
        
        # # with tqdm(total = min(100, num_images)) as t:
        # prev_percentage = 0
        # with Popen(command,
        #         stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=1, shell=True,
        #         universal_newlines=True) as p:
        #     for line in p.stdout:
                
        #         if line.startswith("Loaded model in"):
        #             print(line)
                
        #         elif "%" in line[0:4]:
        #             percentage = int(re.search("\d*%", line[0:4])[0][:-1])
        #             if percentage > prev_percentage:
        #                 prev_percentage = percentage
        #                 print(percentage)
                        # t.update(1)
    
        print("Bounding Boxes Created")

        return json_dir, log

    def run(self):
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
            gpu_name = torch.cuda.get_device_name()
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)  
                    
            except:
                pass
        else:
            print("No GPUs available.")
        
        ## GET SUBDIRECTORIES
        sub_dirs = []
        for dirpath, dirnames, filenames in os.walk(self.dest_dir):
            if not "Cropped_images" in dirpath and not self.dest_dir == dirpath: # 'not "Cropped_images" in dirnames and' at the start if required
                if dirpath.split("\\")[-1] not in ["GIB", "Goat_Sheep", "Hare", "Human", "Raptor", "Small Bird", "Small Carnivore", "Ungulate", "Vehicle", "Wild Pig"]:
                    sub_dirs.append(dirpath)
                
        if sub_dirs == []:
            sub_dirs.append(self.dest_dir)

        sub_dirs = sub_dirs[1:]
        print("Printing Subdirs")
        for i in sub_dirs:
            print(i)

        ## LOAD MODELS
        print("Loading Models...")
        order_level_class_names = ["GIB", "Goat_Sheep", "Hare", "Human", "Raptor", "Small Bird", "Small Carnivore", "Ungulate", "Vehicle", "Wild Pig"]
        order_level_class_names.sort()
        ungulate_class_names = ["Camel", "Chinkara", "Nilgai", "Cattle"]
        ungulate_class_names.sort() 
        small_carnivores_class_names = ["Dog", "Desert Cat", "Fox"]
        small_carnivores_class_names.sort()
        
        model_load_start=time.time()
        if areModelsLoaded == False:
            order_level_model_path = os.path.join(os.getcwd(), r"Models\Refined_Hierarchical.ckpt")
            order_level_model = tf.keras.models.load_model(order_level_model_path)
            
            ungulate_model_path = os.path.join(os.getcwd(), r"Models\Efficient_Net_Ungulates_3.ckpt")
            ungulate_model = tf.keras.models.load_model(ungulate_model_path)   
            
            small_carnivore_model_path = os.path.join(os.getcwd(), r"Models\Efficient_Net_Small_Carnivores_1.ckpt")
            small_carnivore_model= tf.keras.models.load_model(small_carnivore_model_path)
            
            areModelsLoaded = True
            
        model_load_end = time.time()
        model_load_time = str(timedelta(seconds=round(model_load_end - model_load_start)))
        log.update({"Species Model Load Time" : model_load_time})
        print(model_load_time)
        
        for data_dir in sub_dirs:
            print()
            print(f"Running Megadetector on {data_dir}")
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
            
            ## RUN MEGADETECTOR AND CREATE DETECTIONS.DF
            
            megadetector_start = time.time()
            json_dir, megadetector_log = self.megadetector(data_dir)
            if not megadetector_log == {}:
                log.update(megadetector_log)
            else:
                megadetector_end = time.time()
                megadetector_time = str(timedelta(seconds=round(megadetector_end - megadetector_start)))
                log.update({"Megadetector time" : megadetector_time})
                log.update({"Megadetector Filename" : os.path.basename(json_dir)})

            df_detections = get_detection_df(data_dir, json_dir)
            
            ## CROP IMAGES
            
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
                print("Images already cropped...")
            cropping_end = time.time()
            cropping_time = str(timedelta(seconds=round(cropping_end - cropping_start)))
            log.update({"Cropping Time" : cropping_time})
            log.update({"Number of Detections" : len(df_detections)})
            
            ## ORDER LEVEL PREDICTIONS
            
            if os.path.exists(os.path.join(cropped_dir,r"Ungulate")) or os.path.exists(os.path.join(cropped_dir,r"Small Carnivore")):
                print("Order level classification already complete")
            else:
                order_level_start = time.time()
                print("Predicting Order Level Classes...")
                df_temp, num_cropped = predict_lower_level_species(data_dir, 
                                                                r"Cropped_images\*", 
                                                                order_level_class_names,
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
                move_images_batch(df_order["Cropped_image_path"], df_order["Order_level_path"])
                
                order_shift_end = time.time()
                order_pred_time = str(timedelta(seconds=round(order_level_end - order_level_start)))
                order_shift_time = str(timedelta(seconds=round(order_shift_end - order_level_end)))
                log.update({"Order Level Prediction Time" : order_shift_time})
                log.update({"Order Level Shifting Time" : order_shift_time})
            
            ## SMALL CARNIVORES PREDICT
            small_carnivores_start = time.time()
            if os.path.exists(os.path.join(cropped_dir,r"Small Carnivore")):
                print("Predicting Small Carnivores...")
                df_small_carnivore, num_small_carnivores = predict_lower_level_species(cropped_dir, 
                                                                                    r"Small Carnivore\*", 
                                                                                    small_carnivores_class_names,
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

            if os.path.exists(os.path.join(cropped_dir,r"Ungulate")):
                ungulate_start = time.time()
                print("Predicting Ungulates...")
                df_ungulate, num_ungulates = predict_lower_level_species(cropped_dir, 
                                                                        r"Ungulate\*", 
                                                                        ungulate_class_names,
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
            df_move = df_species[df_species["Order_level_path"] != df_species["Species_level_path"]]
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
            log.update({"Species Level Shift Imgs Time" : species_shift_time})
            log.update({"Species Level Predict and Shift" : species_level_time})
            
            ## SAVE FINAL PREDICTIONS.CSV
            if os.path.exists(os.path.join(data_dir,r"predictions.csv")):
                predictions = pd.read_csv(os.path.join(data_dir,r"predictions.csv"))
                preds = predictions.copy().dropna()
                preds_all = pd.concat([preds,df_species[~df_species["Cropped_image_name"].isin(preds["Cropped_image_name"].tolist())]])
                preds_all = preds_all[["Cropped_image_name","Species_pred","Species_pred_prob","Species_dir","Species_level_path"]]
                predictions.drop(columns = ["Species_pred","Species_pred_prob","Species_dir","Species_level_path"], inplace = True)
                df_final = pd.merge(predictions, preds_all, how='left', on = "Cropped_image_name")
            else:
                df_final = pd.merge(df_order, df_species.drop(columns=['Order_dir', 'Order_level_path']), on='Cropped_image_name', how='left')
                df_final.drop(columns=['Order_dir', 'Order_level_path','Cropped_image_path'], inplace=True)
            df_final_path = os.path.join(data_dir, "predictions.csv")
            df_final.to_csv(df_final_path, index=False)
            
            ## SAVE LOG FILE
            print("Saving Logs")
            log_file_name = "_".join(data_dir.split("\\")[-3:])
            log_file_path = os.path.join(data_dir, f"{log_file_name}_log.json")
            with open(log_file_path, "w") as f:
                json.dump(log, f, indent=2)
            gc.collect()
        
            return
        