# from dependencies import *
# from gui_functs import *
from find_corrupt_files import *
from geotag import *
from rename_and_copy import *
from aniPredictor import *
from shift_original_images import *

customtkinter.set_appearance_mode("dark")  # Modes: "System" (standard), "Dark", "Light"
customtkinter.set_default_color_theme("green")  # Themes: "blue" (standard), "green", "dark-blue"
    
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
        self.heading = customtkinter.CTkLabel(master=self.frame, text = "AniPredictor.py", font=("Roboto Medium", -25))
        self.heading.grid(row = 0, column = 0, columnspan = 6, pady = 20)
        self.runModel = BooleanVar(value = False)
        
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

        self.model_label = customtkinter.CTkLabel(master=self.model_frame, text = "Select the Desired Model : ",font=("Roboto Medium", -18, "bold"))
        self.model_label.grid(row = 0, column = 0, padx = 10, pady = 10, sticky = E, columnspan = 3)

        self.megadetector_model = customtkinter.CTkRadioButton(master=self.model_frame,
                                                               variable=self.model_choice,
                                                               value=0,
                                                               text = "Megadetector (Animal, Vehicle, Human)",
                                                               font=("Roboto Medium", -18))
        self.megadetector_model.grid(row=1, column=2, sticky=E+W, columnspan = 3, padx = 10, pady = 10)

        self.order_level_model = customtkinter.CTkRadioButton(master=self.model_frame,
                                                               variable=self.model_choice,
                                                               value=1,
                                                               text = "Order Level (9 classes)",
                                                               font=("Roboto Medium", -18))
        self.order_level_model.grid(row=2, column=2, sticky=E+W, columnspan = 3, padx = 10, pady = 10)

        self.order_level_model = customtkinter.CTkRadioButton(master=self.model_frame,
                                                               variable=self.model_choice,
                                                               value=2,
                                                               text = "Complete (14 classes)",
                                                               font=("Roboto Medium", -18))
        self.order_level_model.grid(row=3, column=2, sticky=E+W, columnspan = 3, padx = 10, pady = 10)

        self.order_level_model = customtkinter.CTkRadioButton(master=self.model_frame,
                                                               variable=self.model_choice,
                                                               value=3,
                                                               text = "GIB Identification",
                                                               font=("Roboto Medium", -18))
        self.order_level_model.grid(row=4, column=2, sticky=E+W, columnspan = 3, padx = 10, pady = 10)

        ##------------------------------------------------------ Shift Original Imaages ---------------------------------------------------

        self.species_list = ["GIB", "Ungulates", "Small Carnivores", "Hare", "Small Birds", "Raptors", "Wild Pig", "Human", "Vehicle"]
        self.shiftImages = customtkinter.CTkLabel(master=self.shift_images_frame, text = "Shift Original Images For : ",font=("Roboto Medium", -18, "bold"))
        self.shiftImages.grid(row = 0, column = 0, padx = 10, pady = 10, sticky = E)

        
        
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
        self.output_label.grid(row = 1, column = 0, columnspan = 3)

        self.progress_bar_label = customtkinter.CTkLabel(master=self.frame, text = "", font=("Roboto Medium", -15))
        self.progress_bar_label.grid(row = 2, column = 0, columnspan = 3)

    ## ----------------------------------------------------------------------- End of UI --------------------------------------------------
    
    ## ----------------------------------------------------------------------- Predict Function -------------------------------------------

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
        # shiftOriginals = self.doShifting.get()
        species_list = []
        run_model = self.runModel.get()

        # Delete Corrupt Files 
        corruptFileObj = Corrupt_Files()
        corrupt_files = corruptFileObj.list_corrupt_files_in_directory(src_path)
        print(f"Found {len(corrupt_files)} corrupt files..")
        corruptFileObj.delete_corrupt_files(corrupt_files, src_path)

        # Copy and Rename 
        if copyAndRename:
            print("Copying and Renaming Files")
            copyObj = RenameAndCopy(dest_dir=dst_path, camera=camera, station=station)
            self.unique_directories = copyObj.run(input_dir=src_path)
        
        # Geotag Images
        if geotag:
            print("Geotagging Images...")
            geotagObj = Geotag(lat=lat, long=long)
            geotagObj.run(self.unique_directories)
        
        # Run Model
        run_model = 1
        if run_model:
            print("Predicting Species..")
            
            aniPredictorObj = AniPredictor(dst_path)
            aniPredictorObj.run()

        # Shift Original Images
        # if shiftOriginals:
        #     print("Shifting Original Images...")
        #     shiftImagesObj = ShiftOriginals()
        #     shiftImagesObj.run()

        






        return
    ## ----------------------------------------------------------------------- Other Functions --------------------------------------------


    def checkbox_frame_event(self):
        print(f"checkbox frame modified: {self.scrollable_checkbox_frame.get_checked_items()}")

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

    def on_closing(self, event=0):
        self.destroy()
    
if __name__ == "__main__":
    app = App()
    app.attributes("-fullscreen", "True")
    app.mainloop()