from dependencies import *

class RenameAndCopy():
    def __init__(self, dest_dir, camera, station) -> None:
        self.DEST_DIR = dest_dir
        self.CAMERA = camera
        self.STATION = station

    def list_files_in_directory(self, directory):
        file_paths = []

        for root, dirs, files in os.walk(directory):
            for file in files:
                if file.lower().endswith(('.jpg', '.jpeg','.csv','.json')):
                    file_path = os.path.join(root, file)
                    file_paths.append(file_path)
                else:
                    continue
        return file_paths

    def convert_datetime(self, dt):
        try:
            dt = pd.to_datetime(dt, format='%Y-%m-%d %H-%M-%S')
            return dt.strftime('%Y%m%d_%H%M%S')
        except pd.errors.OutOfBoundsDatetime:
            return 'OutOfBoundsDatetime'

    def clean_path(self, path):
        return os.path.normpath(path)

    def process_image(self, image_path):
        with open(image_path, 'rb') as image_file:
            tags = exifread.process_file(image_file, details=False)
            
            directory = os.path.dirname(image_path)
            filename = os.path.basename(image_path)
            filetype_extension = os.path.splitext(filename)[1]
            make = tags.get('Image Make', 'N/A')
            model = tags.get('Image Model', 'N/A')
            datetime_original = tags.get('EXIF DateTimeOriginal', 'N/A')
            
            return {
                'SourceFile': image_path,
                'Directory': directory,
                'FileName': filename,
                'FileTypeExtension': filetype_extension,
                'Make': make,
                'Model': model,
                'DateTimeOriginal': datetime_original
            }

    def read_exif(self, image_dir):
        file_paths = self.list_files_in_directory(image_dir)
        print(len(file_paths))
        with ThreadPoolExecutor(20) as executor:  # Adjust max_workers as needed
            image_metadata_list = list(executor.map(self.process_image, file_paths))
        exif_info = pd.DataFrame(image_metadata_list)
        return exif_info

    def create_new_filenames(self, exif_info):
        exif_info = exif_info[exif_info["DateTimeOriginal"] != "N/A"]
        highest_subfolder_number = self.get_highest_subfolder_number(self.DEST_DIR)
        exif_info['Station'] = self.STATION
        exif_info['Camera'] = self.CAMERA
        exif_info['DateTimeOriginal'] = pd.to_datetime(exif_info['DateTimeOriginal'], format='%Y:%m:%d %H:%M:%S')
        exif_info['FormattedDateTime'] = exif_info['DateTimeOriginal'].apply(self.convert_datetime)
        exif_info = exif_info.sort_values(by=['Station', 'Camera', 'DateTimeOriginal']).reset_index(drop=True)
        exif_info['diff'] = exif_info.groupby(['Station', 'Camera'])['DateTimeOriginal'].diff()
        exif_info['image_number']=exif_info.groupby(['Station','Camera']).cumcount()+1
        exif_info['Directory'] = exif_info['Directory'].apply(self.clean_path)
        exif_info['SourceFile'] = exif_info['SourceFile'].apply(self.clean_path)
        exif_info['Dest_subfolder_number'] = (highest_subfolder_number + exif_info['image_number'].apply(lambda x: math.ceil(x / 10000))).astype(str)
        
        subfolders=[]
        min_dates=[]
        max_dates=[]
        for d in exif_info["Dest_subfolder_number"].unique():
            subfolders.append(d)
            temp_renaming_table = exif_info.loc[exif_info["Dest_subfolder_number"] == d]  
            min_date = datetime.strftime(min(temp_renaming_table.FormattedDateTime.apply(lambda x: datetime.strptime(x,'%Y%m%d_%H%M%S'))),'%Y%m%d_%H%M%S')
            min_dates.append(min_date)
            max_date = datetime.strftime(max(temp_renaming_table.FormattedDateTime.apply(lambda x: datetime.strptime(x,'%Y%m%d_%H%M%S'))),'%Y%m%d_%H%M%S')
            max_dates.append(max_date)
        subfolder_intervals_df = pd.DataFrame({"Dest_subfolder_number" : subfolders, "min_date" : min_dates, "max_date" : max_dates})
        exif_info = exif_info.merge(subfolder_intervals_df, how = "left")
        exif_info['Dest_Directory'] = (str(self.DEST_DIR) + "\\" + exif_info['min_date'] + "__to__" + exif_info["max_date"]).apply(self.clean_path)
        # exif_info['Dest_Directory'] = (self.DEST_DIR + "\\" + exif_info['min_date'] + "__to__" + exif_info["max_date"]).apply(self.clean_path)
        #exif_info['Dest_Directory'] = (dest_dir + "\\" + exif_info["Dest_subfolder_number"]).apply(clean_path)

        ### Add sequence number
        threshold = timedelta(seconds=1)
        Sequence = []
        for i in range(len(exif_info)):
            diff = exif_info['diff'][i]
            if pd.isna(diff) or diff > threshold:
                sequence = 1
            else:
                sequence = Sequence[i - 1] + 1
            Sequence.append(sequence)
        exif_info['Sequence'] = Sequence

        ### Construct new filename
        exif_info['FileNameNew'] = exif_info['Station'] + '_' + exif_info['Camera'] + '_' + exif_info['FormattedDateTime'] + '(' + exif_info['Sequence'].astype(str) + ')' + exif_info['FileTypeExtension']
        exif_info['DestFile'] = (exif_info['Dest_Directory'] + "\\" + exif_info['FileNameNew']).apply(self.clean_path)
        
        return exif_info

    def get_highest_subfolder_number(self, dest_dir):
        # Get existing subdirectories
        if os.path.isdir(dest_dir):
            existing_subdirs = [d for d in os.listdir(dest_dir) if os.path.isdir(os.path.join(dest_dir, d))]
            subfolder_numbers = [int(dir) for dir in existing_subdirs]
            try:
                max_number = max(subfolder_numbers)
            except:
                max_number = 0
        else:
            max_number = 0
        # Return the highest subfolder number or 0 if none exist
        return max_number

    def copy_images_batch(self, table, batch_size=1000):
        src_files=table['SourceFile']
        dest_files=table['DestFile']
        with ThreadPoolExecutor(20) as exe:
            for i in range(0, len(src_files), batch_size):
                src_batch = src_files[i:i + batch_size]
                dest_batch = dest_files[i:i + batch_size]
                
                batch_tasks = [exe.submit(shutil.copy, src, dest) for src, dest in zip(src_batch, dest_batch)]
                # Wait for all tasks in the batch to complete before proceeding to the next batch
                _ = [task.result() for task in batch_tasks]
                print(f"First {i+1 * 1000} images copied at {datetime.now()}")
                gc.collect()

        return
    
    def run(self, input_dir):
        start = datetime.now()
        ###Create Renaming Table
        exif = self.read_exif(input_dir)
        renaming_table=self.create_new_filenames(exif)
        print(f"Renaming table created in {datetime.now() - start}")

        ### Copy and rename in batches, based on renaming table
        unique_directories = set(renaming_table['Dest_Directory'])
        for d in unique_directories:
            if not os.path.exists(d):
                os.makedirs(d)
        self.copy_images_batch(renaming_table)
        print(f"Renaming and copying completed in {datetime.now() - start}")

        return unique_directories