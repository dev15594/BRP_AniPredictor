from dependencies import *

# def megadetector(img_dir, num_images):
#     print("Megadetector model")
#     log = {}
#     local_detector_path = os.path.join(os.getcwd(), r"cameratraps\\detection\\run_detector_batch.py")
#     megadetector_path = os.path.join(os.getcwd(), "md_v5a.0.0.pt")
#     output_file_name = "_".join(img_dir.split("\\")[-3:])
#     json_dir = os.path.join(img_dir, f"{output_file_name}_megadetector.json")

#     if os.path.exists(json_dir):
#         print("Megadetector output file already exists.. Going for species classification")
#         log_dir = os.path.join(img_dir, f"{output_file_name}_log.json")
#         if os.path.exists(log_dir):
#             with open(log_dir, 'r') as f:
#                 log = json.load(f)
#         return json_dir, log
    
#     print(f"Saving detections at {json_dir}...")
    
#     command = [sys.executable,
#                local_detector_path,
#                megadetector_path,
#                img_dir,
#                json_dir,
#                "--recursive"]
    
#     prev_percentage = 0
#     with Popen(command,
#                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=1, shell=True,
#                universal_newlines=True) as p:
#         for line in p.stdout:
#             if line.startswith("Loaded model in"):
#                 print(line)
            
#             elif "%" in line[0:4]:
#                 percentage = re.search("\d*%", line[0:4])[0][:-1]
#                 if percentage > prev_percentage:
#                     prev_percentage = percentage
#                     print(percentage)
                

#     # command = [sys.executable,
#     #             local_detector_path,
#     #             megadetector_path,
#     #             img_dir,
#     #             json_dir]
    
#     # # with tqdm(total = min(100, num_images)) as t:
#     # prev_percentage = 0
#     # with Popen(command,
#     #         stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=1, shell=True,
#     #         universal_newlines=True) as p:
#     #     for line in p.stdout:
            
#     #         if line.startswith("Loaded model in"):
#     #             print(line)
            
#     #         elif "%" in line[0:4]:
#     #             percentage = int(re.search("\d*%", line[0:4])[0][:-1])
#     #             if percentage > prev_percentage:
#     #                 prev_percentage = percentage
#     #                 print(percentage)
#                     # t.update(1)
 
#     print("Bounding Boxes Created")

#     return json_dir, log

def get_detection_df(img_path, json_dir):
    print("Generating detections.csv...")

    with open(json_dir, 'r') as f:
        data = json.load(f)
        df = pd.DataFrame(data["images"])

    records = []
    for i, row in df.iterrows():
        filepath = row["file"]
        filename = os.path.splitext(os.path.basename(filepath))[0]
        detections = row["detections"]
        for j, detection in enumerate(detections):
            area = detection["bbox"][2] * detection["bbox"][3]
            y_position = detection["bbox"][1]
            if (
                detection["conf"] > 0.1
                and area > 0.001
                and not (area <= 0.01 and y_position > 0.6)
            ):
                if detection["category"] == '1':
                    category = "Animal"
                elif detection["category"] == '2':
                    category = "Person"
                else:
                    category = "Vehicle"

                records.append(
                    {
                        "Filepath": filepath,
                        "Filename": filename,
                        "Detection_number": j + 1,
                        "Category": category,
                        "Detection_Confidence": detection["conf"],
                        "Detection_bbox": detection["bbox"],
                    }
                )

    new_df = pd.DataFrame(records)
    new_df["Filepath"] = (img_path + "\\" + new_df["Filename"] + ".jpg").apply(clean_path)
    new_df["File_directory"] = new_df["Filepath"].apply(os.path.dirname)
    df_path = os.path.join(img_path, "detections.csv")
    new_df.to_csv(df_path, index=False)
    
    small_obj_df = new_df[new_df["Category"] == "Small Object"]
    if not small_obj_df.empty:
        small_obj_df_path = os.path.join(img_path, "small_objects.csv")
        small_obj_df.to_csv(small_obj_df_path, index=False)

    return new_df
    
def crop_img(img_dir, bbox):
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

def crop_img_gpu(img_dir, bbox):
    # Load the image using TensorFlow
    img = tf.io.read_file(img_dir)
    img = tf.image.decode_jpeg(img, channels=3)
    x, y, w, h = bbox
    mul_x = tf.cast(tf.shape(img)[1], tf.float32)
    mul_y = tf.cast(tf.shape(img)[0], tf.float32)
    x1 = tf.cast(x * mul_x, tf.int32)
    y1 = tf.cast(y * mul_y, tf.int32)
    x2 = tf.cast((x + w) * mul_x, tf.int32)
    y2 = tf.cast((y + h) * mul_y, tf.int32)
    cropped = img[y1:y2, x1:x2, :]
    cropped_np = cropped.numpy()
    # Convert the NumPy array to a PIL Image
    cropped_img = Image.fromarray(cropped_np)
    return cropped_img

# Define a lock for thread safety when modifying shared data
lock = threading.Lock()

# Define the function for processing a single row in the DataFrame
def crop_row(row):
    filepath = row[0]
    filename = row[1]
    d_num = row[2]
    bbox = row[5]
    directory = row[6]
    cropped_dir = row[7]
    
    cropped_img = crop_img(filepath, bbox)
    cropped_dir = os.path.join(directory,r"Cropped_images")
    cropped_name = f"{filename}_{d_num}.jpg"
    cropped_img_path = os.path.join(cropped_dir, cropped_name)
    os.makedirs(cropped_dir, exist_ok=True)
    cropped_img.save(cropped_img_path)
    with lock:
        return cropped_name

def crop_row_gpu(row):
    filepath = row[0]
    filename = row[1]
    d_num = row[2]
    bbox = row[5]
    directory = row[6]
    cropped_dir = row[7]
    
    cropped_img = crop_img_gpu(filepath, bbox)
    #cropped_dir = os.path.join(directory,r"Cropped_images")
    cropped_name = f"{filename}_{d_num}.jpg"
    cropped_img_path = os.path.join(cropped_dir, cropped_name)
    os.makedirs(cropped_dir, exist_ok=True)
    cropped_img.save(cropped_img_path)
    with lock:
        return cropped_name

def resize_img(filepath, IMG_SIZE = [224, 224]):
    img = tf.io.read_file(filepath)
    img = tf.image.decode_jpeg(img)
    img = tf.image.resize(img, IMG_SIZE)
    # img = tf.expand_dims(img, axis = 1)
    return img

def clean_path(path):
    return os.path.normpath(path)

def crop_images_batch_gpu(df, batch_size):   
    cropped_names=[]
    results = []
    # Split the DataFrame into batches
    num_batches, remainder = divmod(len(df), batch_size)
    
    for i in tqdm(range(num_batches)):
        batch = df[i * batch_size: (i + 1) * batch_size]
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [executor.submit(crop_row_gpu, row) for row in batch.itertuples(index=False, name=None)]
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            results.append(result)
        gc.collect()
    if remainder > 0:
        remaining_data = df[num_batches * batch_size:]
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [executor.submit(crop_row_gpu, row) for row in remaining_data.itertuples(index=False, name=None)]
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            results.append(result)

    for result in results:
        cropped_name = result
        cropped_names.append(cropped_name)
    return df

def crop_images_batch(df,batch_size=512):
    cropped_names = []
    # Use tqdm to track progress
    with tqdm(total=len(df)) as pbar:
        def update_progress(_):
            pbar.update(1)

        with concurrent.futures.ProcessPoolExecutor() as executor:
            futures = [executor.submit(crop_row, row) for row in df.itertuples(index=False, name=None)]
            concurrent.futures.wait(futures, return_when=concurrent.futures.ALL_COMPLETED)

            # Collect results and update progress
            for future in futures:
                result = future.result()
                cropped_names.append(result)
                update_progress(None)
                
    df["Cropped_image_name"] = cropped_names
    return df

def delete_images_batch(src_list, batch_size=512):
    src_files = src_list
    with concurrent.futures.ProcessPoolExecutor() as exe:
        batch_tasks = []
        for i in tqdm(range(0, len(src_files), batch_size)):
            src_batch = src_files[i:i + batch_size]
            batch_tasks.extend([exe.submit(os.remove, src) for src in src_batch])
            # Wait for all tasks in the batch to complete
            _ = [task.result() for task in batch_tasks]
    return

def copy_images_batch(src_list, dest_list, batch_size=512):
    src_files=src_list
    dest_files=dest_list    
    with concurrent.futures.ThreadPoolExecutor() as exe:
        batch_tasks = []
        for i in tqdm(range(0, len(src_files), batch_size)):
            src_batch = src_files[i:i + batch_size]
            dest_batch = dest_files[i:i + batch_size]
            batch_tasks.extend([exe.submit(shutil.copy, src, dest) for src, dest in zip(src_batch, dest_batch)])
            # Wait for all tasks in the batch to complete before proceeding to the next batch
            _ = [task.result() for task in batch_tasks]
    return

def predict_lower_level_species(cropped_dir, folder, class_names, model, level):
    images = os.path.join(cropped_dir, folder)
    img_dir = os.path.join(cropped_dir, folder[:-2])
    dataset = tf.data.Dataset.list_files(images, shuffle = False)
    img_names = []
    for file_path in dataset:
        cropped_name = file_path.numpy().decode("utf-8").split("\\")[-1]
        img_names.append(cropped_name)
    dataset = dataset.map(resize_img, num_parallel_calls=tf.data.AUTOTUNE).batch(batch_size=64).prefetch(buffer_size=tf.data.AUTOTUNE)

    preds = model.predict(dataset)
    num_images = len(preds)
    pred_classes = []
    pred_probs=[]
    
    for pred in preds:
        if max(pred) >= 0.8:
            species = class_names[np.argmax(pred)]
        else:
            if level == "Species":
                species = folder[:-2]
            else:
                species = "Others"
        pred_classes.append(species)
        pred_probs.append(max(pred))
    
    if level == "Species":
        new_df = pd.DataFrame({
            'Cropped_image_name': img_names,
            'Species_pred': pred_classes,
            'Species_pred_prob': pred_probs
        })
    else:
        new_df = pd.DataFrame({
            'Cropped_image_name': img_names,
            'Order_pred': pred_classes,
            'Order_pred_prob': pred_probs
        })
        pass
    return new_df, num_images


def move_images_batch(src_list, dest_list, batch_size=512):
    src_files = src_list
    dest_files = dest_list

    with tqdm(total=len(src_files), desc="Copying images") as pbar_copy:
        with concurrent.futures.ThreadPoolExecutor() as exe_thread:
            # Copy images to the destination directory
            batch_tasks_copy = []
            for i in range(0, len(src_files), batch_size):
                src_batch = src_files[i:i + batch_size]
                dest_batch = dest_files[i:i + batch_size]
                batch_tasks_copy.extend([exe_thread.submit(shutil.copy, src, dest) for src, dest in zip(src_batch, dest_batch)])

            # Wait for all copy tasks in the batch to complete
            _ = [pbar_copy.update(1) for _ in concurrent.futures.as_completed(batch_tasks_copy)]
            gc.collect()

    with tqdm(total=len(src_files), desc="Removing source images") as pbar_remove:
        with concurrent.futures.ProcessPoolExecutor() as exe_process:
            # Now, remove the source images
            batch_tasks_remove = []
            for i in range(0, len(src_files), batch_size):
                src_batch = src_files[i:i + batch_size]
                batch_tasks_remove.extend([exe_process.submit(os.remove, src) for src in src_batch])

            # Wait for all remove tasks in the batch to complete
            _ = [pbar_remove.update(1) for _ in concurrent.futures.as_completed(batch_tasks_remove)]
            gc.collect()
        
        print(f"All {len(src_files)} images copied and removed at {datetime.now()}")
    return