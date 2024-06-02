from dependencies import *

def megadetector(img_dir, num_images):
    print("Megadetector model")
    log = {}
    local_detector_path = os.path.join(os.getcwd(), r"MegaDetector\megadetector\detection\run_detector_batch.py")
    megadetector_path = os.path.join(os.getcwd(), "md_v5a.0.0.pt")
    output_file_name = "_".join(img_dir.split("\\")[-3:])
    json_dir = os.path.join(img_dir, f"{output_file_name}_megadetector.json")

    if os.path.exists(json_dir):
        print("Megadetector output file already exists.. Going for species classification")
        log_dir = os.path.join(img_dir, f"{output_file_name}_log.json")
        if os.path.exists(log_dir):
            with open(log_dir, 'r') as f:
                log = json.load(f)
        print(f"Saving detections at {json_dir}...")           
    
    else:                
        command = [sys.executable,
                   local_detector_path,
                   megadetector_path,
                   img_dir,
                   json_dir]
           
        print(f"Detecting objects in {num_images} images...")
        
        with tqdm(total = 100) as t:
            prev_percentage = 0
            with Popen(command,
                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=1, shell=True,
                    universal_newlines=True) as p:
                for line in p.stdout:
                    
                    if line.startswith("Loaded model in"):
                        pass
                    
                    elif "%" in line[0:4]:
                        percentage = int(re.search("\d*%", line[0:4])[0][:-1])
                        if percentage > prev_percentage:
                            prev_percentage = percentage
                            t.update(1)
                            
        print("Bounding Boxes Created")
    return json_dir, log