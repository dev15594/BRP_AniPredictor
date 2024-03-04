from dependencies import *

class Corrupt_Files():
    def __init__(self):
        pass
        # super().__init__

    def list_corrupt_files_in_directory(self, directory):
        corrupt_files=[]
        for root, dirs, files in os.walk(directory):
            for file in files:
                file_path = os.path.join(root, file)
                try:
                    # Use the os.path.getsize() method to check if the file is empty or very small
                    file_size = os.path.getsize(file_path)
                    if file_size < 10:  # Adjust the threshold as needed
                        corrupt_files.append(file_path)
                        print(f"Corrupt file: {file_path}")
                    else:
                        continue
                except Exception as e:
                    # Handle other exceptions that might occur
                    print(f"Error processing file {file_path}: {e}")
        return corrupt_files

    def delete_corrupt_files(self, corrupt_files, input_dir):

        # corrupt_files=self.list_corrupt_files_in_directory(input_dir)
        if len(corrupt_files) > 0:
            for c in corrupt_files:
                c_path = os.path.join(input_dir, c)
                os.remove(c_path)
        print(f"Corrupt files : {len(corrupt_files)} images removed")

        return
