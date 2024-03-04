from dependencies import *

class Geotag():
    def __init__(self, lat, long) -> None:

        self.EXIF_PATH = r"C:\Windows\exiftool.exe"
        self.LATITUDE = lat
        self.LONGITUDE = long
        pass

    def add_gps_info(self, image_dir):
        command = [self.EXIF_PATH, '-GPSLatitude=' + str(self.LATITUDE), '-GPSLongitude=' + str(self.LONGITUDE), '-overwrite_original','-r',image_dir]
        try:
            subprocess.run(command, check=True)
            print(f"GPS information added to {image_dir} successfully.")
        except subprocess.CalledProcessError as e:
            print(f"Error while processing {image_dir}: {e}")

        return

    def run(self, unique_directories):
        start = datetime.now()
        for d in unique_directories:
            self.add_gps_info(d)
        end= datetime.now()
        print(f"Geotagging completed in {end-start}")

        return