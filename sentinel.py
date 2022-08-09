import os
import xarray as xr
from osgeo import gdal, gdalconst
from xrspatial.multispectral import true_color
from helper_functions import get_all_files


sentinel_ids = {
    "T31UFT": "0",
    "T33UUT": "1",
    "T33UUU": "2",
    "T32VNM": "3",
    "T30TVP": "4",
    "T30TWP": "5",
    "T32VPM": "7",
}


class SentinelScene:
    """
    SentinelScene class. It holds all crucial data of a Sentinel scene - scene path and bands images paths. Each scene
    consists of a set of images taken at different times.

    Attributes
    ----------
    path : str
        Path to the Sentinel scene, for example 0_31UFT.
    images_paths : dictionary
        Dictionary of Sentinel images. It is a dictionary where keys are the bands names and the values are lists of
        paths to the sentinel images taken at different times.


    Methods
    -------
    __init__()
        Initializes SentinelScene object.
    init_sentinel_images_paths():
        Initializes a images_paths dict with sentinel images paths.

    """
    def __init__(self, path):
        """Initializes SentinelScene object.

         Parameters
         ----------
         path : str
            Path to the Sentinel scene, for example 0_31UFT.
         images_paths : dictionary
            Dictionary of Sentinel images. It is a dictionary where keys are the bands names and the values are lists of
            paths to the sentinel images taken at different times.
        """
        super(SentinelScene, self).__init__()
        self.path = path
        self.images_paths = self.init_sentinel_images_paths()

    def init_sentinel_images_paths(self):
        data = {}
        for band in os.listdir(self.path):
            if band not in ["WorldView", "Metadata", "CLD"] and not band.startswith("."):
                data.update({band: get_all_files(os.path.join(self.path, band))})
            if band == "CLD":
                data.update({band + "_20m": get_all_files(os.path.join(*[self.path, band, "20m"]))})
                data.update({band + "_60m": get_all_files(os.path.join(*[self.path, band, "60m"]))})
        return data


def get_color_sentinel_img(blue_img_path, green_img_path, red_image_path):
    """Returns an RGB image based on RGB bands of Sentinel image.

    Parameters
    ----------
    blue_img_path : str
         Path to the blue Sentinel band image.
    green_img_path : str
         Path to the green Sentinel band image.
    red_img_path : str
         Path to the red Sentinel band image.

    Returns
    -------
    color_image : numpy.array
        RGB Sentinel image.

   """
    paths = [blue_img_path, green_img_path, red_image_path]
    channels = []
    for path in paths:
        img = gdal.Open(path, gdalconst.GA_ReadOnly)
        img_array = img.ReadAsArray()
        img_array = xr.DataArray(data=img_array,  dims=["y", "x"],)
        channels.append(img_array)
    color_image = true_color(r=channels[0], g=channels[1], b=channels[2])
    return color_image.data