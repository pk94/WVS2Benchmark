import os
from osgeo import gdal, gdalconst
from xrspatial.multispectral import true_color
import xarray as xr
from helper_functions import get_all_files


class WorldViewScene:
    """
    WorldViewScene class. It holds all crucial data of a WorldView scene - type (multispectral or panchromatic) and
    paths to the scene tiles.

    Attributes
    ----------
    path : str
        Path to the WorldView scene, for example 0_31UFT\WorldView\MUL\11JUN03110252-M.
    type : str
        WorldView scene type - multispectral (MUL) or panchromatic (PAN).
    images_paths : int
        Paths of specific tiles of WorldView scene tiles.


    Methods
    -------
    __init__()
        Initializes WorldViewScenes object.

    """
    def __init__(self, path):
        """Initializes WorldViewScenes object.

         Parameters
         ----------
         path : str
            Path to the WorldView scene, for example 0_31UFT\WorldView\MUL\11JUN03110252-M.
         type : str
            WorldView scene type - multispectral (MUL) or panchromatic (PAN).
         images_paths : list
            Paths of specific tiles of WorldView scene tiles.
        """
        super(WorldViewScene, self).__init__()
        self.path = path
        self.type = "MUL" if os.path.basename(self.path).split("-")[-1] == "M" else "PAN"
        self.images_paths = get_all_files(self.path)


def get_color_worldview_img(img_path):
    """Returns an RGB image based on multipsectral WorldView image.

    Parameters
    ----------
     img_path : str
         Path to the multispectral WorldView image.

    Returns
    -------
    color_image : numpy.array
        RGB WorldView image.

   """
    img = gdal.Open(img_path, gdalconst.GA_ReadOnly)
    img_array = img.ReadAsArray()
    r = xr.DataArray(data=img_array[1, ...], dims=["y", "x"])
    g = xr.DataArray(data=img_array[2, ...], dims=["y", "x"])
    b = xr.DataArray(data=img_array[4, ...], dims=["y", "x"])
    color_image = true_color(r=r, g=g, b=b)
    return color_image.data
