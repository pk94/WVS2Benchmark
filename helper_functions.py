from os.path import exists, join, isfile
from os import makedirs, listdir
from osgeo import gdal, gdalconst
from pyproj import CRS, Transformer


def get_geo_coordinates(img_path):
    """Extracts the geographical location information from geo images metadata, such as jp2 or TIF images.

    Parameters
    ----------
    img_path : str
         Input geo image path.

    Returns
    -------
    coordinates : dictionary
        Dict of coordinates of each corner of an image given as easting and northing or latitude and longitude.
    srs_coordinates: list
        List of coordinates in original image's SRS.

   """
    src = gdal.Open(img_path, gdalconst.GA_ReadOnly)
    epsg = gdal.Info(src, format='json')['coordinateSystem']['wkt'].rsplit('"EPSG",', 1)[-1].split(']]')[0]
    ulx, xres, xskew, uly, yskew, yres = src.GetGeoTransform()
    lrx = ulx + (src.RasterXSize * xres)
    lry = uly + (src.RasterYSize * yres)
    srs_coordinates = [ulx, uly, lrx, lry]
    crs = CRS.from_epsg(epsg)
    proj = Transformer.from_crs(crs, crs.geodetic_crs, always_xy=True)
    coordinates = {}
    point = proj.transform(ulx, uly)
    coordinates.update({"ul": {"easting": ulx, "northing": uly, "latitude": point[1], "longitude": point[0]}})
    point = proj.transform(lrx, uly)
    coordinates.update({"ur": {"easting": lrx, "northing": uly, "latitude": point[1], "longitude": point[0]}})
    point = proj.transform(lrx, lry)
    coordinates.update({"lr": {"easting": lrx, "northing": lry, "latitude": point[1], "longitude": point[0]}})
    point = proj.transform(ulx, lry)
    coordinates.update({"ll": {"easting": ulx, "northing": lry, "latitude": point[1], "longitude": point[0]}})
    return coordinates, srs_coordinates


def get_all_files(path):
    """Extracts files paths from a given directory.

    Parameters
    ----------
    path : str
         Path to the directory containing files.

    Returns
    -------
    files_list : list
        List of full files paths.

   """
    return [join(path, f) for f in listdir(path) if isfile(join(path, f))]


def create_directory(path):
    if not exists(path):
        makedirs(path)
