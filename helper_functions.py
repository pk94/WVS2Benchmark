from os.path import exists, join, isfile
from os import makedirs, listdir
import json
import torch
import numpy as np
import cv2
import torchvision.models as models
import torchvision.transforms as transforms
from collections import namedtuple
from osgeo import gdal, gdalconst
from pyproj import CRS, Transformer

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class StatsUpdate:
    def __init__(self):
        self.n = 0
        self.M = 0
        self.S = 0

    def update(self, x):
        x_mean = np.mean(x)
        x_std = np.std(x)
        newM = (self.n * self.mean + x.size * x_mean) / (self.n + x.size)
        correction_factor = (self.n * x.size * (self.M - x_mean) ** 2) / ((self.n + x.size) * (self.n + x.size - 1))
        newS = ((self.n - 1) * self.S + (x.size - 1) * x_std) / (self.n + x.size - 1) + correction_factor
        self.M = newM
        self.S = newS
        self.n += x.size

    @property
    def mean(self):
        return self.M

    @property
    def std(self):
        return self.S

    def return_dict(self):
        return {'mean': self.M, "std": self.S, "num_samples": self.n}


class vgg16(torch.nn.Module):
    def __init__(self, requires_grad=False, pretrained=True):
        super(vgg16, self).__init__()
        vgg_pretrained_features = models.vgg16(pretrained=pretrained).features.to(DEVICE).eval()
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        self.N_slices = 5
        for x in range(4):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(9, 16):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(16, 23):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(23, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h = self.slice1(X)
        h_relu1_2 = h
        h = self.slice2(h)
        h_relu2_2 = h
        h = self.slice3(h)
        h_relu3_3 = h
        h = self.slice4(h)
        h_relu4_3 = h
        h = self.slice5(h)
        h_relu5_3 = h
        vgg_outputs = namedtuple("VggOutputs", ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3', 'relu5_3'])
        out = vgg_outputs(h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3, h_relu5_3)

        return out


class FeatureExtractor:
    def __init__(self, config):
        self.vgg16 = vgg16()
        self.config = config
        self.features_distributions_hr = {}
        self.features_distributions_lr = {}
        self.init_features_distributions()

    def init_features_distributions(self):
        self.features_distributions_hr = {band: {
            0: {ii: StatsUpdate() for ii in range(64)},
            1: {ii: StatsUpdate() for ii in range(128)},
            2: {ii: StatsUpdate() for ii in range(256)},
            3: {ii: StatsUpdate() for ii in range(512)},
            4: {ii: StatsUpdate() for ii in range(512)}
        } for band in self.config["dataset"]["bands"]}

        self.features_distributions_lr = {band: {
            0: {ii: StatsUpdate() for ii in range(64)},
            1: {ii: StatsUpdate() for ii in range(128)},
            2: {ii: StatsUpdate() for ii in range(256)},
            3: {ii: StatsUpdate() for ii in range(512)},
            4: {ii: StatsUpdate() for ii in range(512)}
        } for band in self.config["dataset"]["bands"]}

    def __call__(self, img):
        img = self.prepare_input(img)
        activations = self.vgg16(img)
        return activations

    @staticmethod
    def prepare_input(img):
        img = (img - 127.5) / 127.5
        img = np.stack([img for _ in range(3)])
        img = torch.from_numpy(img).to(DEVICE).type(torch.cuda.FloatTensor)
        return img

    @staticmethod
    def prepare_output(img):
        img = img.detach().cpu().numpy()
        return img

    def save_activations(self, activations):
        makedirs("test", exist_ok=True)
        for ii, activation in enumerate(activations):
            makedirs(f"test/{ii}", exist_ok=True)
            for jj in range(activation.shape[0]):
                img = self.prepare_output(activation[jj, ...])
                cv2.imwrite(f"test/{ii}/{jj}.jpg", img)

    def save_stats(self):
        features_distributions_hr = {band: {
            0: {ii: self.features_distributions_hr[band][0][ii].return_dict() for ii in range(64)},
            1: {ii: self.features_distributions_hr[band][1][ii].return_dict() for ii in range(128)},
            2: {ii: self.features_distributions_hr[band][2][ii].return_dict() for ii in range(256)},
            3: {ii: self.features_distributions_hr[band][3][ii].return_dict() for ii in range(512)},
            4: {ii: self.features_distributions_hr[band][4][ii].return_dict() for ii in range(512)}
        } for band in self.config["dataset"]["bands"]}

        features_distributions_lr = {band: {
            0: {ii: self.features_distributions_lr[band][0][ii].return_dict() for ii in range(64)},
            1: {ii: self.features_distributions_lr[band][1][ii].return_dict() for ii in range(128)},
            2: {ii: self.features_distributions_lr[band][2][ii].return_dict() for ii in range(256)},
            3: {ii: self.features_distributions_lr[band][3][ii].return_dict() for ii in range(512)},
            4: {ii: self.features_distributions_lr[band][4][ii].return_dict() for ii in range(512)}
        } for band in self.config["dataset"]["bands"]}

        stats_dict = {
            "hr": features_distributions_hr,
            "lr": features_distributions_lr
        }
        with open(f"{self.config['results_save_path']}/stats.json", "w") as f:
            json.dump(stats_dict, f)


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


def min_max_norm(array, min_val, max_val, new_min, new_max):
    return (array - min_val) * (new_max - new_min) / (max_val - min_val) + new_min \
        if min_val != max_val else np.zeros_like(array)


def z_score_norm(array, mean, std):
    return (array - mean) / std if array.any() else array
