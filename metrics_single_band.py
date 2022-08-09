from os import listdir
from os.path import join, basename
import cv2
import numpy as np
import lpips
import torch
import pandas as pd
import yaml
from osgeo import gdal, gdalconst
from skimage.exposure import match_histograms
from skimage.metrics import structural_similarity as ssim
from tqdm import tqdm
from helper_functions import get_all_files, create_directory

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


INTERPOLATION_METHODS = {
    "Linear": cv2.INTER_LINEAR,
    "NN": cv2.INTER_NEAREST,
    "Bicubic": cv2.INTER_CUBIC,
    "Area": cv2.INTER_AREA,
    "Lanczos": cv2.INTER_LANCZOS4
}

SR_METHODS_NAMES = {"HighResNet_p96_b32_cMSE_ProbaV_bNIR_20220105-143437": "hrn_nir",
                    "HighResNet_p96_b32_cMSE_ProbaV_bRED_20220209-101557": "hrn_red",
                    "HighResNet_p96_b32_cMSE_s2ab_ab5_bb8_20220213-122958": "hrn_sim",
                    "RAMS_p96_b32_cMSE_ProbaV_bNIR_20220314-110156": "rams_nir",
                    "RAMS_p96_b32_cMSE_ProbaV_bRED_20220310-100617": "rams_red",
                    "RAMS_p96_b32_cMSE_s2ab_ab5_bb8_20220318-152516": "rams_sim"}


def cPSNR(sr, hr, hr_map):
    n_clear = np.sum(hr_map)
    diff = hr - sr
    bias = np.sum(diff * hr_map) / n_clear
    cmse = np.sum(np.square((diff - bias) * hr_map)) / n_clear
    cpsnr = -10 * np.log10(cmse)
    return cpsnr


def cSSIM(sr, hr, hr_map):
    n_clear = np.sum(hr_map)
    diff = hr - sr
    bias = np.sum(diff * hr_map) / n_clear
    cssim = ssim((sr + bias) * hr_map, hr * hr_map, data_range=255)
    return cssim


def get_masked_sentinel_image(sentinel_img, wv_img, mask):
    new_shape = (sentinel_img.shape[1], sentinel_img.shape[0])
    mask = cv2.resize(mask.astype(np.uint8), new_shape, interpolation=cv2.INTER_AREA)
    inverse_mask = np.invert(mask.astype(np.bool_)).astype(np.uint8)
    wv_masked = cv2.bitwise_and(wv_img, wv_img, mask=mask)
    resized_masked_sentinel = cv2.bitwise_and(sentinel_img, sentinel_img, mask=inverse_mask)
    resized_sentinel = wv_masked + resized_masked_sentinel
    return resized_sentinel, mask


class MetricsSingleBand:
    def __init__(self):
        self.ssim = ssim
        self.lpips = lpips.LPIPS(net='alex').to(DEVICE)
        self.shift = None
        self.metrics = {
            "PSNR": self.get_psnr,
            "SSIM": self.get_ssim,
            "LPIPS": self.get_lpips,
            "cPSNR": self.get_cpsnr,
            "cSSIM": self.get_cssim
        }

    @staticmethod
    def get_psnr(sr, hr, mask=None):
        mask = np.ones_like(hr) if mask is None else \
            np.invert(cv2.resize(mask.astype(np.uint8), (hr.shape[1], hr.shape[0]),
                                 interpolation=cv2.INTER_AREA).astype(np.bool_))
        if np.max(sr) > 1:
            sr = sr / 255
        if np.max(hr) > 1:
            hr = hr / 255
        n_clear = np.sum(mask)
        mse = np.sum(((sr - hr) * mask) ** 2) / n_clear
        if mse == 0:
            return 361
        psnr = -10 * np.log10(mse)
        return psnr

    def get_ssim(self, sr, hr, mask):
        mask = np.ones_like(hr) if mask is None else \
            np.invert(cv2.resize(mask.astype(np.uint8), (hr.shape[1], hr.shape[0]),
                                 interpolation=cv2.INTER_AREA).astype(np.bool_))
        return self.ssim(sr * mask, hr * mask, data_range=255)

    def get_lpips(self, sr, hr, mask):
        def prepare_input_tensor(img):
            img_interp = (img - 127.5) / 127.5
            img_interp = np.expand_dims(img_interp, axis=0)
            img_tensor = torch.tensor(img_interp, device=DEVICE).float()
            return img_tensor

        if mask is not None:
            sr, _ = get_masked_sentinel_image(sr, hr, mask)
        img_1_tensor = prepare_input_tensor(sr)
        img_2_tensor = prepare_input_tensor(hr)
        return float(self.lpips(img_1_tensor, img_2_tensor))

    def get_cpsnr(self, sr, hr, mask=None, max_shift=3):
        if np.max(sr) > 1:
            sr = sr / 255
        if np.max(hr) > 1:
            hr = hr / 255
        sr_cropped = sr[max_shift:-max_shift, max_shift:-max_shift]
        shape = hr.shape
        mask = np.ones_like(hr) if mask is None else \
            np.invert(cv2.resize(mask.astype(np.uint8), (hr.shape[1], hr.shape[0]),
                                 interpolation=cv2.INTER_AREA).astype(np.bool_))
        cpsnr_values = []
        shifts = []
        for ii in range(2 * max_shift):
            for jj in range(2 * max_shift):
                ii_end = shape[0] - (2 * max_shift - ii)
                jj_end = shape[1] - (2 * max_shift - jj)
                registered_hr = hr[ii:ii_end, jj:jj_end]
                registered_mask = mask[ii:ii_end, jj:jj_end]
                cpsnr = cPSNR(sr_cropped, registered_hr, registered_mask)
                cpsnr_values.append(cpsnr)
                shifts.append((ii, jj))
        self.shift = shifts[np.argmax(cpsnr_values)]
        return np.max(cpsnr_values)

    def get_cssim(self, sr, hr, mask=None, max_shift=3):
        hr = hr.astype(float)
        sr = sr.astype(float)
        sr_cropped = sr[max_shift:-max_shift, max_shift:-max_shift]
        shape = hr.shape
        mask = np.ones_like(hr) if mask is None else \
            np.invert(cv2.resize(mask.astype(np.uint8), (hr.shape[1], hr.shape[0]),
                                 interpolation=cv2.INTER_AREA).astype(np.bool_))
        ii = self.shift[0]
        jj = self.shift[1]
        ii_end = shape[0] - (2 * max_shift - ii)
        jj_end = shape[1] - (2 * max_shift - jj)
        registered_hr = hr[ii:ii_end, jj:jj_end]
        registered_mask = mask[ii:ii_end, jj:jj_end]
        cssim = cSSIM(sr_cropped, registered_hr, registered_mask)
        return cssim


class Scene:
    def __init__(self, path):
        self._path = path
        self._id = basename(path)
        self._band_matches = {
            "b2": 1,
            "b3": 2,
            "b4": 4,
            "b8": 6
        }
        self._wv_multispectral_img = self._load_wv_multispectral_img()
        self.sentinel_images = []

    @property
    def id(self):
        return self._id

    def _load_wv_multispectral_img(self):
        img = gdal.Open(join(self._path, "hr_mul.tif"), gdalconst.GA_ReadOnly)
        img_array = img.ReadAsArray()
        return img_array.astype(float)

    def _get_resized_wv_band_img(self, new_shape, band, interpolation_method_hr):
        new_shape = (new_shape[1], new_shape[0])
        wv_band_image = self._wv_multispectral_img[self._band_matches[band], :]
        interpolated_wv_img = np.interp(wv_band_image, (wv_band_image.min(), wv_band_image.max()), (0, 255))
        wv_resized = cv2.resize(interpolated_wv_img.astype(np.uint8), new_shape, interpolation=interpolation_method_hr)
        return wv_resized

    def _get_resized_sentinel_band_imgs(self, resize_factor, interpolation_method_lr):
        new_shape = (int(self.sentinel_images[0].shape[1] / resize_factor),
                     int(self.sentinel_images[0].shape[0] / resize_factor))
        sentinel_resized = [cv2.resize(img.astype(np.uint8), new_shape, interpolation=interpolation_method_lr)
                            for img in self.sentinel_images]
        return sentinel_resized

    @staticmethod
    def _match_sentinel_wv_histograms(sentinel_resized_band_images, wv_resized_band_img):
        sentinel_matched_band_images = [match_histograms(sentinel_image.astype(np.uint8), wv_resized_band_img)
                                        for sentinel_image in sentinel_resized_band_images]
        sentinel_matched_mean_img = match_histograms(np.mean(sentinel_resized_band_images, axis=0).astype(np.uint8),
                                                     wv_resized_band_img)
        return sentinel_matched_band_images, sentinel_matched_mean_img

    def load_sentinel_band_images(self, band):
        files = get_all_files(join(*[self._path, band, "lrs"]))
        band_images = []
        for file in files:
            img = gdal.Open(file, gdalconst.GA_ReadOnly)
            img_array = img.ReadAsArray().astype(float)
            band_images.append(np.interp(img_array, (img_array.min(), img_array.max()), (0, 255)))
        return band_images

    def get_resized_band_images(self, band, resize_factor, interpolation_method_lr, interpolation_method_hr):
        sentinel_resized_band_images = self._get_resized_sentinel_band_imgs(resize_factor, interpolation_method_lr)
        wv_resized_band_img = self._get_resized_wv_band_img(sentinel_resized_band_images[0].shape,
                                                            band, interpolation_method_hr)
        sentinel_matched_band_images, sentinel_matched_mean_img = self._match_sentinel_wv_histograms(
            sentinel_resized_band_images, wv_resized_band_img)
        return wv_resized_band_img, sentinel_matched_band_images, sentinel_matched_mean_img

    def get_sr_wv_images(self, band, reconstructed_images_path, reconstruction_method, interpolation_method_hr):
        sr_images_paths = join(*[reconstructed_images_path, band, reconstruction_method])
        sr_image_path = [join(sr_images_paths, img_path) for img_path in listdir(sr_images_paths)
                         if self.id == img_path.split(".")[0]][0]
        sr_image = cv2.imread(sr_image_path, cv2.IMREAD_GRAYSCALE)
        wv_resized_band_img = self._get_resized_wv_band_img(sr_image.shape, band, interpolation_method_hr)
        sr_matched_image = match_histograms(sr_image, wv_resized_band_img)
        return wv_resized_band_img, sr_matched_image


class Mask:
    def __init__(self, mask_mode, mask_path):
        self._mask_mode = mask_mode
        self._mask_path = mask_path
        self._mask = None
        self._interpolations = []
        self._reconstructions = []
        self.hr_img = None

    def __add__(self, other):
        self._mask = np.bitwise_or(self._mask, other.mask)

    @property
    def mask(self):
        return self._mask

    @property
    def mask_mode(self):
        return self._mask_mode

    @property
    def interpolations(self):
        return self._interpolations

    @property
    def reconstructions(self):
        return self._reconstructions

    def load_mask(self, mask_img_path):
        mask = cv2.imread(mask_img_path, cv2.IMREAD_GRAYSCALE)
        self._mask = mask.astype(np.bool_)

    def build_score_mask(self, save_path, scene_id, band):
        create_directory(join(save_path, f"score_masks/score_masks_{band}/pairs"))
        hr_img = self.hr_img.astype(np.float64)
        final_mask = np.zeros_like(hr_img).astype(np.bool_)
        for interpolation_name, interpolation in self._interpolations:
            interpolation = interpolation.astype(np.float64)
            interpolation_diff = np.abs(interpolation - hr_img)
            for reconstruction_name, reconstruction in self._reconstructions:
                reconstruction = reconstruction.astype(np.float64)
                reconstruction_diff = np.abs(reconstruction - hr_img)
                score_mask = reconstruction_diff < interpolation_diff
                final_mask = np.bitwise_or(final_mask, score_mask)
                cv2.imwrite(join(*[save_path, f"score_masks/score_masks_{band}/pairs",
                                   f"{interpolation_name}_{reconstruction_name}_{scene_id}.png"]),
                            np.invert(score_mask) * 255)
        cv2.imwrite(join(*[save_path, f"score_masks/score_masks_{band}/", f"mask_{scene_id}.png"]),
                    np.invert(final_mask) * 255)

    def _get_difference_mask(self, img_1, img_2, threshold_level=30):
        diff = np.abs(np.subtract(img_1.astype(np.float64), img_2.astype(np.float64)))
        self._mask = diff > threshold_level

    def _get_variance_mask(self, img_1, img_2):
        raise NotImplementedError()


class SingleEvaluationResults:
    def __init__(self, band, mask_mode, reconstruction_method, interpolation_hr, resize_factor):
        self._band = band
        self._mask_mode = mask_mode
        self._reconstruction_method = reconstruction_method
        self._interpolation_hr = interpolation_hr
        self._resize_factor = resize_factor
        self._results = []

    @property
    def results(self):
        return self._results

    @property
    def reconstruction_method(self):
        return self._reconstruction_method

    def filter(self, band, mask_mode, interpolation_hr, reconstruction_method, resize_factor):
        return band == self._band and mask_mode == self._mask_mode and interpolation_hr == self._interpolation_hr and \
               reconstruction_method == self._reconstruction_method and resize_factor == self._resize_factor

    def save(self, save_path):
        df = pd.DataFrame(self._results)
        df.to_csv(join(
            *[save_path,
              "per_scene",
              f"{self._band}_{self._interpolation_hr}_{self._reconstruction_method}_{self._mask_mode}_"
              f"{self._resize_factor}.csv"]), index=False)


class EvaluationResults:
    def __init__(self, config):
        self._config = config
        self._results_database = self._build_results_database()

    def _build_results_database(self):
        results_database = []
        for band in self._config["dataset"]["bands"]:
            for reconstruction_method in self._config["interpolations"]["LR"] + \
                                         list(self._config["reconstructions"].keys()
                                              if self._config["reconstructions"] else []):
                for interpolation_hr in self._config["interpolations"]["HR"]:
                    for mask_mode, _ in self._config["dataset"]["masks_paths"].items():
                        results_database.append(SingleEvaluationResults(
                            band, mask_mode, reconstruction_method, interpolation_hr, self._config["resize_factor"]))
        return results_database

    def _get_mean_evaluation_results(self, results):
        mean_results = []
        reconstruction_methods = []
        for result in results:
            df = pd.DataFrame(result.results)[self._config["metrics"]]
            reconstruction_methods.append(result.reconstruction_method)
            mean_results.append(df.mean(axis=0))
        mean_results = pd.DataFrame(mean_results)
        mean_results.insert(0, "Interpolation_method", reconstruction_methods)
        return mean_results

    def add_element(self, element, band, mask_mode, interpolation_hr, reconstruction_method, resize_factor):
        idx = 0
        for ii, single_ev_result in enumerate(self._results_database):
            idx = ii
            if single_ev_result.filter(band, mask_mode, interpolation_hr, reconstruction_method, resize_factor):
                break
        self._results_database[idx].results.append(element)

    def save_evaluation_results(self):
        for single_ev_result in self._results_database:
            single_ev_result.save(self._config["results_save_path"])
        reconstruction_methods = self._config["interpolations"]["LR"] + list(self._config["reconstructions"].keys()
                                                                             if self._config["reconstructions"] else [])
        for band in self._config["dataset"]["bands"]:
            for interpolation_hr in self._config["interpolations"]["HR"]:
                for mask_mode, _ in self._config["dataset"]["masks_paths"].items():
                    results = []
                    for result in self._results_database:
                        for reconstruction_method in reconstruction_methods:
                            if result.filter(band, mask_mode, interpolation_hr, reconstruction_method,
                                             self._config["resize_factor"]):
                                results.append(result)
                    mean_results = self._get_mean_evaluation_results(results)
                    mean_results.to_csv(
                        join(*[self._config["results_save_path"], "mean",
                               f"{band}_{interpolation_hr}_{mask_mode}_{self._config['resize_factor']}.csv"]),
                        index=False)


class Evaluation:
    def __init__(self, config_path):
        self._config = self.load_config(config_path=config_path)
        self._results = EvaluationResults(self._config)
        self._metrics = MetricsSingleBand()
        self._create_results_save_path()

    @staticmethod
    def load_config(config_path):
        with open(config_path) as config_file:
            config_data = yaml.full_load(config_file)
        config_file.close()
        return config_data

    def _create_results_save_path(self):
        create_directory(self._config["results_save_path"])
        create_directory(join(self._config["results_save_path"], "per_scene"))
        create_directory(join(self._config["results_save_path"], "mean"))

    def calculate_metrics(self, pairs_path):
        scenes_paths = [join(pairs_path, file) for file in listdir(pairs_path)]
        for scene_path in tqdm(scenes_paths):
            scene = Scene(scene_path)
            self._calculate_metrics_per_band(scene)
        self._results.save_evaluation_results()

    def _calculate_metrics_per_band(self, scene):
        for band in self._config["dataset"]["bands"]:
            scene.sentinel_images = scene.load_sentinel_band_images(band)
            self._calculate_metrics_per_mask(scene, band)

    def _calculate_metrics_per_mask(self, scene, band):
        for mask_mode, masks_path in self._config["dataset"]["masks_paths"].items():
            masks_path += f"_{band}"
            self._calculate_metrics_per_hr_inter(scene, band, mask_mode, masks_path)

    def _calculate_metrics_per_hr_inter(self, scene, band, mask_mode, masks_path):
        for interpolation_hr in self._config["interpolations"]["HR"]:
            self._calculate_metrics_per_lr_inter(scene, band, mask_mode, masks_path, interpolation_hr)

    def _calculate_metrics_per_lr_inter(self, scene, band, mask_mode, masks_path, interpolation_hr):
        mask = self._get_mask(mask_mode, masks_path, scene.id)
        for interpolation_lr in self._config["interpolations"]["LR"] + list(self._config["reconstructions"].keys()
                                                                            if self._config["reconstructions"] else []):
            self._calculate_metrics_per_scene(scene, band, mask, interpolation_hr, interpolation_lr)
        if self._config["save_images"]["masks"]:
            mask.build_score_mask(self._config["results_save_path"], scene.id, band)

    def _calculate_metrics_per_scene(self, scene, band, mask, interpolation_hr, interpolation_lr):
        if interpolation_lr in self._config["interpolations"]["LR"]:
            wv_img, _, reconstructed_image = scene.get_resized_band_images(band, self._config["resize_factor"],
                                                                           INTERPOLATION_METHODS[interpolation_lr],
                                                                           INTERPOLATION_METHODS[interpolation_hr])
            if self._config["save_images"]["masks"]:
                if interpolation_lr in self._config["build_score_masks"]["interpolations"]:
                    mask.interpolations.append((interpolation_lr, reconstructed_image))
        else:
            wv_img, reconstructed_image = scene.get_sr_wv_images(
                band, self._config["dataset"]["reconstructed_images_path"],
                self._config["reconstructions"][interpolation_lr], INTERPOLATION_METHODS[interpolation_hr])
            if self._config["save_images"]["masks"]:
                if interpolation_lr in self._config["build_score_masks"]["reconstructions"]:
                    mask.reconstructions.append((interpolation_lr, reconstructed_image))
        mask.hr_img = wv_img
        if self._config["save_images"]["images"]:
            self._save_images(wv_img, reconstructed_image, interpolation_lr, scene.id)
        try:
            image_metrics_values = {**{"scene_filename": scene.id}, **{key: self._metrics.metrics[key](
                reconstructed_image.astype(np.uint8),
                wv_img.astype(np.uint8),
                mask=mask.mask) for key in self._config["metrics"]}}
            self._results.add_element(image_metrics_values, band, mask.mask_mode, interpolation_hr, interpolation_lr,
                                      self._config["resize_factor"])
        except Exception as e:
            print(e)

    @staticmethod
    def _get_mask(mask_mode, masks_path, scene_id):
        mask = Mask(mask_mode, masks_path)
        if mask.mask_mode == "score":
            mask_path = [join(masks_path, path) for path in listdir(masks_path) if scene_id in path][0]
            mask.load_mask(mask_path)
        return mask

    def _save_images(self, wv_img, reconstructed_image, interpolation_name, scene_id):
        images_path = join(*[self._config["results_save_path"], "images", scene_id])
        create_directory(images_path)
        cv2.imwrite(join(images_path, f"hr.jpg"), wv_img)
        cv2.imwrite(join(images_path, f"{interpolation_name}.jpg"), reconstructed_image)

#
# a = Evaluation("config.yaml")
# a.calculate_metrics(pairs_path="C:\\Users\\pkowaleczko\\PycharmProjects\\superdeep\\datasets\\matched_full")



