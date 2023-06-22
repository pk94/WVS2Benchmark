from os import listdir
from os.path import join, basename, exists
import bisect
import json
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
from helper_functions import get_all_files, create_directory, FeatureExtractor, z_score_norm

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
        self.names = {"wv": None, "sentinel": []}

    @property
    def id(self):
        return self._id

    def _load_wv_multispectral_img(self):
        band_images = []
        for ii in range(8):
            band_img = cv2.imread(join(*[self._path, "hr_resized", f"mul_band_{ii}.tiff"]), cv2.IMREAD_GRAYSCALE)
            band_images.append(band_img.astype(float))
        return np.asarray(band_images)

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
        names = []
        for file in files:
            img = gdal.Open(file, gdalconst.GA_ReadOnly)
            img_array = img.ReadAsArray().astype(float)
            band_images.append(np.interp(img_array, (img_array.min(), img_array.max()), (0, 255)))
            names.append(basename(file).split(".")[0])
        return band_images, names

    def get_resized_band_images(self, band, resize_factor, interpolation_method_lr):
        hr_img = self._wv_multispectral_img[self._band_matches[band], :]
        sentinel_resized_band_images = self._get_resized_sentinel_band_imgs(resize_factor, interpolation_method_lr)
        sentinel_matched_band_images, sentinel_matched_mean_img = self._match_sentinel_wv_histograms(
            sentinel_resized_band_images, hr_img)
        return hr_img, sentinel_matched_band_images, sentinel_matched_mean_img

    def get_band_images(self, band):
        hr_img = self._wv_multispectral_img[self._band_matches[band], :]
        sentinel_matched_band_images, sentinel_matched_mean_img = self._match_sentinel_wv_histograms(
            self.sentinel_images, hr_img)
        return hr_img, sentinel_matched_band_images, sentinel_matched_mean_img

    def get_sr_wv_images(self, band, reconstructed_images_path, reconstruction_method):
        sr_images_paths = join(*[reconstructed_images_path, band, reconstruction_method])
        sr_image_path = [join(sr_images_paths, img_path) for img_path in listdir(sr_images_paths)
                         if self.id == img_path.split(".")[0]][0]
        sr_image = cv2.imread(sr_image_path, cv2.IMREAD_GRAYSCALE)
        sr_matched_image = match_histograms(sr_image, self._wv_multispectral_img[self._band_matches[band], :])
        return self._wv_multispectral_img[self._band_matches[band], :], sr_matched_image


class Mask:
    def __init__(self, mask_mode=None):
        self._mask_mode = mask_mode
        self._mask = None
        self._interpolations = []
        self._reconstructions = []
        self.hr_img = None
        self.lr_img = None
        self.lr_batch_img = []
        self.thresholds = {"ranges": [0, 20, 40, 50, 90, 100, 150, 200, 255],
                           "threshold_vals": [10, 15, 17, 35, 50, 55, 65, 75]}

    def __add__(self, other):
        if self.mask_mode is not None and other.mask_mode is not None:
            new_mask_mode = self.mask_mode + f"+{other.mask_mode}"
        else:
            new_mask_mode = None
        sum_mask = Mask(new_mask_mode)
        sum_mask._mask = np.bitwise_or(self._mask, other.mask)
        return sum_mask

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

    def save_mask(self, save_path):
        cv2.imwrite(save_path, self.mask * 255)

    def update(self, scene, band):
        hr_img, lr_batch_img, lr_img = scene.get_band_images(band)
        self.hr_img = hr_img
        self.lr_img = lr_img
        self.lr_batch_img = lr_batch_img

    def build_relevance_mask(self, save_path, scene_id, band, interpolations, reconstructions):
        create_directory(join(save_path, f"relevance_masks/relevance_masks_{band}/pairs"))
        final_mask = np.zeros_like(self.hr_img).astype(np.bool_)
        for interpolation_name, interpolation in self._interpolations:
            if interpolation_name in interpolations:
                interpolation = interpolation.astype(np.float64)
                interpolation_diff = np.abs(interpolation - self.hr_img)
                for reconstruction_name, reconstruction in self._reconstructions:
                    if reconstruction_name in reconstructions:
                        reconstruction = reconstruction.astype(np.float64)
                        reconstruction_diff = np.abs(reconstruction - self.hr_img)
                        relevance_mask = reconstruction_diff < interpolation_diff
                        final_mask = np.bitwise_or(final_mask, relevance_mask)
                        cv2.imwrite(join(*[save_path, f"relevance_masks/relevance_masks_{band}/pairs",
                                           f"{interpolation_name}_{reconstruction_name}_{scene_id}.png"]),
                                    np.invert(relevance_mask) * 255)
        cv2.imwrite(join(*[save_path, f"relevance_masks/relevance_masks_{band}/", f"mask_{scene_id}.png"]),
                    np.invert(final_mask) * 255)

    def mask_out(self, downsampled_hr_img, downsampled_lr_image):
        max_img = np.amax(np.stack([downsampled_hr_img, downsampled_lr_image]), axis=0)
        bs = lambda num: self.thresholds["threshold_vals"][bisect.bisect_left(self.thresholds["ranges"], num) - 1]
        vbs = np.vectorize(bs)
        thresholds = vbs(max_img)
        mask = (abs(downsampled_hr_img - downsampled_lr_image) > thresholds).astype(np.uint8) * 255
        upsampled_mask = cv2.resize(mask, (self.hr_img.shape[1], self.hr_img.shape[0]), interpolation=cv2.INTER_AREA)
        kernel = np.ones((4, 4), np.uint8)
        upsampled_mask = cv2.dilate(cv2.erode(upsampled_mask, kernel, iterations=5), kernel, iterations=10)
        upsampled_mask[upsampled_mask > 0] = 255
        return upsampled_mask

    def _perceptual_mask(self, band, feature_extractor, downsampled_hr_img, downsampled_lr_image, stats, threshold=50,
                         calc_stats=False):
        hr_features = feature_extractor(downsampled_hr_img)
        lr_features = feature_extractor(downsampled_lr_image)
        diffs = []
        for layer_num, (hr_feature, lr_feature) in enumerate(zip(hr_features[:3], lr_features[:3])):
            for feature_num, (hr_channel, lr_channel) in enumerate(zip(hr_feature, lr_feature)):
                hr_channel = feature_extractor.prepare_output(hr_channel)
                lr_channel = feature_extractor.prepare_output(lr_channel)
                if calc_stats:
                    feature_extractor.features_distributions_hr[band][layer_num][feature_num].update(hr_channel)
                    feature_extractor.features_distributions_lr[band][layer_num][feature_num].update(lr_channel)
                else:
                    hr_channel = z_score_norm(hr_channel, stats["hr"][band][str(layer_num)][str(feature_num)]["mean"],
                                              stats["hr"][band][str(layer_num)][str(feature_num)]["std"])
                    lr_channel = z_score_norm(lr_channel, stats["lr"][band][str(layer_num)][str(feature_num)]["mean"],
                                              stats["lr"][band][str(layer_num)][str(feature_num)]["std"])
                diff = abs(hr_channel - lr_channel)
                diffs.append(cv2.resize(diff, (self.hr_img.shape[1], self.hr_img.shape[0]),
                                        interpolation=cv2.INTER_AREA))
        diffs = np.sqrt(np.sum(np.power(np.stack(diffs), 2), axis=0))
        mask = (diffs > threshold).astype(np.uint8) * 255
        return mask

    def build_difference_mask(self, save_path, scene_id, band, downsample_factor=4):
        create_directory(join(save_path, f"difference_masks/difference_masks_{band}"))
        new_shape = (int(self.lr_img.shape[1] / downsample_factor), int(self.lr_img.shape[0] / downsample_factor))
        downsampled_lr_image = cv2.resize(self.lr_img, new_shape)
        downsampled_hr_img = cv2.resize(self.hr_img, new_shape)
        mask = self.mask_out(downsampled_hr_img, downsampled_lr_image)
        cv2.imwrite(join(*[save_path, f"difference_masks/difference_masks_{band}/", f"mask_{scene_id}.png"]), mask)

    def build_difference_mask_newest(self, save_path, scene_id, band, sentinel_names, downsample_factor=4):
        create_directory(join(save_path, f"difference_masks_newest/difference_masks_newest_{band}"))
        create_directory(join(save_path, f"difference_masks_newest/images_newest_{band}"))
        new_shape = (int(self.lr_img.shape[1] / downsample_factor), int(self.lr_img.shape[0] / downsample_factor))
        sentinel_names = [int(name.split("_")[-1].split("T")[0]) for name in sentinel_names]
        downsampled_lr_image = cv2.resize(self.lr_batch_img[np.argmax(sentinel_names)], new_shape)
        downsampled_hr_img = cv2.resize(self.hr_img, new_shape)
        cv2.imwrite(join(*[save_path, f"difference_masks_newest/images_newest_{band}/", f"lr_img_{scene_id}.png"]),
            downsampled_lr_image)
        cv2.imwrite(join(*[save_path, f"difference_masks_newest/images_newest_{band}/", f"hr_img_{scene_id}.png"]),
                    downsampled_hr_img)
        mask = self.mask_out(downsampled_hr_img, downsampled_lr_image)
        cv2.imwrite(join(*[save_path,
                           f"difference_masks_newest/difference_masks_newest_{band}/", f"mask_{scene_id}.png"]), mask)

    def build_perceptual_mask(self, feature_extractor, save_path, scene_id, band, stats):
        create_directory(join(save_path, f"perceptual_masks/perceptual_masks_{band}"))
        mask = self._perceptual_mask(band, feature_extractor, cv2.resize(
            self.hr_img, (int(self.lr_img.shape[1]), int(self.lr_img.shape[0]))), self.lr_img, stats)
        cv2.imwrite(join(*[save_path, f"perceptual_masks/perceptual_masks_{band}/", f"mask_{scene_id}.png"]), mask)

    def _get_variance_mask(self, img_1, img_2):
        raise NotImplementedError()


class SingleEvaluationResults:
    def __init__(self, band, mask_mode, reconstruction_method, resize_factor):
        self._band = band
        self._mask_mode = mask_mode
        self._reconstruction_method = reconstruction_method
        self._resize_factor = resize_factor
        self._results = []

    @property
    def results(self):
        return self._results

    @property
    def reconstruction_method(self):
        return self._reconstruction_method

    def filter(self, band, mask_mode, reconstruction_method, resize_factor):
        return band == self._band and mask_mode == self._mask_mode and \
               reconstruction_method == self._reconstruction_method and resize_factor == self._resize_factor

    def save(self, save_path):
        df = pd.DataFrame(self._results)
        df.to_csv(join(
            *[save_path,
              "per_scene",
              f"{self._band}_{self._reconstruction_method}_{self._mask_mode}_"
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
                for mask_mode, _ in self._config["dataset"]["masks_paths"].items():
                    results_database.append(SingleEvaluationResults(
                        band, mask_mode, reconstruction_method, self._config["resize_factor"]))
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
        if "Bicubic" in reconstruction_methods:
            bicubic_row = mean_results.loc[mean_results['Interpolation_method'] == "Bicubic"]
            balanced_metric_values = []
            for reconstruction_method in reconstruction_methods:
                rec_row = mean_results.loc[mean_results['Interpolation_method'] == reconstruction_method]
                balanced_metric = 0
                for metric in mean_results.columns[1:]:
                    bicubic_metric = bicubic_row[metric].values[0]
                    reconstruction_metric = rec_row[metric].values[0]
                    if metric == "cPSNR" or metric == "cSSIM":
                        balanced_metric += bicubic_metric / reconstruction_metric
                    elif metric == "LPIPS":
                        balanced_metric += reconstruction_metric / bicubic_metric
                balanced_metric /= len(mean_results.columns[1:])
                balanced_metric_values.append(balanced_metric)
            mean_results.insert(len(mean_results.columns), "Balanced_metric", balanced_metric_values)
        return mean_results

    def add_element(self, element, band, mask_mode, reconstruction_method, resize_factor):
        idx = 0
        for ii, single_ev_result in enumerate(self._results_database):
            idx = ii
            if single_ev_result.filter(band, mask_mode, reconstruction_method, resize_factor):
                break
        self._results_database[idx].results.append(element)

    def save_evaluation_results(self):
        for single_ev_result in self._results_database:
            single_ev_result.save(self._config["results_save_path"])
        reconstruction_methods = self._config["interpolations"]["LR"] + list(self._config["reconstructions"].keys()
                                                                             if self._config["reconstructions"] else [])
        mean_results_dict = {mask: {band: None for band in self._config["dataset"]["bands"]}
                             for mask in list(self._config["dataset"]["masks_paths"].keys())}
        for band in self._config["dataset"]["bands"]:
            for mask_mode, _ in self._config["dataset"]["masks_paths"].items():
                results = []
                for result in self._results_database:
                    for reconstruction_method in reconstruction_methods:
                        if result.filter(band, mask_mode, reconstruction_method,
                                         self._config["resize_factor"]):
                            results.append(result)
                mean_results = self._get_mean_evaluation_results(results)
                mean_results_dict[mask_mode][band] = mean_results
                mean_results.to_csv(
                    join(*[self._config["results_save_path"], "mean",
                           f"{band}_{mask_mode}_{self._config['resize_factor']}.csv"]),
                    index=False)
        self.save_summarized_results(mean_results_dict)

    def save_summarized_results(self, mean_results_dict):
        for mask in mean_results_dict.keys():
            mask_results = {}
            for band in mean_results_dict[mask].keys():
                df = mean_results_dict[mask][band]
                for column in df.columns[1:]:
                    mask_results.update({f"{column}_{band}": mean_results_dict[mask][band][column].values})
            mask_results = pd.DataFrame(mask_results)
            mask_results.insert(0, "Interpolation_method", mean_results_dict[mask][band]["Interpolation_method"].values)
            mask_results.to_csv(join(*[self._config["results_save_path"], f"results_{mask}.csv"]), index=False)


class Evaluation:
    def __init__(self, config_path):
        self._config, self._stats = self.load_config(config_path=config_path)
        self._results = EvaluationResults(self._config)
        self._metrics = MetricsSingleBand()
        self._create_results_save_path()
        self.feature_extractor = FeatureExtractor(self._config)

    @staticmethod
    def load_config(config_path):
        with open(config_path) as config_file:
            config_data = yaml.full_load(config_file)
        config_file.close()
        if exists(config_data["dataset"]["stats_path"]):
            with open(config_data["dataset"]["stats_path"]) as json_file:
                stats = json.load(json_file)
        else:
            stats = {}
        return config_data, stats

    def _create_results_save_path(self):
        create_directory(self._config["results_save_path"])
        create_directory(join(self._config["results_save_path"], "per_scene"))
        create_directory(join(self._config["results_save_path"], "mean"))

    def calculate_metrics(self, pairs_path):
        scenes_paths = [join(pairs_path, file) for file in listdir(pairs_path)]
        for scene_path in tqdm(scenes_paths[:1]):
            scene = Scene(scene_path)
            self._calculate_metrics_per_band(scene)
        if not exists(f"{self._config['results_save_path']}/stats.json"):
            self.feature_extractor.save_stats()
        if not self._config["save_images"]["masks"]:
            self._results.save_evaluation_results()

    def _calculate_metrics_per_band(self, scene):
        for band in self._config["dataset"]["bands"]:
            scene.sentinel_images, scene.names["sentinel"] = scene.load_sentinel_band_images(band)
            self._calculate_metrics_per_mask(scene, band)

    def _calculate_metrics_per_mask(self, scene, band):
        for mask_mode, masks_path in self._config["dataset"]["masks_paths"].items():
            masks_path += f"_{band}"
            self._calculate_metrics_per_lr_inter(scene, band, mask_mode, masks_path)

    def _calculate_metrics_per_lr_inter(self, scene, band, mask_mode, masks_path):
        mask = self._get_mask(mask_mode, masks_path, scene.id)
        for interpolation_lr in self._config["interpolations"]["LR"] + list(self._config["reconstructions"].keys()
                                                                            if self._config["reconstructions"] else []):
            self._calculate_metrics_per_scene(scene, band, mask, interpolation_lr)
        if self._config["save_images"]["masks"] and mask_mode == list(self._config["dataset"]["masks_paths"].keys())[0]:
            mask.update(scene, band)
            mask.build_relevance_mask(
                self._config["results_save_path"], scene.id, band,
                self._config["build_relevance_masks"]["interpolations"],
                self._config["build_relevance_masks"]["reconstructions"])
            mask.build_difference_mask(self._config["results_save_path"], scene.id, band)
            if band in list(self._stats["hr"].keys()):
                mask.build_perceptual_mask(
                    self.feature_extractor, self._config["results_save_path"], scene.id, band, self._stats)
            mask.build_difference_mask_newest(
                self._config["results_save_path"], scene.id, band, scene.names["sentinel"])

    def _calculate_metrics_per_scene(self, scene, band, mask, interpolation_lr):
        if interpolation_lr in self._config["interpolations"]["LR"]:
            wv_img, _, reconstructed_image = scene.get_resized_band_images(band, self._config["resize_factor"],
                                                                           INTERPOLATION_METHODS[interpolation_lr])
            if self._config["save_images"]["masks"]:
                mask.interpolations.append((interpolation_lr, reconstructed_image))
        else:
            wv_img, reconstructed_image = scene.get_sr_wv_images(
                band, self._config["dataset"]["reconstructed_images_path"],
                self._config["reconstructions"][interpolation_lr])
            if self._config["save_images"]["masks"]:
                mask.reconstructions.append((interpolation_lr, reconstructed_image))
        if self._config["save_images"]["images"]:
            self._save_images(wv_img, reconstructed_image, interpolation_lr, scene.id)
        if not self._config["save_images"]["masks"]:
            try:
                image_metrics_values = {**{"scene_filename": scene.id}, **{key: self._metrics.metrics[key](
                    reconstructed_image.astype(np.uint8),
                    wv_img.astype(np.uint8),
                    mask=mask.mask) for key in self._config["metrics"]}}
                self._results.add_element(image_metrics_values, band, mask.mask_mode, interpolation_lr,
                                          self._config["resize_factor"])
            except Exception as e:
                print(e)

    @staticmethod
    def _get_mask(mask_mode, masks_path, scene_id):
        mask = Mask(mask_mode)
        if mask.mask_mode != "none":
            mask_path = [join(masks_path, path) for path in listdir(masks_path) if scene_id in path][0]
            mask.load_mask(mask_path)
        return mask

    def _save_images(self, wv_img, reconstructed_image, interpolation_name, scene_id):
        images_path = join(*[self._config["results_save_path"], "images", scene_id])
        create_directory(images_path)
        cv2.imwrite(join(images_path, f"hr.jpg"), wv_img)
        cv2.imwrite(join(images_path, f"{interpolation_name}.jpg"), reconstructed_image)




