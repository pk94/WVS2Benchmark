import shutil
from os import listdir, remove
from os.path import basename, join, exists
import cv2
import numpy as np
from tqdm import tqdm
from pathlib import Path
from skimage.exposure import match_histograms
from osgeo import gdal
from shapely.geometry import Polygon
from sentinel import SentinelScene, get_color_sentinel_img
from worldview import WorldViewScene, get_color_worldview_img
from helper_functions import get_geo_coordinates, create_directory


class Clipper:
    """
    Clipper class. It clips the Sentinel low resolution scenes images to the common area with WorldView high resolution
    tiles images.

    Attributes
    ----------
    sentinel_scene_path : str
        Path to the Sentinel scene.
    scene_name : str
        Sentinel scene name, for example 0_31UFT.
    sentinel_scene : SentinelScene
        SentinelScene object on which the clipping will be performed.
    worldview_scenes_mul : list
        List of WorldViewScene objets of multispectral type on which the clipping will be performed.
    worldview_scenes_pan : list
        List of WorldViewScene objets of panchromatic type on which the clipping will be performed.


    Methods
    -------
    __init__()
        Initializes Clipper object.
    get_worldview_scenes(scene_type)
        Returns the list of WorldViewScene objets of specific type on which the clipping will be performed. The
        WorldView scenes are contained in the Sentinel scene path.
    make_rgb_pairs(pairs_save_path)
        Generates RGB images which are the common areas of Sentinel images and WorldView tiles images. Those images are
        presented in three ways - as a pairs side by side, as a checkerboard where images crops are taken from the
        Sentinel and WorldView images alternately and as a blended image.
    clip_full_scene(pairs_save_path)
        Clips Sentinel and WorldView images (all bands for Sentinel and multispectral and panchromatic for WorldView) to
        the common area and saves them using the predefined nomenclature.

    """

    def __init__(self, sentinel_scene_path):
        """Initializes Clipper object.

        Parameters
        ----------
        sentinel_scene_path : str
            Path to the Sentinel scene.
        """
        self.sentinel_scene_path = sentinel_scene_path
        self.scene_name = basename(sentinel_scene_path)
        self.sentinel_scene = SentinelScene(sentinel_scene_path)
        self.worldview_scenes_mul = self.get_worldview_scenes("MUL")
        self.worldview_scenes_pan = self.get_worldview_scenes("PAN")

    def get_worldview_scenes(self, scene_type):
        """Returns the list of WorldViewScene objets of specific type on which the clipping will be performed. The
        WorldView scenes are contained in the Sentinel scene path.

        Parameters
        ----------
        scene_type : str
            WorldView scene type. "MUL" for multispectral or "PAN" for panchromatic.

        Returns
        -------
        world_view_scenes : list
            List of WorldViewScene objets of chosen type on which the clipping will be performed.

        """
        world_view_scenes = []
        scenes_path = join(*[self.sentinel_scene_path, "WorldView", f"{scene_type}"])
        for directory in listdir(scenes_path):
            new_scene = WorldViewScene(join(scenes_path, directory))
            new_scene.images_paths = [img_path for img_path in new_scene.images_paths if img_path.endswith("TIF")]
            world_view_scenes.append(new_scene)
        return world_view_scenes

    def make_rgb_pairs(self, pairs_save_path):
        """Generates RGB images which are the common areas of Sentinel images and WorldView tiles images. Those images
        are presented in three ways - as a pairs side by side, as a checkerboard where images crops are taken from the
        Sentinel and WorldView images alternately and as a blended image. During the procedure the cache files are
        produced - before run make sure that there are no cache files from previous run.

        Parameters
        ----------
        pairs_save_path : str
            Path where the paired images will be saved.

        """
        # TODO implement clouds masks.
        quality_check_path = join(pairs_save_path, self.scene_name)
        cache_filenames = ["b_cache.jp2", "g_cache.jp2", "r_cache.jp2", "scl.jp2"]
        dirs_to_create = [quality_check_path, join(quality_check_path, "pairs"), join(quality_check_path, "blended"),
                          join(quality_check_path, "tiled"), join(quality_check_path, "pairs_original")]
        for directory in dirs_to_create:
            create_directory(directory)
        for worldview_scene in self.worldview_scenes_mul:
            for wv_image_path in tqdm(worldview_scene.images_paths):
                for b, g, r, scl in zip(self.sentinel_scene.images_paths["b2"], self.sentinel_scene.images_paths["b3"],
                                        self.sentinel_scene.images_paths["b4"],
                                        self.sentinel_scene.images_paths["SCL"]):
                    pair_name = basename(wv_image_path)[:-4] + "_" + basename(b)[:-4]
                    channels = [b, g, r, scl]
                    channels_are_clipped = [clip_sentinel_to_wv(channel, None, wv_image_path, cache_filename, ".")
                                            for channel, cache_filename in zip(channels, cache_filenames)]
                    if all(channels_are_clipped):
                        sentinel_color = get_color_sentinel_img("b_cache.jp2", "g_cache.jp2", "r_cache.jp2")
                        wv_color_original = get_color_worldview_img("wv_mul.TIF")
                        wv_color = cv2.resize(wv_color_original, (sentinel_color.shape[1], sentinel_color.shape[0]))
                        sentinel_color = match_histograms(sentinel_color, wv_color, channel_axis=-1).astype(np.uint8)
                        tiled_image = make_tiled_image(sentinel_color, wv_color, 5, 5)
                        blended_images = cv2.addWeighted(wv_color, 0.5, sentinel_color, 0.5, 0)
                        cv2.imwrite(join(*[quality_check_path, "tiled", f"{pair_name}_tiled.jpg"]), tiled_image)
                        cv2.imwrite(join(*[quality_check_path, "blended",
                                                   f"{pair_name}_blended.jpg"]), blended_images)
                        cv2.imwrite(join(*[quality_check_path, "pairs", f"{pair_name}.jpg"]),
                                    np.concatenate([sentinel_color, wv_color], axis=1))
                        create_directory(join(*[quality_check_path, "pairs_original", pair_name]))
                        cv2.imwrite(join(*[quality_check_path, "pairs_original", pair_name,
                                                   f"{pair_name}_sentinel.tif"]),
                                    match_histograms(sentinel_color, wv_color_original,
                                                     channel_axis=-1).astype(np.uint8))
                        cv2.imwrite(join(*[quality_check_path, "pairs_original", pair_name,
                                                   f"{pair_name}_wv.tif"]), wv_color_original)
                    if exists("wv_mul.TIF"):
                        remove("wv_mul.TIF")
                    for cache_filename in cache_filenames:
                        if exists(cache_filename):
                            remove(cache_filename)

    def clip_full_scene(self, pairs_save_path):
        """Clips Sentinel and WorldView images (all bands for Sentinel and multipsectral and panchromatic for WorldView)
        to the common area and saves them using the predefined nomenclature. Panchromatic and multispectral WV scenes,
        that are exactly the same can have different names - differing by 1 second of acquisition time (for example
        14SEP05105015-P and 14SEP05105016-M). Because of that these are processed separately - first panchromatic scenes
        are processed and then the left multispectral (those that were not paired by the name with panchromatic scenes)
        scenes are processed.

        Parameters
        ----------
        pairs_save_path : str
            Path where the paired images will be saved.

        """
        create_directory("cache")
        num_sentinel_images = len(self.sentinel_scene.images_paths["b1"])
        bands = list(self.sentinel_scene.images_paths.keys())
        cache_filenames = [join("cache", band + ".jp2") for band in bands]
        processed_wv_mul = []

        def process_wv_scenes(wv_scenes):
            """Main clipping function for WorldView scene. For each WV scene tile it finds a common area with paired
            Sentinel scene. If area is not empty, each band of the Sentinel scene, as well as WV tile, is clipped to
            this area and saved as a cache file. Those cache files are than moved to the pair directory with predefined
            nomenclature.

            Parameters
            ----------
            wv_scenes: list
                List of WorldView scenes to be processed.

            """
            for worldview_scene in wv_scenes:
                is_pan = True if worldview_scene.type == "PAN" else False
                if is_pan:
                    pan_path = Path(worldview_scene.path)
                    wv_image_path_mul = Path(*pan_path.parts[:-2], "MUL", pan_path.name[:-1] + "M")
                    processed_wv_mul.append(wv_image_path_mul)
                for wv_image_path in tqdm(worldview_scene.images_paths):
                    wv_scene_name = basename(wv_image_path)
                    wv_id = wv_scene_name.split("-")[0][:-2] + "-" + wv_scene_name.split("-")[1][1:]
                    pair_id = self.scene_name + "_" + wv_id
                    if is_pan:
                        mul_path = get_wv_multispectral_image_path(wv_image_path, wv_image_path_mul)
                        mul_path = mul_path if exists(mul_path) else None
                    for ii in range(num_sentinel_images):
                        bands_images = [self.sentinel_scene.images_paths[band][ii] for band in bands]
                        sentinel_img_name = basename(self.sentinel_scene.images_paths["b1"][ii])
                        channels_are_clipped = [clip_sentinel_to_wv(channel, wv_image_path, mul_path,
                                                                    cache_filename, "cache")
                                                for channel, cache_filename in zip(bands_images, cache_filenames)] \
                            if is_pan else [clip_sentinel_to_wv(channel, None, wv_image_path,
                                                                cache_filename, "cache")
                                            for channel, cache_filename in zip(bands_images, cache_filenames)]
                        if all(channels_are_clipped):
                            pair_path = join(pairs_save_path, pair_id)
                            create_directory(pair_path)
                            base_dirs = [band for band in bands if not band.startswith("CLD")]
                            move_sentinel_files(pair_path, base_dirs, sentinel_img_name)

        process_wv_scenes(self.worldview_scenes_pan)
        not_processed_wv_mul = [wv_mul_scene for wv_mul_scene in self.worldview_scenes_mul if
                                wv_mul_scene.path not in processed_wv_mul]
        process_wv_scenes(not_processed_wv_mul)


def clip_sentinel_to_wv(sentinel_img_path, wv_image_path_pan, wv_image_path_mul, save_path,
                        wv_clipped_save_path="cache"):
    """Finds an intersection polygon of WV tile and Sentinel Scene band image and clips these images to this polygon.

    Parameters
    ----------
    sentinel_img_path : str
        Sentinel band image path.
    wv_image_path_pan : str
        WorldView panchromatic tile image path. None if not available.
    wv_image_path_mul : str
        WorldView multispectral tile image path. None if not available.
    save_path : str
        Clipped Sentinel image save path.
    wv_clipped_save_path : str
        Directory in which the clipped WV tiles will be saved.

    Returns
    -------
    was_clipped : bool
        True if intersection polygon is not empty, else False.

    """
    intersection = get_intersection_polygon(wv_image_path_pan, sentinel_img_path) if wv_image_path_pan \
        else get_intersection_polygon(wv_image_path_mul, sentinel_img_path)
    proj_win = get_intersection_projection_window(intersection)
    if proj_win:
        gdal.Translate(destName=save_path, srcDS=sentinel_img_path, projWin=proj_win)
        if wv_image_path_pan and not exists(join(wv_clipped_save_path, "wv_pan.TIF")):
            gdal.Translate(destName=join(wv_clipped_save_path, "wv_pan.TIF"), srcDS=wv_image_path_pan,
                           projWin=proj_win)
        if wv_image_path_mul and not exists(join(wv_clipped_save_path, "wv_mul.TIF")):
            gdal.Translate(destName=join(wv_clipped_save_path, "wv_mul.TIF"), srcDS=wv_image_path_mul,
                           projWin=proj_win)
    return True if proj_win else False


def get_intersection_polygon(wv_image_path, sentinel_image_path):
    """Finds geographical coordinates of an intersection polygon of WV tile and Sentinel Scene band image.

    Parameters
    ----------
    sentinel_image_path : str
        Sentinel band image path.
    wv_image_path : str
        WorldView tile image path.

    Returns
    -------
    intersection : shapely.geometry.polygon.Polygon
        Intersection polygon.

    """
    sentinel_coordinates = get_geo_coordinates(sentinel_image_path)
    wv_coordinates = get_geo_coordinates(wv_image_path)
    sentinel_polygon, wv_polygon = [], []
    for sentinel_point, wv_point in zip(sentinel_coordinates[0].values(), wv_coordinates[0].values()):
        sentinel_polygon.append((sentinel_point["easting"], sentinel_point["northing"]))
        wv_polygon.append((wv_point["easting"], wv_point["northing"]))
    sentinel_polygon = Polygon(sentinel_polygon)
    wv_polygon = Polygon(wv_polygon)
    intersection = sentinel_polygon.intersection(wv_polygon)
    return intersection


def get_intersection_projection_window(intersection_polygon):
    """Returns the list of geographical coordinates of the intersection polygon.

    Parameters
    ----------
    intersection_polygon : shapely.geometry.polygon.Polygon
        Intersection polygon of the WV tile image and sentinel band image.

    Returns
    -------
    proj_win : list
        List of geographical coordinates of intersection polygon.

    """
    xs = intersection_polygon.exterior.coords.xy[0]
    ys = intersection_polygon.exterior.coords.xy[1]
    proj_win = [min(xs), max(ys), max(xs), min(ys)] if xs and ys else None
    return proj_win


def make_tiled_image(sentinel_img, wv_img, num_tiles_width, num_tiles_height):
    """Produces tiled (checkerboard) image from matching RGB WorldView and Sentinel images. Number of tile is
    num_tiles_width * num_tiles_height. Tiles are placed alternately - one is from Sentinel image and the following is
    from WorldView image.

    Parameters
    ----------
    sentinel_img : numpy.array
        Sentinel RGB image.
    wv_img : numpy.array
        WorldView RGB image.
    num_tiles_width : int
        Number of tiles columns.
    num_tiles_height : int
        Number of tiles rows

    Returns
    -------
    tiled_image : numpy.array
        Image composed of tiles from Sentinel and WorldView images.

    """
    tile_width = sentinel_img.shape[1] / num_tiles_width
    tile_height = sentinel_img.shape[0] / num_tiles_height
    is_sentinel_tile = True
    rows = []
    for idx_row in range(num_tiles_height):
        tiles_rows = []
        start_row_idx = int(idx_row * tile_height)
        for idx_column in range(num_tiles_width):
            source_image = sentinel_img if is_sentinel_tile else wv_img
            start_column_idx = int(idx_column * tile_width)
            if idx_row + 1 != num_tiles_height and idx_column + 1 != num_tiles_width:
                tile = source_image[start_row_idx:int((idx_row + 1) * tile_height),
                       start_column_idx:int((idx_column + 1) * tile_width), :]
            elif idx_row + 1 == num_tiles_height and idx_column + 1 != num_tiles_width:
                tile = source_image[start_row_idx:, start_column_idx:int((idx_column + 1) * tile_width), :]
            elif idx_row + 1 != num_tiles_height and idx_column + 1 == num_tiles_width:
                tile = source_image[start_row_idx:int((idx_row + 1) * tile_height), start_column_idx:, :]
            elif idx_row + 1 == num_tiles_height and idx_column + 1 == num_tiles_width:
                tile = source_image[start_row_idx:, start_column_idx:, :]
            tiles_rows.append(tile)
            is_sentinel_tile = not is_sentinel_tile
        rows.append(np.concatenate(tiles_rows, axis=1))
    tiled_image = np.concatenate(rows, axis=0)
    return tiled_image


def get_wv_multispectral_image_path(wv_pan_img_path, mul_path):
    """Returns matching multispectral tile image path for a panchromatic image.

    Parameters
    ----------
    wv_pan_img_path : str
        Path to the panchromatic image
    mul_path : str
        Path to the WorldView multispectral scene.

    Returns
    -------
    wv_mul_img_path : str
        Path to the matching multispectral tile image.

    """
    pan_filename = basename(wv_pan_img_path).split("-")
    mul_filename = pan_filename[0] + f"-M{pan_filename[1][1:]}-" + pan_filename[2]
    return join(mul_path, mul_filename)


def move_sentinel_files(pair_path, bands, save_name):
    """Moves clipped cache files to the target directory.

    Parameters
    ----------
    pair_path : str
        Path to the target pair directory.
    bands : str
        List of Sentinel bands.
    save_name : str
        Sentinel base image filename.

    """
    for band in bands:
        band_path = join(pair_path, band)
        create_directory(band_path)
        lrs_path = join(band_path, "lrs")
        create_directory(lrs_path)
        shutil.move(join("cache", band + ".jp2"), join(lrs_path, save_name))
    if exists(join("cache", "wv_pan.TIF")):
        shutil.move(join("cache", "wv_pan.TIF"), join(pair_path, f"hr_pan.TIF"))
    if exists(join("cache", "wv_mul.TIF")):
        shutil.move(join("cache", "wv_mul.TIF"), join(pair_path, f"hr_mul.TIF"))
    create_directory(join(*[pair_path, "CLD", "20m"]))
    create_directory(join(*[pair_path, "CLD", "60m"]))
    shutil.move(join("cache", "CLD_20m.jp2"), join(*[pair_path, "CLD", "20m", save_name]))
    shutil.move(join("cache", "CLD_60m.jp2"), join(*[pair_path, "CLD", "60m", save_name]))
