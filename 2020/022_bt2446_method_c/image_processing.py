# -*- coding: utf-8 -*-
"""
Image Processing for 'parameter adjustment tool'
================================================

"""

# import standard libraries
import os

# import third-party libraries
import cv2
import numpy as np
from colour import write_LUT, LUT3D

# import my libraries
from key_names import KeyNames
import bt2446_method_c as bmc
import bt2047_gamut_mapping as bgm
import transfer_functions as tf
import test_pattern_generator2 as tpg
import colormap as cmap
import color_space as cs

# information
__author__ = 'Toru Yoshihara'
__copyright__ = 'Copyright (C) 2020 - Toru Yoshihara'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Toru Yoshihara'
__email__ = 'toru.ver.11 at-sign gmail.com'

__all__ = []

kns = KeyNames()


TP_IMAGE_PATH = "./img/step_ramp_step_65.png"
LOW_IMAGE_PATH = "./img/dark.png"
MID_IMAGE_PATH = "./img/middle.png"
HIGH_IMAGE_PATH = "./img/high.png"


class ImageProcessing():
    """
    event controller for parameter_adjustment_tool.
    """
    def __init__(self, width=720, peak_luminance=1000):
        self.tp_img_path = TP_IMAGE_PATH
        self.low_image_path = LOW_IMAGE_PATH
        self.mid_image_path = MID_IMAGE_PATH
        self.high_image_path = HIGH_IMAGE_PATH
        self.peak_luminance = peak_luminance
        self.width = width

    def read_img(self, path):
        img = cv2.imread(
            path, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_COLOR)[..., ::-1]
        max_value = np.iinfo(img.dtype).max
        img = img / max_value

        return img

    def read_and_concat_img(self):
        img_tp = self.resize_to_fixed_width(
            self.read_img(self.tp_img_path))
        img_low = self.resize_to_fixed_width(
            self.read_img(self.low_image_path))
        img_mid = self.resize_to_fixed_width(
            self.read_img(self.mid_image_path))
        img_high = self.resize_to_fixed_width(
            self.read_img(self.high_image_path))
        self.raw_img = np.vstack([img_tp, img_low, img_mid, img_high])
        self.raw_img_linear = tf.eotf(self.raw_img, tf.ST2084)

        # keep coordinate information
        tp_height = img_tp.shape[0]
        low_height = img_low.shape[0]
        mid_height = img_mid.shape[0]
        high_height = img_high.shape[0]
        self.img_v_pos_info = [
            dict(st=0, ed=tp_height),
            dict(st=tp_height, ed=tp_height+low_height),
            dict(st=tp_height+low_height, ed=tp_height+low_height+mid_height),
            dict(st=tp_height+low_height+mid_height,
                 ed=tp_height+low_height+mid_height+high_height)
        ]

    def extract_image(self, img, img_idx=0):
        st = self.img_v_pos_info[img_idx]['st']
        ed = self.img_v_pos_info[img_idx]['ed']
        devided_img = img[st:ed]

        return devided_img

    def non_linear_to_luminance(self, img):
        img_luminance = tf.eotf_to_luminance(img, tf.ST2084)
        return img_luminance

    def get_concat_raw_image(self):
        return self.raw_img

    def conv_8bit_int(self, img):
        return np.uint8(np.round(img * 0xFF))

    def conv_16bit_int(self, img):
        return np.uint16(np.round(img * 0xFFFF))

    def conv_io_stream(self, img):
        is_success, buffer = cv2.imencode(".png", img[..., ::-1])
        return buffer.tobytes()

    def resize_to_fixed_width(self, img):
        src_width = img.shape[1]
        src_height = img.shape[0]
        dst_width = self.width
        dst_height = int(self.width / src_width * src_height + 0.5)

        dst_img = cv2.resize(
            img, (dst_width, dst_height), interpolation=cv2.INTER_AREA)

        return dst_img

    def make_colormap_image(self, turbo_peak_luminance=1000):
        self.colormap_img = cmap.apply_st2084_to_srgb_colormap(
            self.raw_img, sdr_pq_peak_luminance=100,
            turbo_peak_luminance=turbo_peak_luminance)

    def get_colormap_image(self):
        return self.colormap_img

    def make_sdr_image(
            self, src_color_space_name=cs.BT2020, tfc=tf.ST2084,
            alpha=0.15, sigma=0.5,
            hdr_ref_luminance=203, hdr_peak_luminance=1000,
            k1=0.8, k3=0.7, y_sdr_ip=60, bt2407_gamut_mapping=True):
        sdr_img_linear = bmc.bt2446_method_c_tonemapping(
             img=self.raw_img_linear,
             src_color_space_name=src_color_space_name,
             tfc=tfc, alpha=alpha, sigma=sigma,
             hdr_ref_luminance=hdr_ref_luminance,
             hdr_peak_luminance=hdr_peak_luminance,
             k1=k1, k3=k3, y_sdr_ip=y_sdr_ip)

        if bt2407_gamut_mapping:
            sdr_img_linear = bgm.bt2407_gamut_mapping_for_rgb_linear(
                rgb_linear=sdr_img_linear,
                outer_color_space_name=cs.BT2020,
                inner_color_space_name=cs.BT709)
        sdr_img_linear = sdr_img_linear ** (1/2.4)

        return sdr_img_linear


def make_3dlut(
        src_color_space_name=cs.BT2020, tfc=tf.ST2084,
        alpha=0.15, sigma=0.5, gamma=2.4,
        hdr_ref_luminance=203, hdr_peak_luminance=1000,
        k1=0.8, k3=0.7, y_sdr_ip=60, bt2407_gamut_mapping=True,
        grid_num=65, prefix=""):
    x = LUT3D.linear_table(grid_num).reshape((1, grid_num ** 3, 3))
    print(x.shape)
    x_linear = tf.eotf(x, tf.ST2084)
    sdr_img_linear = bmc.bt2446_method_c_tonemapping(
         img=x_linear,
         src_color_space_name=src_color_space_name,
         tfc=tfc, alpha=alpha, sigma=sigma,
         hdr_ref_luminance=hdr_ref_luminance,
         hdr_peak_luminance=hdr_peak_luminance,
         k1=k1, k3=k3, y_sdr_ip=y_sdr_ip)
    if bt2407_gamut_mapping:
        sdr_img_linear = bgm.bt2407_gamut_mapping_for_rgb_linear(
            rgb_linear=sdr_img_linear,
            outer_color_space_name=cs.BT2020,
            inner_color_space_name=cs.BT709)
    sdr_img_nonlinear = sdr_img_linear ** (1/gamma)
    sdr_img_nonlinear = sdr_img_nonlinear.reshape(
        ((grid_num, grid_num, grid_num, 3)))
    print(sdr_img_nonlinear.shape)

    lut_name = "ty tone mapping"
    lut3d = LUT3D(table=sdr_img_nonlinear, name=lut_name)

    file_name = f"./3DLUT/{prefix}_a_{alpha:.2f}_s_{sigma:.2f}_k1_{k1:.2f}_"\
        + f"k3_{k3:.2f}_y_s_{y_sdr_ip}_grid_{grid_num}_gamma_{gamma:.1f}.cube"
    write_LUT(lut3d, file_name)


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    # im_pro = ImageProcessing()
    # im_pro.read_and_concat_img()
    # im_pro.apply_colormap(im_pro.raw_img, 4000)
    make_3dlut(
        src_color_space_name=cs.BT2020, tfc=tf.ST2084,
        alpha=0.10, sigma=0.6, gamma=2.4,
        hdr_ref_luminance=203, hdr_peak_luminance=1000,
        k1=0.69, k3=0.74, y_sdr_ip=49.0, bt2407_gamut_mapping=True,
        grid_num=65, prefix="1000nits_v3")
    make_3dlut(
        src_color_space_name=cs.BT2020, tfc=tf.ST2084,
        alpha=0.10, sigma=0.6, gamma=2.4,
        hdr_ref_luminance=203, hdr_peak_luminance=1000,
        k1=0.69, k3=0.74, y_sdr_ip=41.0, bt2407_gamut_mapping=True,
        grid_num=65, prefix="4000nits_v3")
