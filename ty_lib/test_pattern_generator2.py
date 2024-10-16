#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
評価用のテストパターン作成ツール集

"""

import os
import subprocess
from colour.models.rgb.rgb_colourspace import RGB_to_RGB
from colour.utilities import tstack
import cv2
import numpy as np
from colour.colorimetry import CCS_ILLUMINANTS
from colour.models import xy_to_XYZ, XYZ_to_RGB, XYZ_to_xyY
from colour.models import xy_to_xyY, xyY_to_XYZ, Lab_to_XYZ, LCHab_to_Lab
from colour.models import RGB_COLOURSPACE_BT709, RGB_COLOURSPACE_BT2020,\
    RGB_COLOURSPACE_ACES2065_1, RGB_COLOURSPACE_ACESCG
from colour.algebra import normalise_maximum, vector_dot
from colour.adaptation import matrix_chromatic_adaptation_VonKries
from colour import RGB_COLOURSPACES, CCS_COLOURCHECKERS
import math
from jzazbz import jzczhz_to_jzazbz

import transfer_functions as tf
import create_gamut_booundary_lut as cgbl
import font_control as fc
import color_space as cs
from ty_utility import add_suffix_to_filename

CMFS_NAME = 'CIE 1931 2 Degree Standard Observer'
D65_WHITE = CCS_ILLUMINANTS[CMFS_NAME]['D65']
YCBCR_CHECK_MARKER = [0, 0, 0]

UNIVERSAL_COLOR_LIST = ["#F6AA00", "#FFF100", "#03AF7A",
                        "#005AFF", "#4DC4FF", "#804000"]
# for 8bit 10bit pattern
L_LOW_C_LOW = 'lightness_low_chroma_low'
L_LOW_C_MIDDLE = 'lightness_low_chroma_middle'
L_LOW_C_HIGH = 'lightness_low_chroma_high'
L_MIDDLE_C_LOW = 'lightness_middle_chroma_low'
L_MIDDLE_C_MIDDLE = 'lightness_middle_chroma_middle'
L_MIDDLE_C_HIGH = 'lightness_middle_chroma_high'
L_HIGH_C_LOW = 'lightness_high_chroma_low'
L_HIGH_C_MIDDLE = 'lightness_high_chroma_middle'
L_HIGH_C_HIGH = 'lightness_high_chroma_high'


def preview_image(img, order='rgb', over_disp=False):
    if order == 'rgb':
        cv2.imshow('preview', img[:, :, ::-1])
    elif order == 'bgr':
        cv2.imshow('preview', img)
    elif order == 'mono':
        cv2.imshow('preview', img)
    else:
        raise ValueError("order parameter is invalid")

    if over_disp:
        cv2.resizeWindow('preview', )
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def h_mono_line_to_img(line, height):
    """
    create image from horizontal line data.

    Parameters
    ----------
    line : ndarray
        line value
    height : int
        height

    Examples
    --------
    >>> line = np.linspace(0, 1, 9)
    >>> print(line)
    [ 0.     0.125  0.25   0.375  0.5    0.625  0.75   0.875  1.   ]

    >>> img = h_mono_line_to_img(line, 4)
    >>> print(img)
    [[[ 0.     0.     0.   ]
      [ 0.125  0.125  0.125]
      [ 0.25   0.25   0.25 ]
      [ 0.375  0.375  0.375]
      [ 0.5    0.5    0.5  ]
      [ 0.625  0.625  0.625]
      [ 0.75   0.75   0.75 ]
      [ 0.875  0.875  0.875]
      [ 1.     1.     1.   ]]

     [[ 0.     0.     0.   ]
      [ 0.125  0.125  0.125]
      [ 0.25   0.25   0.25 ]
      [ 0.375  0.375  0.375]
      [ 0.5    0.5    0.5  ]
      [ 0.625  0.625  0.625]
      [ 0.75   0.75   0.75 ]
      [ 0.875  0.875  0.875]
      [ 1.     1.     1.   ]]

     [[ 0.     0.     0.   ]
      [ 0.125  0.125  0.125]
      [ 0.25   0.25   0.25 ]
      [ 0.375  0.375  0.375]
      [ 0.5    0.5    0.5  ]
      [ 0.625  0.625  0.625]
      [ 0.75   0.75   0.75 ]
      [ 0.875  0.875  0.875]
      [ 1.     1.     1.   ]]

     [[ 0.     0.     0.   ]
      [ 0.125  0.125  0.125]
      [ 0.25   0.25   0.25 ]
      [ 0.375  0.375  0.375]
      [ 0.5    0.5    0.5  ]
      [ 0.625  0.625  0.625]
      [ 0.75   0.75   0.75 ]
      [ 0.875  0.875  0.875]
      [ 1.     1.     1.   ]]]
    """
    return line.reshape(1, -1, 1).repeat(3, axis=2).repeat(height, axis=0)


def v_mono_line_to_img(line, height):
    """
    create image from horizontal line data.

    Parameters
    ----------
    line : ndarray
        line value
    height : int
        height

    Examples
    --------
    >>> line = np.linspace(0, 1, 4)
    >>> print(line)
    [ 0.          0.33333333  0.66666667  1.        ]

    >>> img = v_mono_line_to_img(line, 6)
    >>> print(img)
    [[[ 0.          0.          0.        ]
      [ 0.          0.          0.        ]
      [ 0.          0.          0.        ]
      [ 0.          0.          0.        ]
      [ 0.          0.          0.        ]
      [ 0.          0.          0.        ]]

     [[ 0.33333333  0.33333333  0.33333333]
      [ 0.33333333  0.33333333  0.33333333]
      [ 0.33333333  0.33333333  0.33333333]
      [ 0.33333333  0.33333333  0.33333333]
      [ 0.33333333  0.33333333  0.33333333]
      [ 0.33333333  0.33333333  0.33333333]]

     [[ 0.66666667  0.66666667  0.66666667]
      [ 0.66666667  0.66666667  0.66666667]
      [ 0.66666667  0.66666667  0.66666667]
      [ 0.66666667  0.66666667  0.66666667]
      [ 0.66666667  0.66666667  0.66666667]
      [ 0.66666667  0.66666667  0.66666667]]

     [[ 1.          1.          1.        ]
      [ 1.          1.          1.        ]
      [ 1.          1.          1.        ]
      [ 1.          1.          1.        ]
      [ 1.          1.          1.        ]
      [ 1.          1.          1.        ]]]
    """
    return line.reshape(-1, 1, 1).repeat(3, axis=2).repeat(height, axis=1)


def h_color_line_to_img(line, height):
    """
    create image from horizontal line data.

    Parameters
    ----------
    line : ndarray
        color line value. shape is (N, M, 3) or (M, 3).
    height : int
        height

    Examples
    --------
    >>> line_r = np.linspace(0, 4, 5)
    >>> line_g = np.linspace(0, 4, 5) * 2
    >>> line_b = np.linspace(0, 4, 5) * 3
    >>> line_color = tstack([line_r, line_g, line_b])
    >>> print(line_color)
    [[  0.   0.   0.]
     [  1.   2.   3.]
     [  2.   4.   6.]
     [  3.   6.   9.]
     [  4.   8.  12.]]

    >>> img = h_color_line_to_img(line_color, 4)
    >>> print(img)
    [[[  0.   0.   0.]
      [  1.   2.   3.]
      [  2.   4.   6.]
      [  3.   6.   9.]
      [  4.   8.  12.]]

     [[  0.   0.   0.]
      [  1.   2.   3.]
      [  2.   4.   6.]
      [  3.   6.   9.]
      [  4.   8.  12.]]

     [[  0.   0.   0.]
      [  1.   2.   3.]
      [  2.   4.   6.]
      [  3.   6.   9.]
      [  4.   8.  12.]]

     [[  0.   0.   0.]
      [  1.   2.   3.]
      [  2.   4.   6.]
      [  3.   6.   9.]
      [  4.   8.  12.]]]
    """
    return line.reshape((1, -1, 3)).repeat(height, axis=0)


def v_color_line_to_img(line, height):
    """
    create image from horizontal line data.

    Parameters
    ----------
    line : ndarray
        color line value. shape is (N, M, 3) or (M, 3).
    height : int
        height

    Examples
    --------
    >>> line_r = np.linspace(0, 4, 5)
    >>> line_g = np.linspace(0, 4, 5) * 2
    >>> line_b = np.linspace(0, 4, 5) * 3
    >>> line_color = tstack([line_r, line_g, line_b])
    >>> print(line_color)
    [[  0.   0.   0.]
     [  1.   2.   3.]
     [  2.   4.   6.]
     [  3.   6.   9.]
     [  4.   8.  12.]]

    >>> img = v_color_line_to_img(line_color, 4)
    >>> print(img)
    [[[  0.   0.   0.]
      [  0.   0.   0.]
      [  0.   0.   0.]
      [  0.   0.   0.]]

     [[  1.   2.   3.]
      [  1.   2.   3.]
      [  1.   2.   3.]
      [  1.   2.   3.]]

     [[  2.   4.   6.]
      [  2.   4.   6.]
      [  2.   4.   6.]
      [  2.   4.   6.]]

     [[  3.   6.   9.]
      [  3.   6.   9.]
      [  3.   6.   9.]
      [  3.   6.   9.]]

     [[  4.   8.  12.]
      [  4.   8.  12.]
      [  4.   8.  12.]
      [  4.   8.  12.]]]
    """
    return line.reshape((-1, 1, 3)).repeat(height, axis=1)


def img_read(filename):
    """
    OpenCV の BGR 配列が怖いので並べ替えるwrapperを用意。
    """
    img = cv2.imread(filename, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_COLOR)

    if img is not None:
        if img.shape[2] == 3:
            return img[:, :, ::-1]
        elif img.shape[2] == 4:
            shape = img.shape
            b, g, r, a = np.dsplit(img, 4)
            return np.dstack((r, g, b, a)).reshape((shape))
        else:
            raise ValueError("not supported img shape for immg_write")
    else:
        return img


def img_read_as_float(filename):
    img_int = img_read(filename)
    img_max_value = np.iinfo(img_int.dtype).max
    img_float = img_int / img_max_value

    return img_float


def img_write(filename, img, comp_val=9):
    """
    OpenCV の BGR 配列が怖いので並べ替えるwrapperを用意。
    """
    if img.shape[2] == 3:
        # cv2.imwrite(filename, img[:, :, ::-1], [cv2.IMWRITE_PNG_COMPRESSION, 9])
        img_save = img[:, :, ::-1]
    elif img.shape[2] == 4:
        shape = img.shape
        r, g, b, a = np.dsplit(img, 4)
        img_save = np.dstack((b, g, r, a)).reshape((shape))
    else:
        raise ValueError("not supported img shape for immg_write")
    cv2.imwrite(filename, img_save, [cv2.IMWRITE_PNG_COMPRESSION, comp_val])


def img_wirte_float_as_16bit_int(filename, img_float, comp_val=9):
    img_int = np.uint16(np.round(np.clip(img_float, 0.0, 1.0) * 0xFFFF))
    img_write(filename, img_int, comp_val)


def img_wirte_float_as_16bit_int_with_icc(
        filename, img_float, icc_profile_name, comp_val=9):
    temp_fname = add_suffix_to_filename(filename, "_temp")
    img_int = np.uint16(np.round(np.clip(img_float, 0.0, 1.0) * 0xFFFF))
    img_write(temp_fname, img_int, comp_val)
    cmd = [
        'convert', temp_fname, '-profile', icc_profile_name, filename]
    subprocess.run(cmd)
    os.remove(temp_fname)


def equal_devision(length, div_num):
    """
    # 概要
    length を div_num で分割する。
    端数が出た場合は誤差拡散法を使って上手い具合に分散させる。
    """
    if div_num < 1:
        return []

    base = length / div_num
    ret_array = [base for x in range(div_num)]

    # 誤差拡散法を使った辻褄合わせを適用
    # -------------------------------------------
    diff = 0
    for idx in range(div_num):
        diff += math.modf(ret_array[idx])[0]
        if diff >= 1.0:
            diff -= 1.0
            ret_array[idx] = int(math.floor(ret_array[idx]) + 1)
        else:
            ret_array[idx] = int(math.floor(ret_array[idx]))

    # 計算誤差により最終点が +1 されない場合への対処
    # -------------------------------------------
    diff = length - sum(ret_array)
    if diff != 0:
        ret_array[-1] += diff

    # 最終確認
    # -------------------------------------------
    if length != sum(ret_array):
        raise ValueError("the output of equal_division() is abnormal.")

    return ret_array


def xy_to_rgb(xy, name='ITU-R BT.2020', normalize='maximum', specific=None):
    """
    xy値からRGB値を算出する。
    いい感じに正規化もしておく。

    Parameters
    ----------
    xy : array_like
        xy value.
    name : string
        color space name.
    normalize : string
        normalize method. You can select 'maximum', 'specific' or None.

    Returns
    -------
    array_like
        rgb value. the value is normalized.
    """
    illuminant_XYZ = D65_WHITE
    illuminant_RGB = D65_WHITE
    chromatic_adaptation_transform = 'CAT02'
    large_xyz_to_rgb_matrix = get_xyz_to_rgb_matrix(name)
    if normalize == 'specific':
        xyY = xy_to_xyY(xy)
        xyY[..., 2] = specific
        large_xyz = xyY_to_XYZ(xyY)
    else:
        large_xyz = xy_to_XYZ(xy)

    rgb = XYZ_to_RGB(large_xyz, illuminant_XYZ, illuminant_RGB,
                     large_xyz_to_rgb_matrix,
                     chromatic_adaptation_transform)

    """
    そのままだとビデオレベルが低かったりするので、
    各ドット毎にRGB値を正規化＆最大化する。必要であれば。
    """
    if normalize == 'maximum':
        rgb = normalise_maximum(rgb, axis=-1)
    else:
        if(np.sum(rgb > 1.0) > 0):
            print("warning: over flow has occured at xy_to_rgb")
        if(np.sum(rgb < 0.0) > 0):
            print("warning: under flow has occured at xy_to_rgb")
        rgb[rgb < 0] = 0
        rgb[rgb > 1.0] = 1.0

    return rgb


def get_white_point(name):
    """
    white point を求める。CIE1931ベース。
    """
    if name != "DCI-P3":
        illuminant = RGB_COLOURSPACES[name].illuminant
        white_point = CCS_ILLUMINANTS[CMFS_NAME][illuminant]
    else:
        white_point = CCS_ILLUMINANTS[CMFS_NAME]["D65"]

    return white_point


# def plot_chromaticity_diagram(
#         rate=480/755.0*2, xmin=0.0, xmax=0.8, ymin=0.0, ymax=0.9, **kwargs):
#     # キーワード引数の初期値設定
#     # ------------------------------------
#     monitor_primaries = kwargs.get('monitor_primaries', None)
#     secondaries = kwargs.get('secondaries', None)
#     test_scatter = kwargs.get('test_scatter', None)
#     intersection = kwargs.get('intersection', None)

#     # プロット用データ準備
#     # ---------------------------------
#     xy_image = get_chromaticity_image(
#         xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax)
#     cmf_xy = _get_cmfs_xy()

#     bt709_gamut, _ = get_primaries(name=cs.BT709)
#     bt2020_gamut, _ = get_primaries(name=cs.BT2020)
#     dci_p3_gamut, _ = get_primaries(name=cs.P3_D65)
#     ap0_gamut, _ = get_primaries(name=cs.ACES_AP0)
#     ap1_gamut, _ = get_primaries(name=cs.ACES_AP1)
#     xlim = (min(0, xmin), max(0.8, xmax))
#     ylim = (min(0, ymin), max(0.9, ymax))

#     ax1 = pu.plot_1_graph(fontsize=20 * rate,
#                           figsize=((xmax - xmin) * 10 * rate,
#                                    (ymax - ymin) * 10 * rate),
#                           graph_title="CIE1931 Chromaticity Diagram",
#                           graph_title_size=None,
#                           xlabel=None, ylabel=None,
#                           axis_label_size=None,
#                           legend_size=18 * rate,
#                           xlim=xlim, ylim=ylim,
#                           xtick=[x * 0.1 + xmin for x in
#                                  range(int((xlim[1] - xlim[0])/0.1) + 1)],
#                           ytick=[x * 0.1 + ymin for x in
#                                  range(int((ylim[1] - ylim[0])/0.1) + 1)],
#                           xtick_size=17 * rate,
#                           ytick_size=17 * rate,
#                           linewidth=4 * rate,
#                           minor_xtick_num=2,
#                           minor_ytick_num=2)
#     ax1.plot(cmf_xy[..., 0], cmf_xy[..., 1], '-k', lw=3.5*rate, label=None)
#     ax1.plot((cmf_xy[-1, 0], cmf_xy[0, 0]), (cmf_xy[-1, 1], cmf_xy[0, 1]),
#              '-k', lw=3.5*rate, label=None)
#     ax1.plot(bt709_gamut[:, 0], bt709_gamut[:, 1],
#              c=UNIVERSAL_COLOR_LIST[0], label="BT.709", lw=2.75*rate)
#     ax1.plot(bt2020_gamut[:, 0], bt2020_gamut[:, 1],
#              c=UNIVERSAL_COLOR_LIST[1], label="BT.2020", lw=2.75*rate)
#     ax1.plot(dci_p3_gamut[:, 0], dci_p3_gamut[:, 1],
#              c=UNIVERSAL_COLOR_LIST[2], label="DCI-P3", lw=2.75*rate)
#     ax1.plot(ap1_gamut[:, 0], ap1_gamut[:, 1],
#              c=UNIVERSAL_COLOR_LIST[3], label="ACES AP1", lw=2.75*rate)
#     ax1.plot(ap0_gamut[:, 0], ap0_gamut[:, 1],
#              c=UNIVERSAL_COLOR_LIST[4], label="ACES AP0", lw=2.75*rate)
#     if monitor_primaries is not None:
#         ax1.plot(monitor_primaries[:, 0], monitor_primaries[:, 1],
#                  c="#202020", label="???", lw=3*rate)
#     if secondaries is not None:
#         xy, rgb = secondaries
#         ax1.scatter(xy[..., 0], xy[..., 1], s=700*rate, marker='s', c=rgb,
#                     edgecolors='#404000', linewidth=2*rate)
#     if test_scatter is not None:
#         xy, rgb = test_scatter
#         ax1.scatter(xy[..., 0], xy[..., 1], s=300*rate, marker='s', c=rgb,
#                     edgecolors='#404040', linewidth=2*rate)
#     if intersection is not None:
#         ax1.scatter(intersection[..., 0], intersection[..., 1],
#                     s=300*rate, marker='s', c='#CCCCCC',
#                     edgecolors='#404040', linewidth=2*rate)

#     ax1.imshow(xy_image, extent=(xmin, xmax, ymin, ymax))
#     plt.legend(loc='upper right')
#     plt.savefig('temp_fig.png', bbox_inches='tight')
#     plt.show()


def get_csf_color_image(width=640, height=480,
                        lv1=np.uint16(np.array([1.0, 1.0, 1.0]) * 1023 * 0x40),
                        lv2=np.uint16(np.array([1.0, 1.0, 1.0]) * 512 * 0x40),
                        stripe_num=18):
    """
    長方形を複数個ズラして重ねることでCSFパターンっぽいのを作る。
    入力信号レベルは16bitに限定する。

    Parameters
    ----------
    width : numeric.
        width of the pattern image.
    height : numeric.
        height of the pattern image.
    lv1 : numeric
        video level 1. this value must be 10bit.
    lv2 : numeric
        video level 2. this value must be 10bit.
    stripe_num : numeric
        number of the stripe.

    Returns
    -------
    array_like
        a cms pattern image.
    """
    width_list = equal_devision(width, stripe_num)
    height_list = equal_devision(height, stripe_num)
    h_pos_list = equal_devision(width // 2, stripe_num)
    v_pos_list = equal_devision(height // 2, stripe_num)
    lv1_16bit = lv1
    lv2_16bit = lv2
    img = np.zeros((height, width, 3), dtype=np.uint16)

    width_temp = width
    height_temp = height
    h_pos_temp = 0
    v_pos_temp = 0
    for idx in range(stripe_num):
        lv = lv1_16bit if (idx % 2) == 0 else lv2_16bit
        temp_img = np.ones((height_temp, width_temp, 3), dtype=np.uint16)
        # temp_img *= lv
        temp_img[:, :] = lv
        ed_pos_h = h_pos_temp + width_temp
        ed_pos_v = v_pos_temp + height_temp
        img[v_pos_temp:ed_pos_v, h_pos_temp:ed_pos_h] = temp_img
        width_temp -= width_list[stripe_num - 1 - idx]
        height_temp -= height_list[stripe_num - 1 - idx]
        h_pos_temp += h_pos_list[idx]
        v_pos_temp += v_pos_list[idx]

    return img


def log_tick_formatter(val, pos=None):
    return "{:.0e}".format(10**val)


def get_3d_grid_cube_format(grid_num=4):
    """
    # 概要
    (0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0), (0, 0, 1), ...
    みたいな配列を返す。
    CUBE形式の3DLUTを作成する時に便利。
    """

    base = np.linspace(0, 1, grid_num)
    ones_x = np.ones((grid_num, grid_num, 1))
    ones_y = np.ones((grid_num, 1, grid_num))
    ones_z = np.ones((1, grid_num, grid_num))
    r_3d = base[np.newaxis, np.newaxis, :] * ones_x
    g_3d = base[np.newaxis, :, np.newaxis] * ones_y
    b_3d = base[:, np.newaxis, np.newaxis] * ones_z
    r_3d = r_3d.flatten()
    g_3d = g_3d.flatten()
    b_3d = b_3d.flatten()

    return np.dstack((r_3d, g_3d, b_3d))


def quadratic_bezier_curve(t, p0, p1, p2, samples=1024):
    # x = ((1 - t) ** 2) * p0[0] + 2 * (1 - t) * t * p1[0]\
    #     + (t ** 2) * p2[0]
    # y = ((1 - t) ** 2) * p0[1] + 2 * (1 - t) * t * p1[1]\
    #     + (t ** 2) * p2[1]

    x = ((1 - t) ** 2) * p0[0] + 2 * (1 - t) * t * p1[0]\
        + (t ** 2) * p2[0]
    y = ((1 - t) ** 2) * p0[1] + 2 * (1 - t) * t * p1[1]\
        + (t ** 2) * p2[1]

    # ax1 = pu.plot_1_graph(fontsize=20,
    #                       figsize=(10, 8),
    #                       graph_title="Title",
    #                       graph_title_size=None,
    #                       xlabel="X Axis Label", ylabel="Y Axis Label",
    #                       axis_label_size=None,
    #                       legend_size=17,
    #                       xlim=None,
    #                       ylim=None,
    #                       xtick=None,
    #                       ytick=None,
    #                       xtick_size=None, ytick_size=None,
    #                       linewidth=3,
    #                       minor_xtick_num=None,
    #                       minor_ytick_num=None)
    # ax1.plot(x, y, label='aaa')
    # plt.legend(loc='upper left')
    # plt.show()


def float_to_int_to_float(data, normalized_val=1024):
    """
    Examples
    --------
    >>> data = [0.00, 0.25, 0.5, 0.75, 1.00]
    >>> data = float_to_int_to_float(data=data, normalized_val=1024)
    >>> print(np.uint16(np.round(data * 1023)))
    [   0  256  512  768 1023]
    """
    maximum_val = normalized_val - 1

    out_data = np.array(data) * normalized_val
    out_data[out_data > maximum_val] = maximum_val

    out_data = out_data / maximum_val

    return out_data


def gen_step_ramp_v2(
        width=1024, height=128, num_of_step=17, max_val=1.0, color=[1, 0, 0]):
    cv_list = max_val / (num_of_step - 1) * np.arange(num_of_step)
    block_width_list = equal_devision(width, num_of_step)
    line_data_list = []
    for block_idx in range(num_of_step):
        line_data\
            = np.ones((1, block_width_list[block_idx], 3))\
            * cv_list[block_idx] * np.array(color)
        line_data_list.append(line_data)
    line_img = np.hstack(line_data_list)
    img = h_color_line_to_img(line=line_img, height=height)
    return img, cv_list


def gen_step_gradation(width=1024, height=128, step_num=17,
                       bit_depth=10, color=(1.0, 1.0, 1.0),
                       direction='h', debug=False):
    """
    # 概要
    階段状に変化するグラデーションパターンを作る。
    なお、引数の調整により正確に1階調ずつ変化するパターンも作成可能。

    # 注意事項
    正確に1階調ずつ変化するグラデーションを作る場合は
    ```step_num = (2 ** bit_depth) + 1```
    となるようにパラメータを指定すること。具体例は以下のExample参照。

    # Example
    ```
    grad_8 = gen_step_gradation(width=grad_width, height=grad_height,
                                step_num=257, bit_depth=8,
                                color=(1.0, 1.0, 1.0), direction='h')

    grad_10 = gen_step_gradation(width=grad_width, height=grad_height,
                                 step_num=1025, bit_depth=10,
                                 color=(1.0, 1.0, 1.0), direction='h')
    ```
    """
    max = 2 ** bit_depth

    # グラデーション方向設定
    # ----------------------
    if direction == 'h':
        pass
    else:
        temp = height
        height = width
        width = temp

    if (max + 1 != step_num):
        """
        1階調ずつの増加では無いパターン。
        末尾のデータが 256 や 1024 になるため -1 する。
        """
        val_list = np.linspace(0, max, step_num)
        val_list[-1] -= 1
    else:
        """
        正確に1階調ずつ変化するパターン。
        末尾のデータが 256 や 1024 になるため除外する。
        """
        val_list = np.linspace(0, max, step_num)[0:-1]
        step_num -= 1  # step_num は 引数で余計に +1 されてるので引く

        # 念のため1階調ずつの変化か確認
        # ---------------------------
        diff = val_list[1:] - val_list[0:-1]
        if (diff == 1).all():
            pass
        else:
            raise ValueError("calculated value is invalid.")

    # まずは水平1LINEのグラデーションを作る
    # -----------------------------------
    step_length_list = equal_devision(width, step_num)
    step_bar_list = []
    for step_idx, length in enumerate(step_length_list):
        step = [np.ones((length)) * color[c_idx] * val_list[step_idx]
                for c_idx in range(3)]
        if direction == 'h':
            step = np.dstack(step)
            step_bar_list.append(step)
            step_bar = np.hstack(step_bar_list)
        else:
            step = np.dstack(step).reshape((length, 1, 3))
            step_bar_list.append(step)
            step_bar = np.vstack(step_bar_list)

    # ブロードキャストを利用して2次元に拡張する
    # ------------------------------------------
    if direction == 'h':
        img = step_bar * np.ones((height, 1, 3))
    else:
        img = step_bar * np.ones((1, height, 3))

    # np.uint16 にコンバート
    # ------------------------------
    # img = np.uint16(np.round(img * (2 ** (16 - bit_depth))))

    if debug:
        preview_image(img, 'rgb')

    return img


def merge(img_a, img_b, pos=(0, 0)):
    """
    img_a に img_b をマージする。
    img_a にデータを上書きする。

    pos = (horizontal_st, vertical_st)
    """
    b_width = img_b.shape[1]
    b_height = img_b.shape[0]

    img_a[pos[1]:b_height+pos[1], pos[0]:b_width+pos[0]] = img_b


def merge_with_alpha(bg_img, fg_img, tf_str=tf.SRGB, pos=(0, 0)):
    """
    合成する。

    Parameters
    ----------
    bg_img : array_like(float, 3-channel)
        image data.
    fg_img : array_like(float, 4-channel)
        image data
    tf : strings
        transfer function
    pos : list(int)
        (pos_h, pos_v)
    """
    f_width = fg_img.shape[1]
    f_height = fg_img.shape[0]

    bg_merge_area = bg_img[pos[1]:f_height+pos[1], pos[0]:f_width+pos[0]]
    bg_linear = tf.eotf_to_luminance(bg_merge_area, tf_str)
    fg_linear = tf.eotf_to_luminance(fg_img, tf_str)
    alpha = fg_linear[:, :, 3:] / tf.PEAK_LUMINANCE[tf_str]

    out_linear = (1 - alpha) * bg_linear + alpha * fg_linear[:, :, :-1]
    out_merge_area = tf.oetf_from_luminance(out_linear, tf_str)
    bg_img[pos[1]:f_height+pos[1], pos[0]:f_width+pos[0]] = out_merge_area

    return bg_img


def merge_with_alpha2(bg_img, fg_img, tf_str=tf.SRGB, pos=(0, 0)):
    """
    合成する。今までは無駄が多かったので、必要なところだけ計算する。
    具体的には alpha != 0 の最小の矩形を探して、その領域だけ合成する。

    Parameters
    ----------
    bg_img : array_like(float, 3-channel)
        image data.
    fg_img : array_like(float, 4-channel)
        image data
    tf : strings
        transfer function
    pos : list(int)
        (pos_h, pos_v)
    """
    f_width = fg_img.shape[1]
    f_height = fg_img.shape[0]

    not_transp = fg_img[..., 3] > 0

    bg_merge_area = bg_img[pos[1]:f_height+pos[1], pos[0]:f_width+pos[0]]
    bg_merge_img = bg_merge_area[not_transp]
    fg_merge_img = fg_img[not_transp]
    bg_linear = tf.eotf_to_luminance(bg_merge_img, tf_str)
    fg_linear = tf.eotf_to_luminance(fg_merge_img, tf_str)
    alpha = fg_linear[..., 3:] / tf.PEAK_LUMINANCE[tf_str]

    out_linear = (1 - alpha) * bg_linear + alpha * fg_linear[..., :-1]
    out_merge_area = tf.oetf_from_luminance(out_linear, tf_str)
    bg_img[pos[1]:f_height+pos[1], pos[0]:f_width+pos[0]][not_transp]\
        = out_merge_area


def dot_pattern(dot_size=4, repeat=4, color=np.array([1.0, 1.0, 1.0])):
    """
    dot pattern 作る。

    Parameters
    ----------
    dot_size : integer
        dot size.
    repeat : integer
        The number of high-low pairs.
    color : array_like
        color value.

    Returns
    -------
    array_like
        dot pattern image.

    """
    # 水平・垂直のピクセル数
    pixel_num = dot_size * 2 * repeat

    # High-Log の 論理配列を生成
    even_logic = [(np.arange(pixel_num) % (dot_size * 2)) - dot_size < 0]
    even_logic = np.dstack((even_logic, even_logic, even_logic))
    odd_logic = np.logical_not(even_logic)

    # 着色
    color = color.reshape((1, 1, 3))
    even_line = (np.ones((1, pixel_num, 3)) * even_logic) * color
    odd_line = (np.ones((1, pixel_num, 3)) * odd_logic) * color

    # V方向にコピー＆Even-Oddの結合
    even_block = np.repeat(even_line, dot_size, axis=0)
    odd_block = np.repeat(odd_line, dot_size, axis=0)
    pair_block = np.vstack((even_block, odd_block))

    img = np.vstack([pair_block for x in range(repeat)])

    return img


def complex_dot_pattern(kind_num=3, whole_repeat=2,
                        fg_color=np.array([1.0, 1.0, 1.0]),
                        bg_color=np.array([0.15, 0.15, 0.15])):
    """
    dot pattern 作る。

    Parameters
    ----------
    kind_num : integer
        作成するドットサイズの種類。
        例えば、kind_num=3 ならば、1dot, 2dot, 4dot のパターンを作成。
    whole_repeat : integer
        異なる複数種類のドットパターンの組数。
        例えば、kind_num=3, whole_repeat=2 ならば、
        1dot, 2dot, 4dot のパターンを水平・垂直に2組作る。
    fg_color : array_like
        foreground color value.
    bg_color : array_like
        background color value.
    reduce : bool
        HDRテストパターンの3840x2160専用。縦横を半分にする。

    Returns
    -------
    array_like
        dot pattern image.

    """
    max_dot_width = 2 ** kind_num
    img_list = []
    for size_idx in range(kind_num)[::-1]:
        dot_size = 2 ** size_idx
        repeat = max_dot_width // dot_size
        dot_img = dot_pattern(dot_size, repeat, fg_color)
        img_list.append(dot_img)
        img_list.append(np.ones_like(dot_img) * bg_color)
        # preview_image(dot_img)

    line_upper_img = np.hstack(img_list)
    line_upper_img = np.hstack([line_upper_img for x in range(whole_repeat)])
    line_lower_img = line_upper_img.copy()[:, ::-1, :]
    h_unit_img = np.vstack((line_upper_img, line_lower_img))

    img = np.vstack([h_unit_img for x in range(kind_num * whole_repeat)])
    # preview_image(img)
    # cv2.imwrite("hoge.tiff", np.uint8(img * 0xFF)[..., ::-1])

    return img


def make_csf_color_image(width=640, height=640,
                         lv1=np.array([940, 940, 940], dtype=np.uint16),
                         lv2=np.array([1023, 1023, 1023], dtype=np.uint16),
                         stripe_num=6):
    """
    長方形を複数個ズラして重ねることでCSFパターンっぽいのを作る。
    入力信号レベルは10bitに限定する。

    Parameters
    ----------
    width : numeric.
        width of the pattern image.
    height : numeric.
        height of the pattern image.
    lv1 : array_like
        video level 1. this value must be 10bit.
    lv2 : array_like
        video level 2. this value must be 10bit.
    stripe_num : numeric
        number of the stripe.

    Returns
    -------
    array_like
        a cms pattern image.
    """
    width_list = equal_devision(width, stripe_num)
    height_list = equal_devision(height, stripe_num)
    h_pos_list = equal_devision(width // 2, stripe_num)
    v_pos_list = equal_devision(height // 2, stripe_num)
    img = np.zeros((height, width, 3), dtype=np.uint16)

    width_temp = width
    height_temp = height
    h_pos_temp = 0
    v_pos_temp = 0
    for idx in range(stripe_num):
        lv = lv1 if (idx % 2) == 0 else lv2
        temp_img = np.ones((height_temp, width_temp, 3), dtype=np.uint16)
        temp_img = temp_img * lv.reshape((1, 1, 3))
        ed_pos_h = h_pos_temp + width_temp
        ed_pos_v = v_pos_temp + height_temp
        img[v_pos_temp:ed_pos_v, h_pos_temp:ed_pos_h] = temp_img
        width_temp -= width_list[stripe_num - 1 - idx]
        height_temp -= height_list[stripe_num - 1 - idx]
        h_pos_temp += h_pos_list[idx]
        v_pos_temp += v_pos_list[idx]

    # preview_image(img / 1023)

    return img


def make_tile_pattern(
        width=480, height=960, h_tile_num=4,
        v_tile_num=4, low_level=(0.5, 0.5, 0.5),
        high_level=(1.0, 1.0, 1.0), dtype=np.uint16):
    """
    タイル状の縞々パターンを作る
    """
    width_array = equal_devision(width, h_tile_num)
    height_array = equal_devision(height, v_tile_num)
    high_level = np.array(high_level, dtype=dtype)
    low_level = np.array(low_level, dtype=dtype)

    v_buf = []

    for v_idx, height in enumerate(height_array):
        h_buf = []
        for h_idx, width in enumerate(width_array):
            tile_judge = (h_idx + v_idx) % 2 == 0
            h_temp = np.zeros((height, width, 3), dtype=dtype)
            h_temp[:, :] = high_level if tile_judge else low_level
            h_buf.append(h_temp)

        v_buf.append(np.hstack(h_buf))
    img = np.vstack(v_buf)
    # preview_image(img/1024.0)
    return img


def get_marker_idx(img, marker_value):
    return np.all(img == marker_value, axis=-1)


def make_ycbcr_checker(height=480, v_tile_num=4):
    """
    YCbCr係数誤りを確認するテストパターンを作る。
    正直かなり汚い組み方です。雑に作ったパターンを悪魔合体させています。

    Parameters
    ----------
    height : numeric.
        height of the pattern image.
    v_tile_num : numeric
        number of the tile in the vertical direction.

    Note
    ----
    横長のパターンになる。以下の式が成立する。

    ```
    h_tile_num = v_tile_num * 2
    width = height * 2
    ```

    Returns
    -------
    array_like
        ycbcr checker image
    """

    cyan_img = make_tile_pattern(width=height, height=height,
                                 h_tile_num=v_tile_num,
                                 v_tile_num=v_tile_num,
                                 low_level=[0, 990, 990],
                                 high_level=[0, 1023, 1023])
    magenta_img = make_tile_pattern(width=height, height=height,
                                    h_tile_num=v_tile_num,
                                    v_tile_num=v_tile_num,
                                    low_level=[990, 0, 312],
                                    high_level=[1023, 0, 312])

    out_img = np.hstack([cyan_img, magenta_img])

    # preview_image(out_img/1023.0)

    return out_img


def plot_color_checker_image(rgb, rgb2=None, size=(1920, 1080),
                             block_size=1/4.5, side_trim=True):
    """
    ColorCheckerをプロットする

    Parameters
    ----------
    rgb : array_like
        RGB value of the ColorChecker.
        RGB's shape must be (24, 3).
    rgb2 : array_like
        It's a optional parameter.
        If You want to draw two different ColorCheckers,
        set the RGB value to this variable.
    size : tuple
        canvas size.
    block_size : float
        A each block's size.
        This value is ratio to height of the canvas.

    Returns
    -------
    array_like
        A ColorChecker image.

    """
    # 基本パラメータ算出
    # --------------------------------------
    cc_h_num = 6
    cc_v_num = 4
    img_height = size[1]
    img_width = size[0]
    patch_width = int(img_height * block_size)
    patch_height = patch_width
    patch_space = int(
        (img_height - patch_height * cc_v_num) / (cc_v_num + 1))

    patch_st_h = int(
        img_width / 2.0
        - patch_width * cc_h_num / 2.0
        - patch_space * (cc_h_num / 2.0 - 0.5)
    )
    patch_st_v = int(
        img_height / 2.0
        - patch_height * cc_v_num / 2.0
        - patch_space * (cc_v_num / 2.0 - 0.5)
    )

    # 24ループで1枚の画像に24パッチを描画
    # -------------------------------------------------
    img_all_patch = np.zeros((img_height, img_width, 3))
    for idx in range(cc_h_num * cc_v_num):
        v_idx = idx // cc_h_num
        h_idx = (idx % cc_h_num)
        patch = np.ones((patch_height, patch_width, 3))
        patch[:, :] = rgb[idx]
        st_h = patch_st_h + (patch_width + patch_space) * h_idx
        st_v = patch_st_v + (patch_height + patch_space) * v_idx
        img_all_patch[st_v:st_v+patch_height, st_h:st_h+patch_width] = patch

        ## rgb2 for triangle plot
        # pt2 = (st_h + patch_width, st_v)  # upper right
        # pt3 = (st_h, st_v + patch_height)  # lower left
        # pt4 = (st_h + patch_width, st_v + patch_height)  # lower right
        # pts = np.array((pt2, pt3, pt4))
        # sub_color = rgb[idx].tolist() if rgb2 is None else rgb2[idx].tolist()
        # cv2.fillPoly(img_all_patch, [pts], sub_color)

        # rgb2 for rectangle in rectangle
        pt2 = (
            st_h + np.uint16(np.round(patch_width/4*3)),
            st_v + patch_width//4)  # upper right
        pt3 = (
            st_h + np.uint16(np.round(patch_width/4*3)),
            st_v + np.uint16(np.round(patch_width/4*3)))  # upper right
        pt4 = (
            st_h + patch_width//4,
            st_v + np.uint16(np.round(patch_width/4*3)))  # upper right
        pt5 = (st_h + patch_width//4, st_v + patch_width//4)
        pts = np.array((pt2, pt3, pt4, pt5))
        sub_color = rgb[idx].tolist() if rgb2 is None else rgb2[idx].tolist()
        cv2.fillPoly(img_all_patch, [pts], sub_color)

    # preview_image(img_all_patch)
    if side_trim:
        img_trim_h_st = patch_st_h - patch_space
        img_trim_h_ed = patch_st_h + (patch_width + patch_space) * 6
        img_all_patch = img_all_patch[:, img_trim_h_st:img_trim_h_ed]

    return img_all_patch


def get_log10_x_scale(
        sample_num=8, ref_val=1.0, min_exposure=-1, max_exposure=6):
    """
    Log10スケールのx軸データを作る。

    Examples
    --------
    >>> get_log2_x_scale(
    ...     sample_num=8, ref_val=1.0, min_exposure=-1, max_exposure=6)
    array([  1.0000e-01   1.0000e+00   1.0000e+01   1.0000e+02
             1.0000e+03   1.0000e+04   1.0000e+05   1.0000e+06])
    """
    x_min = np.log10(ref_val * (10 ** min_exposure))
    x_max = np.log10(ref_val * (10 ** max_exposure))
    x = np.linspace(x_min, x_max, sample_num)

    return 10.0 ** x


def get_log2_x_scale(
        sample_num=32, ref_val=1.0, min_exposure=-6.5, max_exposure=6.5):
    """
    Log2スケールのx軸データを作る。

    Examples
    --------
    >>> get_log2_x_scale(sample_num=10, min_exposure=-4.0, max_exposure=4.0)
    array([[  0.0625       0.11573434   0.214311     0.39685026   0.73486725
              1.36079      2.5198421    4.66611616   8.64047791  16.        ]])
    """
    x_min = np.log2(ref_val * (2 ** min_exposure))
    x_max = np.log2(ref_val * (2 ** max_exposure))
    x = np.linspace(x_min, x_max, sample_num)

    return 2.0 ** x


def shaper_func_linear_to_log2(
        x, mid_gray=0.18, min_exposure=-6.5, max_exposure=6.5):
    """
    ACESutil.Lin_to_Log2_param.ctl を参考に作成。
    https://github.com/ampas/aces-dev/blob/master/transforms/ctl/utilities/ACESutil.Lin_to_Log2_param.ctl

    Parameters
    ----------
    x : array_like
        linear data.
    mid_gray : float
        18% gray value on linear scale.
    min_exposure : float
        minimum value on log scale.
    max_exposure : float
        maximum value on log scale.

    Returns
    -------
    array_like
        log2 value that is transformed from linear x value.

    Examples
    --------
    >>> shaper_func_linear_to_log2(
    ...     x=0.18, mid_gray=0.18, min_exposure=-6.5, max_exposure=6.5)
    0.5
    >>> shaper_func_linear_to_log2(
    ...     x=np.array([0.00198873782209, 16.2917402385])
    ...     mid_gray=0.18, min_exposure=-6.5, max_exposure=6.5)
    array([  1.58232402e-13   1.00000000e+00])
    """
    # log2空間への変換。mid_gray が 0.0 となるように補正
    y = np.log2(x / mid_gray)

    # min, max の範囲で正規化。
    y_normalized = (y - min_exposure) / (max_exposure - min_exposure)

    y_normalized[y_normalized < 0] = 0

    return y_normalized


def shaper_func_log2_to_linear(
        x, mid_gray=0.18, min_exposure=-6.5, max_exposure=6.5):
    """
    ACESutil.Log2_to_Lin_param.ctl を参考に作成。
    https://github.com/ampas/aces-dev/blob/master/transforms/ctl/utilities/ACESutil.Log2_to_Lin_param.ctl

    Log2空間の補足は shaper_func_linear_to_log2() の説明を参照

    Examples
    --------
    >>> x = np.array([0.0, 1.0])
    >>> shaper_func_log2_to_linear(
    ...     x, mid_gray=0.18, min_exposure=-6.5, max_exposure=6.5)
    array([0.00198873782209, 16.2917402385])
    """
    x_re_scale = x * (max_exposure - min_exposure) + min_exposure
    y = (2.0 ** x_re_scale) * mid_gray
    # plt.plot(x, y)
    # plt.show()

    return y


def draw_straight_line(img, pt1, pt2, color, thickness):
    """
    直線を引く。OpenCV だと 8bit しか対応してないっぽいので自作。

    Parameters
    ----------
    img : array_like
        image data.
    pt1 : list(pos_h, pos_v)
        start point.
    pt2 : list(pos_h, pos_v)
        end point.
    color : array_like
        color
    thickness : int
        thickness.

    Returns
    -------
    array_like
        image data with line.

    Notes
    -----
    thickness のパラメータは pt1 の点から右下方向に効きます。
    pt1 を中心として太さではない事に注意。

    Examples
    --------
    >>> pt1 = (0, 0)
    >>> pt2 = (1920, 0)
    >>> color = (940, 940, 940)
    >>> thickness = 4
    >>> draw_straight_line(img, pt1, pt2, color, thickness)
    """
    # parameter check
    if (pt1[0] != pt2[0]) and (pt1[1] != pt2[1]):
        raise ValueError("invalid pt1, pt2 parameters")

    # check direction
    if pt1[0] == pt2[0]:
        thickness_direction = 'h'
    else:
        thickness_direction = 'v'

    if thickness_direction == 'h':
        for h_idx in range(thickness):
            img[pt1[1]:pt2[1], pt1[0] + h_idx, :] = color

    elif thickness_direction == 'v':
        for v_idx in range(thickness):
            img[pt1[1] + v_idx, pt1[0]:pt2[0], :] = color


def draw_outline(img, fg_color, outline_width):
    """
    img に対して外枠線を引く

    Parameters
    ----------
    img : array_like
        image data.
    fg_color : array_like
        color
    outline_width : int
        thickness.

    Returns
    -------
    array_like
        image data with line.

    Examples
    --------
    >>> img = np.zeros((1080, 1920, 3))
    >>> color = (940, 940, 940)
    >>> thickness = 2
    >>> draw_outline(img, color, thickness)
    """
    width = img.shape[1]
    height = img.shape[0]
    # upper left
    pt1 = (0, 0)
    pt2 = (width, 0)
    draw_straight_line(
        img, pt1, pt2, fg_color, outline_width)
    pt1 = (0, 0)
    pt2 = (0, height)
    draw_straight_line(
        img, pt1, pt2, fg_color, outline_width)
    # lower right
    pt1 = (width - outline_width, 0)
    pt2 = (width - outline_width, height)
    draw_straight_line(
        img, pt1, pt2, fg_color, outline_width)
    pt1 = (0, height - outline_width)
    pt2 = (width, height - outline_width)
    draw_straight_line(
        img, pt1, pt2, fg_color, outline_width)


def convert_luminance_to_color_value(luminance, transfer_function):
    """
    convert from luminance [cd/m2] to rgb code values.

    Parameters
    ----------
    lumiannce : float
        lumiannce. the unit is cd/m2.
    transfer_function : str
        A transfer function's name for my "tf" module.

    Examples
    --------
    >>> convert_luminance_to_color_value(100, tf.GAMMA24)
    [ 1.0  1.0  1.0 ]
    >>> convert_luminance_to_color_value(100, tf.ST2084)
    [ 0.50807842  0.50807842  0.50807842 ]
    """
    code_value = convert_luminance_to_code_value(
        luminance, transfer_function)
    return np.array([code_value, code_value, code_value])


def convert_luminance_to_code_value(luminance, transfer_function):
    """
    convert from luminance [cd/m2] to code value (0.0 - 1.0).

    Parameters
    ----------
    lumiannce : float
        lumiannce. the unit is cd/m2.
    transfer_function : str
        A transfer function's name for my "tf" module.

    Examples
    --------
    >>> convert_luminance_to_code_value(100, tf.GAMMA24)
    1.0
    >>> convert_luminance_to_code_value(100, tf.ST2084)
    0.50807842
    """
    return tf.oetf_from_luminance(luminance, transfer_function)


def calc_rad_patch_idx2(outmost_num=5, current_num=3):
    """
    以下のような、中心がGray、周りは CIELAB 空間の a*b*平面のカラーパッチの
    RGB値のリストを得る。
    https://user-images.githubusercontent.com/3609012/75444470-d3bc5600-59a6-11ea-962b-c315648782a9.png

    得られたデータは並べ替えが済んでいないため、calc_rad_patch_idx2() で
    得られる変換テーブルを使った変換が必要。
    本関数はまさにその変換を行う。
    """
    base = np.arange(outmost_num ** 2).reshape((outmost_num, outmost_num))
    # print(base)
    t_idx = (outmost_num - current_num) // 2
    trimmed = base[t_idx:t_idx+current_num, t_idx:t_idx+current_num]
    # print(trimmed)
    # print(np.arange(current_num**2).reshape((current_num, current_num)))

    half_num = current_num // 2
    conv_idx = []
    for idx in range(half_num):
        val = (current_num ** 2) // 2 + half_num - current_num * idx
        conv_idx.append(val)
    for idx in range(current_num)[::-1]:
        conv_idx.append(idx)
    for idx in range(1, current_num - 1):
        conv_idx.append(idx * current_num)
    for idx in range(current_num):
        val = (current_num ** 2) - current_num + idx
        conv_idx.append(val)
    for idx in range(1, half_num):
        val = (current_num ** 2) - 1 - idx * current_num
        conv_idx.append(val)

    conv_idx = trimmed.flatten()[conv_idx]

    return conv_idx


def _calc_rgb_from_same_lstar_radial_data(
        lstar, temp_chroma, current_num, color_space):
    """
    以下のような、中心がGray、周りは CIELAB 空間の a*b*平面のカラーパッチの
    RGB値のリストを得る。
    https://user-images.githubusercontent.com/3609012/75444470-d3bc5600-59a6-11ea-962b-c315648782a9.png

    得られたデータは並べ替えが済んでいないため、calc_rad_patch_idx2() で
    得られる変換テーブルを使った変換が必要。
    """
    current_patch_num = (current_num - 1) * 4 if current_num > 1 else 1
    rad = np.linspace(0, 2 * np.pi, current_patch_num, endpoint=False)
    ll = np.ones((current_patch_num)) * lstar
    aa = np.cos(rad) * temp_chroma
    bb = np.sin(rad) * temp_chroma
    lab = np.dstack((ll, aa, bb))
    large_xyz = Lab_to_XYZ(lab)
    rgb = XYZ_to_RGB(large_xyz, D65_WHITE, D65_WHITE,
                     color_space.matrix_XYZ_to_RGB)

    return np.clip(rgb, 0.0, 1.0)


def calc_same_lstar_radial_color_patch_data(
        lstar=58, chroma=32.5, outmost_num=9,
        color_space=RGB_COLOURSPACE_BT709,
        transfer_function=tf.GAMMA24):
    """
    以下のような、中心がGray、周りは CIELAB 空間の a*b*平面のカラーパッチの
    RGB値のリストを得る。
    https://user-images.githubusercontent.com/3609012/75444470-d3bc5600-59a6-11ea-962b-c315648782a9.png

    得られた RGB値のリストは最初のデータが画像左上の緑データ、
    最後のデータが画像右下の紫データとなるよう既に**並べ替え**が行われている。

    よってパッチをプロットする場合はRGB値リストの先頭から順にデータを取り出し、
    右下に向かって並べていけば良い。
    """
    patch_num = outmost_num ** 2
    transfer_function = tf.GAMMA24
    rgb_list = np.ones((patch_num, 3))

    current_num_list = range(1, outmost_num + 1, 2)
    chroma_list = np.linspace(0, chroma, len(current_num_list))
    for temp_chroma, current_num in zip(chroma_list, current_num_list):
        current_patch_num = (current_num - 1) * 4 if current_num > 1 else 1
        rgb = _calc_rgb_from_same_lstar_radial_data(
            lstar, temp_chroma, current_num, color_space)
        rgb = np.reshape(rgb, (current_patch_num, 3))
        rgb = tf.oetf(rgb, transfer_function)
        conv_idx = calc_rad_patch_idx2(
            outmost_num=outmost_num, current_num=current_num)
        for idx in range(current_patch_num):
            rgb_list[conv_idx[idx]] = rgb[idx]

    return rgb_list


def _plot_same_lstar_radial_color_patch_data(
        lstar=58, chroma=32.5, outmost_num=9,
        color_space=RGB_COLOURSPACE_BT709,
        transfer_function=tf.GAMMA24):
    patch_size = 1080 // outmost_num
    img = np.ones((1080, 1080, 3)) * 0.0
    rgb = calc_same_lstar_radial_color_patch_data(
        lstar=lstar, chroma=chroma, outmost_num=outmost_num,
        color_space=color_space, transfer_function=transfer_function)

    for idx in range(outmost_num ** 2):
        h_idx = idx % outmost_num
        v_idx = idx // outmost_num
        st_pos = (h_idx * patch_size, v_idx * patch_size)
        temp_img = np.ones((patch_size, patch_size, 3))\
            * rgb[idx][np.newaxis, np.newaxis, :]
        merge(img, temp_img, st_pos)

    cv2.imwrite("hoge2.tiff", np.uint16(np.round(img[:, :, ::-1] * 0xFFFF)))


def get_accelerated_x_1x(sample_num=64):
    """
    単調増加ではなく、加速度が 0→1→0 となるような x を作る

    Parameters
    ----------
    sample_num : int
        the number of the sample.

    Returns
    -------
    array_like
        accelerated value list

    Examples
    --------
    >>> x0 = np.linspace(0, 1, 8)
    >>> x1 = get_accelerated_x_1x(8)
    >>> print(x0)
    >>> [ 0.  0.142  0.285  0.428  0.571  0.714  0.857  1. ]
    >>> print(x1)
    >>> [ 0.  0.049  0.188  0.388  0.611  0.811  0.950  1. ]
    """
    rad = np.linspace(-0.5 * np.pi, 0.5 * np.pi, sample_num)
    x = (np.sin(rad) + 1) / 2

    return x


def get_accelerated_x_2x(sample_num=64):
    """
    単調増加ではなく、加速度が 0→1→0 となるような x を作る。
    加速度が `get_accelerated_x_1x` の2倍！！

    Parameters
    ----------
    sample_num : int
        the number of the sample.

    Returns
    -------
    array_like
        accelerated value list

    Examples
    --------
    >>> x0 = np.linspace(0, 1, 8)
    >>> x2 = get_accelerated_x_2x(8)
    >>> print(x0)
    >>> [ 0.  0.142  0.285  0.428  0.571  0.714  0.857  1. ]
    >>> print(x2)
    >>> [ 0.  0.006  0.084  0.328  0.671  0.915  0.993  1. ]
    """
    rad = np.linspace(-0.5 * np.pi, 0.5 * np.pi, sample_num)
    rad = np.sin(rad) * 0.5 * np.pi
    x = (np.sin(rad) + 1) / 2

    return x


def get_accelerated_x_4x(sample_num=64):
    """
    単調増加ではなく、加速度が 0→1→0 となるような x を作る。
    加速度が `get_accelerated_x_1x` の4倍！！

    Parameters
    ----------
    sample_num : int
        the number of the sample.

    Returns
    -------
    array_like
        accelerated value list
    """
    rad = np.linspace(-0.5 * np.pi, 0.5 * np.pi, sample_num)
    rad = np.sin(rad) * 0.5 * np.pi
    rad = np.sin(rad) * 0.5 * np.pi
    x = (np.sin(rad) + 1) / 2

    return x


def get_accelerated_x_8x(sample_num=64):
    """
    単調増加ではなく、加速度が 0→1→0 となるような x を作る。
    加速度が `get_accelerated_x_1x` の4倍！！

    Parameters
    ----------
    sample_num : int
        the number of the sample.

    Returns
    -------
    array_like
        accelerated value list
    """
    rad = np.linspace(-0.5 * np.pi, 0.5 * np.pi, sample_num)
    rad = np.sin(rad) * 0.5 * np.pi
    rad = np.sin(rad) * 0.5 * np.pi
    rad = np.sin(rad) * 0.5 * np.pi
    x = (np.sin(rad) + 1) / 2

    return x


def generate_color_checker_xyY_value():
    """
    Examples
    --------
    >>> generate_color_checker_xyY_value()
    [[ 0.39916181  0.35821703  0.09957404]
     [ 0.38666482  0.35331627  0.34582235]
     [ 0.24754052  0.26723926  0.18655796]
     [ 0.34080597  0.42951053  0.13142126]
     [ 0.26848903  0.25361348  0.23375919]
     [ 0.2565283   0.35604216  0.41993184]
     [ 0.50876153  0.40192304  0.30411666]
     [ 0.20862318  0.1823299   0.11768058]
     [ 0.46829356  0.31116359  0.18994308]
     [ 0.29632139  0.22020755  0.06464193]
     [ 0.3728984   0.49248238  0.438931  ]
     [ 0.4746098   0.44046371  0.4264411 ]
     [ 0.18582484  0.14631664  0.06157595]
     [ 0.29778907  0.48241582  0.23042375]
     [ 0.5436873   0.32106297  0.12197455]
     [ 0.44853251  0.47262115  0.58671043]
     [ 0.37603747  0.24449703  0.2002475 ]
     [ 0.19355382  0.26377155  0.19799273]
     [ 0.31392609  0.33143225  0.91280001]
     [ 0.3110718   0.3286934   0.58953892]
     [ 0.31029945  0.32830964  0.36333244]
     [ 0.31162542  0.32824052  0.1915373 ]
     [ 0.30724016  0.32465011  0.0883915 ]
     [ 0.30767643  0.32357895  0.03113289]]
    """
    colour_checker_param = CCS_COLOURCHECKERS.get('ColorChecker 2005')
    colour_checker_param\
        = CCS_COLOURCHECKERS.get('ColorChecker 2005')

    data = colour_checker_param.data
    whitepoint = colour_checker_param.illuminant
    temp_xyY = []
    for key in data.keys():
        temp_xyY.append(data[key])
    temp_xyY = np.array(temp_xyY)
    large_xyz = xyY_to_XYZ(temp_xyY)
    M_CAT = matrix_chromatic_adaptation_VonKries(
        xyY_to_XYZ(xy_to_xyY(whitepoint)),
        xyY_to_XYZ(xy_to_xyY(cs.D65)),
        transform="CAT02",
    )
    large_xyz = vector_dot(M_CAT, large_xyz)
    xyY = XYZ_to_xyY(large_xyz)

    return xyY


def generate_color_checker_rgb_value(
        color_space=RGB_COLOURSPACE_BT709, target_white=D65_WHITE):
    """
    Generate the 24 RGB values of the color checker.

    Parameters
    ----------
    color_space : color space
        color space object in `colour` module.

    target_white : array_like
        the xy values of the white point of target color space.

    Returns
    -------
    array_like
        24 RGB values. This is linear. OETF is not applied.

    Examples
    --------
    >>> generate_color_checker_rgb_value(
    ...     color_space=colour.models.RGB_COLOURSPACE_BT709,
    ...     target_white=[0.3127, 0.3290])
    >>> [[ 0.17289286  0.08205728  0.05714562]
    >>>  [ 0.5680292   0.29250401  0.21951748]
    >>>  [ 0.10435534  0.19656108  0.32958666]
    >>>  [ 0.1008804   0.14839018  0.05327639]
    >>>  [ 0.22303549  0.2169701   0.43166537]
    >>>  [ 0.10715338  0.513512    0.41415978]
    >>>  [ 0.74639182  0.20020473  0.03081343]
    >>>  [ 0.05947812  0.10659045  0.39897686]
    >>>  [ 0.5673215   0.08485376  0.11945382]
    >>>  [ 0.11177253  0.04285397  0.14166202]
    >>>  [ 0.34250836  0.5062777   0.0557734 ]
    >>>  [ 0.79262553  0.35803886  0.025485  ]
    >>>  [ 0.01864598  0.05139665  0.28886469]
    >>>  [ 0.054392    0.29876719  0.07187681]
    >>>  [ 0.45628547  0.03075684  0.04092033]
    >>>  [ 0.85379178  0.56503558  0.01475575]
    >>>  [ 0.53533883  0.09006355  0.3047824 ]
    >>>  [-0.03662977  0.24753781  0.39824679]
    >>>  [ 0.91177068  0.91497623  0.89427332]
    >>>  [ 0.57973934  0.59203191  0.59370647]
    >>>  [ 0.35495537  0.36538027  0.36772001]
    >>>  [ 0.19009594  0.19180133  0.19316719]
    >>>  [ 0.08524707  0.08890587  0.09255774]
    >>>  [ 0.03038879  0.03118623  0.03279615]]
    """
    colour_checker_param = CCS_COLOURCHECKERS.get('ColorChecker 2005')
    colour_checker_param\
        = CCS_COLOURCHECKERS.get('ColorChecker 2005')

    data = colour_checker_param.data
    whitepoint = colour_checker_param.illuminant
    temp_xyY = []
    for key in data.keys():
        temp_xyY.append(data[key])
    temp_xyY = np.array(temp_xyY)
    large_xyz = xyY_to_XYZ(temp_xyY)
    M_CAT = matrix_chromatic_adaptation_VonKries(
        xyY_to_XYZ(xy_to_xyY(whitepoint)),
        xyY_to_XYZ(xy_to_xyY(cs.D65)),
        transform="CAT02",
    )
    large_xyz = vector_dot(M_CAT, large_xyz)

    rgb = XYZ_to_RGB(
        XYZ=large_xyz, colourspace=color_space, illuminant=cs.D65)

    return rgb


def make_color_checker_image(rgb, width=1920, padding_rate=0.01):
    """
    6x4 の カラーチェッカーの画像を作る。
    Height は Width から自動計算される。padding_rate で少し値が変わる。
    """
    h_patch_num = 6
    v_patch_num = 4

    # 各種パラメータ計算
    each_padding = int(width * padding_rate + 0.5)
    h_padding_total = each_padding * (h_patch_num + 1)
    h_patch_width_total = width - h_padding_total
    patch_height = h_patch_width_total // h_patch_num
    height = patch_height * v_patch_num + each_padding * (v_patch_num + 1)
    patch_width_list = equal_devision(h_patch_width_total, h_patch_num)

    # パッチを並べる
    img = np.zeros((height, width, 3))
    for v_idx in range(v_patch_num):
        h_pos_st = each_padding
        v_pos_st = each_padding + v_idx * (patch_height + each_padding)
        for h_idx in range(h_patch_num):
            rgb_idx = v_idx * h_patch_num + h_idx
            pos = (h_pos_st, v_pos_st)
            patch_img = np.ones((patch_height, patch_width_list[h_idx], 3))\
                * rgb[rgb_idx]
            merge(img, patch_img, pos)
            h_pos_st += (patch_width_list[h_idx] + each_padding)

    return img


def calc_st_pos_for_centering(bg_size, fg_size):
    """
    Calculate start postion for centering.

    Parameters
    ----------
    bg_size : touple(int)
        (width, height) of the background image.

    fg_size : touple(int)
        (width, height) of the foreground image.

    Returns
    -------
    touple (int)
        (st_pos_h, st_pos_v)

    Examples
    --------
    >>> calc_st_pos_for_centering(bg_size=(1920, 1080), fg_size=(640, 480))
    >>> (640, 300)
    """
    bg_width = bg_size[0]
    bg_height = bg_size[1]

    fg_width = fg_size[0]
    fg_height = fg_size[1]

    st_pos_h = bg_width // 2 - fg_width // 2
    st_pos_v = bg_height // 2 - fg_height // 2

    return (st_pos_h, st_pos_v)


def get_size_from_image(img):
    """
    `calc_st_pos_for_centering()` の引数計算が面倒だったので関数化。
    """
    return (img.shape[1], img.shape[0])


def create_8bit_10bit_id_patch(
        width=512, height=1024, total_step=20, direction='h',
        level=L_LOW_C_LOW, hdr10=False):
    """
    create two images. the one is 8bit precision.
    the onother is 10bit precision.

    Parameters
    ----------
    width : int
        image width
    height : int
        image height
    total_step : int
        step num in 8bit precision.
    direction : str
        "h": horizontal (highly recommended)
        "v": vertical
    level : str
        L_LOW_C_LOW : 'lightness_low_chroma_low'
        L_LOW_C_MIDDLE : 'lightness_low_chroma_middle'
        L_LOW_C_HIGH : 'lightness_low_chroma_high'
        L_MIDDLE_C_LOW : 'lightness_middle_chroma_low'
        L_MIDDLE_C_MIDDLE : 'lightness_middle_chroma_middle'
        L_MIDDLE_C_HIGH : 'lightness_middle_chroma_high'
        L_HIGH_C_LOW : 'lightness_high_chroma_low'
        L_HIGH_C_MIDDLE : 'lightness_high_chroma_middle'
        L_HIGH_C_HIGH : 'lightness_high_chroma_high'
    hdr10 : bool
        False: generate pattern for BT.709 - Gamma2.4
        True: generate pattern for BT.2020 - SMPTE ST2084

    Returns
    -------
    img_out_8bit : ndarray (float)
        img with 8 bit precision.
    img_out_10bit : ndarray (float)
        img with 10bit precision.

    Examples
    --------
    >>> import test_pattern_generator2 as tpg
    >>> img_out_8bit, img_out_10bit = create_8bit_10bit_id_patch(
            width=512, height=1024, total_step=20, direction='h',
            level=tpg.L_LOW_C_LOW)
    >>> tpg.img_wirte_float_as_16bit_int("8bit_img.png", img_8bit)
    >>> tpg.img_wirte_float_as_16bit_int("10bit_img.png", img_10bit)
    """

    ll_cl = np.array([61, 61, 61])  # L=20, C=0
    ll_cm = np.array([68, 58, 46])  # L=20, C=10
    ll_ch = np.array([75, 56, 33])  # L=20, C=20

    lm_cl = np.array([94, 94, 94])   # L=35, C=0
    lm_cm = np.array([103, 90, 78])  # L=35, C=10
    lm_ch = np.array([110, 88, 63])  # L=35, C=20

    lh_cl = np.array([127, 127, 127])  # L=50, C=0
    lh_cm = np.array([141, 126, 114])  # L=50, C=10
    lh_ch = np.array([149, 122, 97])   # L=50, C=20

    if level == L_LOW_C_LOW:
        base_rgb_8bit = ll_cl
    elif level == L_LOW_C_MIDDLE:
        base_rgb_8bit = ll_cm
    elif level == L_LOW_C_HIGH:
        base_rgb_8bit = ll_ch
    elif level == L_MIDDLE_C_LOW:
        base_rgb_8bit = lm_cl
    elif level == L_MIDDLE_C_MIDDLE:
        base_rgb_8bit = lm_cm
    elif level == L_MIDDLE_C_HIGH:
        base_rgb_8bit = lm_ch
    elif level == L_HIGH_C_LOW:
        base_rgb_8bit = lh_cl
    elif level == L_HIGH_C_MIDDLE:
        base_rgb_8bit = lh_cm
    elif level == L_HIGH_C_HIGH:
        base_rgb_8bit = lh_ch
    else:
        print("Warning: invalid level parameter")
        base_rgb_8bit = ll_cl

    if hdr10:
        linear = tf.eotf(base_rgb_8bit / 255, tf.GAMMA24)
        linear_2020 = RGB_to_RGB(
            linear, RGB_COLOURSPACE_BT709, RGB_COLOURSPACE_BT2020)
        linear_2020_gain2x = linear_2020 * 100 * 2
        st2084_2020 = tf.oetf_from_luminance(linear_2020_gain2x, tf.ST2084)
        base_rgb_8bit = np.round(st2084_2020 * 255)
        print(f"base_rgb_hdr10 = {base_rgb_8bit}")

    base_gg = base_rgb_8bit[1]
    rr = base_rgb_8bit[0]
    bb = base_rgb_8bit[2]

    gg_min = base_gg - (total_step // 2)
    gg_max = base_gg + (total_step // 2)

    if direction == 'h':
        patch_len = width
    else:
        patch_len = height

    gg_grad = np.linspace(gg_min, gg_max, patch_len)
    rr_static = np.ones_like(gg_grad) * rr
    bb_static = np.ones_like(gg_grad) * bb
    line = np.dstack((rr_static, gg_grad, bb_static))

    if direction == 'h':
        img_base_8bit_float = line * np.ones((height, 1, 3))
    else:
        line = line.reshape((height, 1, 3))
        img_base_8bit_float = line * np.ones((1, width, 3))

    img_out_float_8bit = img_base_8bit_float / 255
    img_out_8bit = np.round(img_out_float_8bit * 255) / 255
    img_out_10bit = np.round(img_out_float_8bit * 1023) / 1023

    return img_out_8bit, img_out_10bit


class IdPatch8bit10bitGenerator():
    """
    create 8bit 10bit idification image (ndarray float).

    Examples
    --------
    >>> generator = tpg.IdPatch8bit10bitGenerator(
    ...     width=width, height=height, total_step=total_step, level=level,
    ...     slide_step=step)
    >>> frame_num = 180
    >>> fname_8bit_base = "img_8bit_{width}x{height}_{step}step_{div}div_"
    >>> fname_8bit_base += "{level}_{idx:04d}.png"
    >>> fname_8bit_base = str(IMG_SEQ_DIR / fname_8bit_base)
    >>> fname_10bit_base = "img_10bit_{width}x{height}_{step}step_{div}div_"
    >>> fname_10bit_base += "{level}_{idx:04d}.png"
    >>> fname_10bit_base = str(IMG_SEQ_DIR / fname_10bit_base)

    >>> for idx in range(frame_num):
    ...     img_8bit, img_10bit = generator.extract_8bit_10bit_img()
    ...     fname_8bit = fname_8bit_base.format(
    ...         width=width, height=height, step=step, div=total_step,
    ...         level=level, idx=idx)
    ...     fname_10bit = fname_10bit_base.format(
    ...         width=width, height=height, step=step, div=total_step,
    ...         level=level, idx=idx)
    ...     print(fname_8bit)
    ...     tpg.img_wirte_float_as_16bit_int(fname_8bit, img_8bit)
    ...     tpg.img_wirte_float_as_16bit_int(fname_10bit, img_10bit)
    """
    def __init__(
            self, width=512, height=1024, total_step=20,
            level='middle', slide_step=2, hdr10=False,
            scroll_direction='left'):
        self.width = width
        self.height = height
        self.total_step = total_step
        direction = 'h'
        self.step = slide_step
        self.cnt = 0
        self.scroll_direction = scroll_direction

        img_8bit, img_10bit = create_8bit_10bit_id_patch(
            width=self.width, height=self.height, total_step=self.total_step,
            direction=direction, level=level, hdr10=hdr10)

        self.img_8bit_buf = np.hstack([img_8bit, img_8bit])
        self.img_10bit_buf = np.hstack([img_10bit, img_10bit])

    def extract_based_on_cnt(self, img, cnt=None):
        if cnt:
            h_st = cnt % self.width
        else:
            h_st = self.cnt % self.width
        h_ed = h_st + self.width
        return img[:, h_st:h_ed]

    def extract_8bit_10bit_img(self, cnt=None):
        out_8bit = self.extract_based_on_cnt(self.img_8bit_buf, cnt)
        out_10bit = self.extract_based_on_cnt(self.img_10bit_buf, cnt)
        if not cnt:
            if self.scroll_direction == 'left':
                self.cnt += self.step
            elif self.scroll_direction == 'right':
                self.cnt -= self.step

        return out_8bit, out_10bit


def _calc_l_focal_to_cups_lch_array(
        inner_lut, outer_lut, h_val, chroma_num,
        l_focal_max=100, l_focal_min=0):
    l_focal = cgbl.calc_l_focal_specific_hue(
        inner_lut=inner_lut, outer_lut=outer_lut, hue=h_val,
        maximum_l_focal=l_focal_max, minimum_l_focal=l_focal_min)
    cups = cgbl.calc_cusp_specific_hue(lut=outer_lut, hue=h_val)
    cc_max = cups[1]
    chroma = np.linspace(0, cc_max, chroma_num)
    bb = l_focal[0]
    aa = (cups[0] - l_focal[0]) / cc_max
    lightness = aa * chroma + bb
    hue_array = np.ones_like(lightness) * h_val
    lch_array = tstack([lightness, chroma, hue_array])

    return lch_array


def _calc_l_focal_to_cups_lch_array_jzazbz(
        focal_lut, outer_lut, h_val, chroma_num):
    cups = cgbl.calc_cusp_specific_hue(
        lut=outer_lut.lut, hue=h_val,
        lightness_max=outer_lut.ll_max)
    focal_point = cgbl.get_focal_point_from_lut(
        focal_point_lut=focal_lut, h_val=h_val)
    print(f"cups={cups}, focal_point={focal_point}")
    cc_max = cups[1]
    # cc_max = cc_max - 0.005
    chroma = np.linspace(0, cc_max, chroma_num)
    bb = focal_point
    aa = (cups[0] - bb) / cc_max
    lightness = aa * chroma + bb
    hue_array = np.ones_like(lightness) * h_val
    lch_array = tstack([lightness, chroma, hue_array])

    return lch_array


def make_bt2020_bt709_hue_chroma_pattern(
        inner_lut, outer_lut, hue_num, width, height,
        l_focal_max=90, l_focal_min=50):
    """
    Parameters
    ----------
    inner_lut : ndarray
        A inner gamut boundary lut. shape is (N, M, 3).
        N is the number of the Lightness.
        M is the number of the Hue.
    outer_lut : ndarray
        A inner gamut boundary lut. shape is (N, M, 3).
        N is the number of the Lightness.
        M is the number of the Hue.
    width : int
        image width
    height : int
        image height
    hue_num : int
        the number of the hue block.
    l_focal_max : float
        An maximum value for l_focal.
        This is a parameter to prevent the data from changing
        from l_focal to cups transitioning to Out-of-Gamut.
        https://twitter.com/toru_ver15/status/1394645785929666561
    l_focal_min : float
        An minimum value for l_focal.
        This is a parameter to prevent the data from changing
        from l_focal to cups transitioning to Out-of-Gamut.
        https://twitter.com/toru_ver15/status/1394645785929666561

    """
    height_org = height
    font_size = int(20 * height / 1080)
    text_h_margin = int(6 * height / 1080)
    font_path = "/usr/share/fonts/opentype/noto/NotoSansCJKjp-Regular.otf"
    text = '"BT.2020 - DCI-P3 - BT.709 Hue-Chroma Pattern",   '
    text += "Gamma 2.4,   BT.2020,   D65,   Revision 2,   "
    text += "Copyright (C) 2021 - Toru Yoshihara,   "
    text += "https://trev16.hatenablog.com/"
    hue = np.linspace(0, 360, hue_num, endpoint=False)
    text_width, text_height = fc.get_text_width_height(
        text=text, font_path=font_path, font_size=font_size)
    text_v_margin = int(text_height * 0.3)

    img = np.ones((height, width, 3)) * 0.1
    text_drawer = fc.TextDrawer(
        img, text=text,
        pos=(text_h_margin, height - text_height - text_v_margin),
        font_color=(0.8, 0.8, 0.8), font_size=font_size, font_path=font_path)
    text_drawer.draw()

    height = height - text_height - 2 * text_v_margin
    h_block_width = width / hue_num
    chroma_num = int(round(height / h_block_width + 0.5))
    h_block_size = equal_devision(width, hue_num)
    v_block_size = equal_devision(height, chroma_num)
    mark_size = max(v_block_size[0] // 10, 5)
    mark_img_2020 = np.zeros((mark_size, mark_size, 3))
    # mark_img_p3 = np.ones((mark_size, mark_size, 3)) * 0.5

    h_buf = []
    for h_idx, h_val in enumerate(hue):
        # calc LCH value in specific HUE
        lch_array = _calc_l_focal_to_cups_lch_array(
            inner_lut=inner_lut, outer_lut=outer_lut,
            h_val=h_val, chroma_num=chroma_num, l_focal_max=l_focal_max)
        lab_array = LCHab_to_Lab(lch_array)
        p3_idx = cgbl.is_outer_gamut(lab=lab_array, color_space_name=cs.BT709)
        bt2020_idx = cgbl.is_outer_gamut(
            lab=lab_array, color_space_name=cs.P3_D65)
        v_buf = []
        for c_idx, lab, in enumerate(lab_array):
            rgb_linear = cs.lab_to_rgb(lab, cs.BT2020)
            rgb = tf.oetf(np.clip(rgb_linear, 0.0, 1.0), tf.GAMMA24)
            img_temp = np.ones((v_block_size[c_idx], h_block_size[h_idx], 3))\
                * rgb
            if p3_idx[c_idx]:
                img_temp_p3 = mark_img_2020.copy()
                img_temp_p3[1:-1, 1:-1]\
                    = np.ones_like(img_temp_p3[1:-1, 1:-1]) * rgb
                merge(img_temp, img_temp_p3, (0, 0))
            if bt2020_idx[c_idx]:
                merge(img_temp, mark_img_2020, (0, 0))
            v_buf.append(img_temp)

        h_buf.append(np.vstack(v_buf))
    img_pat = np.hstack(h_buf)
    merge(img, img_pat, (0, 0))

    return img


def make_bt2020_dci_p3_hue_chroma_pattern(
        inner_lut, outer_lut, hue_num, width, height,
        l_focal_max=90, l_focal_min=50):
    """
    Parameters
    ----------
    inner_lut : ndarray
        A inner gamut boundary lut. shape is (N, M, 3).
        N is the number of the Lightness.
        M is the number of the Hue.
    outer_lut : ndarray
        A inner gamut boundary lut. shape is (N, M, 3).
        N is the number of the Lightness.
        M is the number of the Hue.
    width : int
        image width
    height : int
        image height
    hue_num : int
        the number of the hue block.
    l_focal_max : float
        An maximum value for l_focal.
        This is a parameter to prevent the data from changing
        from l_focal to cups transitioning to Out-of-Gamut.
        https://twitter.com/toru_ver15/status/1394645785929666561
    l_focal_min : float
        An minimum value for l_focal.
        This is a parameter to prevent the data from changing
        from l_focal to cups transitioning to Out-of-Gamut.
        https://twitter.com/toru_ver15/status/1394645785929666561
    """
    height_org = height
    font_size = int(20 * height / 1080)
    text_h_margin = int(6 * height / 1080)
    font_path = "/usr/share/fonts/opentype/noto/NotoSansCJKjp-Regular.otf"
    text = '"BT.2020 - DCI-P3 Hue-Chroma Pattern",   '
    text += "Gamma 2.4,   BT.2020,   D65,   Revision 1   "
    # text += "Copyright (C) 2021 - Toru Yoshihara,   "
    # text += "https://trev16.hatenablog.com/"
    hue = np.linspace(0, 360, hue_num, endpoint=False)
    text_width, text_height = fc.get_text_width_height(
        text=text, font_path=font_path, font_size=font_size)
    text_v_margin = int(text_height * 0.3)

    img = np.ones((height, width, 3)) * 0.1
    text_drawer = fc.TextDrawer(
        img, text=text,
        pos=(text_h_margin, height - text_height - text_v_margin),
        font_color=(0.8, 0.8, 0.8), font_size=font_size, font_path=font_path)
    text_drawer.draw()

    height = height - text_height - 2 * text_v_margin
    h_block_width = width / hue_num
    chroma_num = int(round(height / h_block_width + 0.5))
    h_block_size = equal_devision(width, hue_num)
    v_block_size = equal_devision(height, chroma_num)
    mark_size = max(v_block_size[0] // 10, 5)
    mark_img_2020 = np.zeros((mark_size, mark_size, 3))
    # mark_img_p3 = np.ones((mark_size, mark_size, 3)) * 0.5

    h_buf = []
    for h_idx, h_val in enumerate(hue):
        # calc LCH value in specific HUE
        lch_array = _calc_l_focal_to_cups_lch_array(
            inner_lut=inner_lut, outer_lut=outer_lut,
            h_val=h_val, chroma_num=chroma_num, l_focal_max=l_focal_max)
        lab_array = LCHab_to_Lab(lch_array)
        bt2020_idx = cgbl.is_outer_gamut(
            lab=lab_array, color_space_name=cs.P3_D65)
        v_buf = []
        for c_idx, lab, in enumerate(lab_array):
            rgb_linear = cs.lab_to_rgb(lab, cs.BT2020)
            rgb = tf.oetf(np.clip(rgb_linear, 0.0, 1.0), tf.GAMMA24)
            img_temp = np.ones((v_block_size[c_idx], h_block_size[h_idx], 3))\
                * rgb
            if bt2020_idx[c_idx]:
                merge(img_temp, mark_img_2020, (0, 0))
            v_buf.append(img_temp)

        h_buf.append(np.vstack(v_buf))
    img_pat = np.hstack(h_buf)
    merge(img, img_pat, (0, 0))

    return img


def make_bt2020_bt709_hue_chroma_pattern_jzazbz(
        focal_lut, outer_lut, hue_num, width, height, luminance,
        oetf=tf.ST2084):
    """
    Parameters
    ----------
    inner_lut : ndarray
        A inner gamut boundary lut. shape is (N, M, 3).
        N is the number of the Lightness.
        M is the number of the Hue.
    outer_lut : TyLchLut
        A inner gamut boundary lut. shape is (N, M, 3).
        N is the number of the Lightness.
        M is the number of the Hue.
    width : int
        image width
    height : int
        image height
    hue_num : int
        the number of the hue block.
    """
    font_size = int(20 * height / 1080)
    text_h_margin = int(6 * height / 1080)
    font_path = "/usr/share/fonts/opentype/noto/NotoSansCJKjp-Regular.otf"
    text = '"BT.2020 - DCI-P3 - BT.709 Hue-Chroma Pattern",   '
    text += f"{oetf},   BT.2020,   D65,   {luminance}nits,   Revision 3   "
    # text += "Copyright (C) 2021 - Toru Yoshihara,   "
    # text += "https://trev16.hatenablog.com/"
    hue = np.linspace(0, 360, hue_num, endpoint=False)
    text_width, text_height = fc.get_text_width_height(
        text=text, font_path=font_path, font_size=font_size)
    text_v_margin = int(text_height * 0.3)

    img = np.ones((height, width, 3)) * 0.1
    text_drawer = fc.TextDrawer(
        img, text=text,
        pos=(text_h_margin, height - text_height - text_v_margin),
        font_color=(0.8, 0.8, 0.8), font_size=font_size, font_path=font_path)
    text_drawer.draw()

    height = height - text_height - 2 * text_v_margin
    h_block_width = width / hue_num
    chroma_num = int(round(height / h_block_width + 0.5))
    h_block_size = equal_devision(width, hue_num)
    v_block_size = equal_devision(height, chroma_num)
    mark_size = max(v_block_size[0] // 10, 5)
    mark_img_2020 = np.zeros((mark_size, mark_size, 3))
    # mark_img_p3 = np.ones((mark_size, mark_size, 3)) * 0.5

    h_buf = []
    for h_idx, h_val in enumerate(hue):
        # calc LCH value in specific HUE
        lch_array = _calc_l_focal_to_cups_lch_array_jzazbz(
            focal_lut=focal_lut, outer_lut=outer_lut, h_val=h_val,
            chroma_num=chroma_num)
        # print(lch_array[-4:])
        jzazbz = jzczhz_to_jzazbz(lch_array)
        rgb_linear = cs.jzazbz_to_rgb(
            jzazbz=jzazbz, color_space_name=cs.BT2020, luminance=luminance)
        # print(rgb_linear[-4:])
        rgb = tf.oetf_from_luminance(
            np.clip(rgb_linear, 0.0, 1.0) * luminance, oetf)
        # print(rgb[-4:])
        p3_idx = cgbl.is_outer_gamut_jzazbz(
            jzazbz=jzazbz, color_space_name=cs.BT709, luminance=luminance)
        bt2020_idx = cgbl.is_outer_gamut_jzazbz(
            jzazbz=jzazbz, color_space_name=cs.P3_D65, luminance=luminance)
        v_buf = []
        for c_idx in range(len(lch_array)):
            # rgb_linear = cs.lab_to_rgb(lab, cs.BT2020)
            img_temp = np.ones((v_block_size[c_idx], h_block_size[h_idx], 3))\
                * rgb[c_idx]
            if p3_idx[c_idx]:
                img_temp_p3 = mark_img_2020.copy()
                img_temp_p3[1:-1, 1:-1]\
                    = np.ones_like(img_temp_p3[1:-1, 1:-1]) * rgb[c_idx]
                merge(img_temp, img_temp_p3, (0, 0))
            if bt2020_idx[c_idx]:
                merge(img_temp, mark_img_2020, (0, 0))
            v_buf.append(img_temp)

        h_buf.append(np.vstack(v_buf))
    img_pat = np.hstack(h_buf)
    merge(img, img_pat, (0, 0))

    return img


def make_p3d65_bt709_hue_chroma_pattern_jzazbz(
        focal_lut, outer_lut, hue_num, width, height, luminance,
        oetf=tf.ST2084):
    """
    Parameters
    ----------
    inner_lut : ndarray
        A inner gamut boundary lut. shape is (N, M, 3).
        N is the number of the Lightness.
        M is the number of the Hue.
    outer_lut : TyLchLut
        A inner gamut boundary lut. shape is (N, M, 3).
        N is the number of the Lightness.
        M is the number of the Hue.
    width : int
        image width
    height : int
        image height
    hue_num : int
        the number of the hue block.
    """
    font_size = int(20 * height / 1080)
    text_h_margin = int(6 * height / 1080)
    font_path = "/usr/share/fonts/opentype/noto/NotoSansCJKjp-Regular.otf"
    text = '"DCI-P3 - BT.709 Hue-Chroma Pattern",   '
    text += f"{oetf},   DCI-P3,   D65,   {luminance}nits,   Revision 3,   "
    # text += "Copyright (C) 2021 - Toru Yoshihara,   "
    # text += "https://trev16.hatenablog.com/"
    hue = np.linspace(0, 360, hue_num, endpoint=False)
    text_width, text_height = fc.get_text_width_height(
        text=text, font_path=font_path, font_size=font_size)
    text_v_margin = int(text_height * 0.3)

    img = np.ones((height, width, 3)) * 0.1
    text_drawer = fc.TextDrawer(
        img, text=text,
        pos=(text_h_margin, height - text_height - text_v_margin),
        font_color=(0.8, 0.8, 0.8), font_size=font_size, font_path=font_path)
    text_drawer.draw()

    height = height - text_height - 2 * text_v_margin
    h_block_width = width / hue_num
    chroma_num = int(round(height / h_block_width + 0.5))
    h_block_size = equal_devision(width, hue_num)
    v_block_size = equal_devision(height, chroma_num)
    mark_size = max(v_block_size[0] // 10, 5)
    mark_img_2020 = np.zeros((mark_size, mark_size, 3))
    # mark_img_p3 = np.ones((mark_size, mark_size, 3)) * 0.5

    h_buf = []
    for h_idx, h_val in enumerate(hue):
        # calc LCH value in specific HUE
        lch_array = _calc_l_focal_to_cups_lch_array_jzazbz(
            focal_lut=focal_lut, outer_lut=outer_lut, h_val=h_val,
            chroma_num=chroma_num)
        # print(lch_array[-4:])
        jzazbz = jzczhz_to_jzazbz(lch_array)
        rgb_linear = cs.jzazbz_to_rgb(
            jzazbz=jzazbz, color_space_name=cs.P3_D65, luminance=luminance)
        # print(rgb_linear[-4:])
        rgb = tf.oetf_from_luminance(
            np.clip(rgb_linear, 0.0, 1.0) * luminance, oetf)
        # print(rgb[-4:])
        p3_idx = cgbl.is_outer_gamut_jzazbz(
            jzazbz=jzazbz, color_space_name=cs.BT709, luminance=luminance)
        v_buf = []
        for c_idx in range(len(lch_array)):
            # rgb_linear = cs.lab_to_rgb(lab, cs.BT2020)
            img_temp = np.ones((v_block_size[c_idx], h_block_size[h_idx], 3))\
                * rgb[c_idx]
            if p3_idx[c_idx]:
                img_temp_p3 = mark_img_2020.copy()
                img_temp_p3[1:-1, 1:-1]\
                    = np.ones_like(img_temp_p3[1:-1, 1:-1]) * rgb[c_idx]
                merge(img_temp, img_temp_p3, (0, 0))
            v_buf.append(img_temp)

        h_buf.append(np.vstack(v_buf))
    img_pat = np.hstack(h_buf)
    merge(img, img_pat, (0, 0))

    return img


def make_bt709_hue_chroma_pattern_jzazbz(
        focal_lut, outer_lut, hue_num, width, height, luminance):
    """
    Parameters
    ----------
    inner_lut : ndarray
        A inner gamut boundary lut. shape is (N, M, 3).
        N is the number of the Lightness.
        M is the number of the Hue.
    outer_lut : ndarray
        A inner gamut boundary lut. shape is (N, M, 3).
        N is the number of the Lightness.
        M is the number of the Hue.
    width : int
        image width
    height : int
        image height
    hue_num : int
        the number of the hue block.
    """
    font_size = int(20 * height / 1080)
    text_h_margin = int(6 * height / 1080)
    font_path = "/usr/share/fonts/opentype/noto/NotoSansCJKjp-Regular.otf"
    text = '"BT.709 Hue-Chroma Pattern",   '
    text += "Gamma 2.4,   BT.709,   D65,   Revision 1,   "
    # text += "Copyright (C) 2021 - Toru Yoshihara,   "
    # text += "https://trev16.hatenablog.com/"
    hue = np.linspace(0, 360, hue_num, endpoint=False)
    _, text_height = fc.get_text_width_height(
        text=text, font_path=font_path, font_size=font_size)
    text_v_margin = int(text_height * 0.3)

    img = np.ones((height, width, 3)) * 0.1
    text_drawer = fc.TextDrawer(
        img, text=text,
        pos=(text_h_margin, height - text_height - text_v_margin),
        font_color=(0.8, 0.8, 0.8), font_size=font_size, font_path=font_path)
    text_drawer.draw()

    height = height - text_height - 2 * text_v_margin
    h_block_width = width / hue_num
    chroma_num = int(round(height / h_block_width + 0.5))
    h_block_size = equal_devision(width, hue_num)
    v_block_size = equal_devision(height, chroma_num)

    h_buf = []
    for h_idx, h_val in enumerate(hue):
        # calc LCH value in specific HUE
        lch_array = _calc_l_focal_to_cups_lch_array_jzazbz(
            focal_lut=focal_lut, outer_lut=outer_lut, h_val=h_val,
            chroma_num=chroma_num)
        # print(lch_array[-4:])
        jzazbz = jzczhz_to_jzazbz(lch_array)
        rgb_linear = cs.jzazbz_to_rgb(
            jzazbz=jzazbz, color_space_name=cs.BT709, luminance=luminance)
        rgb = tf.oetf(np.clip(rgb_linear, 0.0, 1.0), tf.GAMMA24)
        v_buf = []
        for c_idx in range(len(lch_array)):
            # rgb_linear = cs.lab_to_rgb(lab, cs.BT2020)
            img_temp = np.ones((v_block_size[c_idx], h_block_size[h_idx], 3))\
                * rgb[c_idx]
            v_buf.append(img_temp)

        h_buf.append(np.vstack(v_buf))
    img_pat = np.hstack(h_buf)
    merge(img, img_pat, (0, 0))

    return img


def complex_dot_pattern2(
        nn=3,
        fg_color=np.array([1.0, 0.5, 0.3]),
        bg_color=np.array([0.1, 0.1, 0.1]),
        bg_color_alpha=0.0,
        mag_rate=4):
    """
    https://github.com/toru-ver4/sample_code/issues/182#issuecomment-1105143061

    Parameters
    ----------
    nn : int
        factor N !!!!
    fg_color : ndarray
        color value. It must be linear.
    bg_color : ndarray
        color value. It must be linear.
    bg_color_alpha : ndarray
        T.B.D
    mag_rate : int
        expantion rate

    Returns
    -------
    ndarray :
        linear image data.
    """
    def recuresive(nn, fg_color, bg_color, bg_color_alpha):
        if nn < 1:
            img = np.ones((1, 1, 3)) * fg_color
        elif nn == 1:
            img = np.ones((2, 2, 3)) * fg_color
            img[0, 0] = bg_color
            img[1, 1] = bg_color
        else:
            size = 2 ** nn
            div4 = size // 4

            pt1 = (div4 * 0, div4 * 0)
            pt2 = (div4 * 3, div4 * 0)
            pt3 = (div4 * 2, div4 * 1)
            pt4 = (div4 * 4, div4 * 1)
            pt5 = (div4 * 1, div4 * 2)
            pt6 = (div4 * 2, div4 * 2)
            pt7 = (div4 * 3, div4 * 2)
            pt8 = (div4 * 0, div4 * 3)
            pt9 = (div4 * 2, div4 * 3)
            pt10 = (div4 * 1, div4 * 4)
            pt11 = (div4 * 4, div4 * 4)

            img = np.ones((size, size, 3)) * bg_color
            img_n1 = recuresive(
                nn=nn-1, fg_color=fg_color, bg_color=bg_color,
                bg_color_alpha=bg_color_alpha)
            img_n2 = recuresive(
                nn=nn-2, fg_color=fg_color, bg_color=bg_color,
                bg_color_alpha=bg_color_alpha)

            img[pt1[1]:pt6[1], pt1[0]:pt6[0]] = fg_color
            img[pt2[1]:pt4[1], pt2[0]:pt4[0]] = img_n2[:, ::-1, :]
            img[pt8[1]:pt10[1], pt8[0]:pt10[0]] = img_n2[::-1, :, :]
            img[pt3[1]:pt7[1], pt3[0]:pt7[0]] = img_n2[:, ::-1, :]
            img[pt5[1]:pt9[1], pt5[0]:pt9[0]] = img_n2[::-1, :, :]
            img[pt6[1]:pt11[1], pt6[0]:pt11[0]] = img_n1
        return img

    img = recuresive(
        nn=nn, fg_color=fg_color, bg_color=bg_color,
        bg_color_alpha=bg_color_alpha)
    out_img = cv2.resize(
            img, None, fx=mag_rate, fy=mag_rate,
            interpolation=cv2.INTER_NEAREST)

    return out_img


def line_cross_pattern(nn, num_of_min_line, fg_color, bg_color, mag_rate=1):
    """
    https://github.com/toru-ver4/sample_code/issues/182#issuecomment-1105256110

    Parameters
    ----------
    nn : int
        factor N
    num_of_min_line : int
        minimum line number
    fg_color : ndarray
        color value. It must be linear.
    bg_color : ndarray
        color value. It must be linear.
    mag_rate : int
        magnitude rate

    Returns
    -------
    ndarray :
        linear image data.
    """
    max_thickness = 2 ** (nn - 1)
    block_len = max_thickness * num_of_min_line * 2
    size = block_len * nn
    print(f"block_len={block_len}, size={size}")
    img = np.ones((size, size, 3)) * bg_color

    for n_idx in range(nn):
        thickness = max_thickness // (2 ** n_idx)
        g_st_pos = [0, block_len * n_idx]
        num_of_line = num_of_min_line * (2 ** n_idx)
        for l_idx in range(num_of_line):
            st_pos = [0, g_st_pos[1] + thickness * 2 * l_idx]
            # print(f"l_idx={l_idx}, st_pos={st_pos}, thick={thickness}")
            draw_hv_straight_line(
                img=img, st_pos=st_pos, width=size, thickness=thickness,
                color=fg_color, direction='h')
            draw_hv_straight_line(
                img=img, st_pos=st_pos, width=size, thickness=thickness,
                color=fg_color, direction='v')

    img = tf.oetf(img, tf.GAMMA24)
    out_img = cv2.resize(
            img, None, fx=mag_rate, fy=mag_rate,
            interpolation=cv2.INTER_NEAREST)
    # fname = f"./img/line_cross_nn-{nn}_nol-{num_of_min_line}.png"
    # write_image(out_img, fname)

    return out_img


def draw_hv_straight_line(img, st_pos, width, thickness, color, direction='h'):
    if direction == 'h':
        st_pos2 = st_pos
        ed_pos = [st_pos[0] + width, st_pos[1] + thickness]
        # print(f"{st_pos2}, {ed_pos}")
    elif direction == 'v':
        st_pos2 = [st_pos[1], st_pos[0]]
        ed_pos = [st_pos2[0] + thickness, st_pos2[1] + width]
        # print(f"{st_pos2}, {ed_pos}")
    else:
        raise ValueError("invalid direction")

    draw_rectangle(img, st_pos2, ed_pos, color)


def draw_rectangle(img, st_pos, ed_pos, color):
    img[st_pos[1]:ed_pos[1], st_pos[0]:ed_pos[0]] = color


def draw_border_line(img, st_pos, length, thickness, color):
    st = st_pos
    pt1 = [st[0], st[1]]
    pt2 = [st[0]+length-thickness, st[1]]
    pt3 = [st[0]+length, st[1]+thickness]
    pt4 = [st[0], st[1]+length-thickness]
    pt5 = [st[0]+thickness, st[1]+length]
    pt6 = [st[0]+length, st[1]+length]

    draw_rectangle(img=img, st_pos=pt1, ed_pos=pt3, color=color)
    draw_rectangle(img=img, st_pos=pt1, ed_pos=pt5, color=color)
    draw_rectangle(img=img, st_pos=pt2, ed_pos=pt6, color=color)
    draw_rectangle(img=img, st_pos=pt4, ed_pos=pt6, color=color)


def calc_thickness_for_block(block_idx, num_of_block):
    bb = block_idx
    thickness = 2 ** (num_of_block - 1 - bb)

    return thickness


def calc_l_for_block(block_idx, num_of_block, num_of_line):
    bb = block_idx
    thickness = calc_thickness_for_block(block_idx, num_of_block)

    if bb >= (num_of_block - 1):
        ll = thickness * num_of_line * 2 * 2 - 1
    else:
        ll = thickness * num_of_line * 2 * 2\
            + calc_l_for_block(
                block_idx=bb+1, num_of_block=num_of_block,
                num_of_line=num_of_line)

    return ll


def create_multi_border_tp(
        num_of_line, num_of_block, fg_color, bg_color, mag_rate=1):
    """
    num_of_line : int
        number of the line
    num_of_block : int
        number of the block
    fg_color : ndarray
        color value. It must be linear.
    bg_color : ndarray
        color value. It must be linear.
    mag_rate : int
        magnitude rate

    Returns
    -------
    ndarray :
        linear image data
    """
    g_st_pos = [0, 0]
    tp_size = calc_l_for_block(
        block_idx=0, num_of_block=num_of_block, num_of_line=num_of_line)
    img = np.ones((tp_size, tp_size, 3)) * bg_color
    for b_idx in range(num_of_block):
        thickness = calc_thickness_for_block(b_idx, num_of_block)
        lb = calc_l_for_block(
            block_idx=b_idx, num_of_block=num_of_block,
            num_of_line=num_of_line)
        for l_idx in range(num_of_line):
            length = lb - thickness * 2 * 2 * l_idx
            st_pos_h = g_st_pos[0] + thickness * 2 * l_idx
            st_pos_v = st_pos_h
            st_pos = [st_pos_h, st_pos_v]
            draw_border_line(
                img=img, st_pos=st_pos, length=length,
                thickness=thickness, color=fg_color)
        g_st_pos[0] = g_st_pos[0] + thickness * 2 * num_of_line
        g_st_pos[1] = g_st_pos[0]

    out_img = cv2.resize(
            img, None, fx=mag_rate, fy=mag_rate,
            interpolation=cv2.INTER_NEAREST)

    # write_image(out_img, "./img/multi_border_tp_test.png")

    return out_img


def create_dot_mesh_image(
        width=640, height=480, dot_size=[4, 2], st_offset=[2, 1]):
    """
    Parameters
    ----------
    width : int
        width
    height : int
        height
    dot_size : list(int)
        dot size. [x_size, y_size]
    st_offset : list(int)
        offset. [x_offset, y_offset]

    Examples
    --------
    >>> img = create_dot_mesh_image(
    ...     width=12, height=8, dot_size=[4, 2], st_offset=[2, 1])
    >>> print(img[..., 0].reshape(8, 12))
    [[ 1.  1.  0.  0.  0.  0.  1.  1.  1.  1.  0.  0.]
     [ 0.  0.  1.  1.  1.  1.  0.  0.  0.  0.  1.  1.]
     [ 0.  0.  1.  1.  1.  1.  0.  0.  0.  0.  1.  1.]
     [ 1.  1.  0.  0.  0.  0.  1.  1.  1.  1.  0.  0.]
     [ 1.  1.  0.  0.  0.  0.  1.  1.  1.  1.  0.  0.]
     [ 0.  0.  1.  1.  1.  1.  0.  0.  0.  0.  1.  1.]
     [ 0.  0.  1.  1.  1.  1.  0.  0.  0.  0.  1.  1.]
     [ 1.  1.  0.  0.  0.  0.  1.  1.  1.  1.  0.  0.]]
    """
    dot_size_h = dot_size[0]
    dot_size_v = dot_size[1]
    offset_h = st_offset[0]
    offset_v = st_offset[1]
    h_ok_idx = (np.arange(width) - offset_h) % (dot_size_h * 2) < dot_size_h
    v_ok_idx = (np.arange(height) - offset_v) % (dot_size_v * 2) < dot_size_v
    h_ng_idx = (np.arange(width) - offset_h) % (dot_size_h * 2) >= dot_size_h
    v_ng_idx = (np.arange(height) - offset_v) % (dot_size_v * 2) >= dot_size_v
    base_img = np.zeros((height, width))
    base_img_ok_v = base_img.copy()
    base_img_ok_v[v_ok_idx] = 1
    base_img_ok_h = base_img.copy()
    base_img_ok_h[:, h_ok_idx] = 1
    base_img_ng_v = base_img.copy()
    base_img_ng_v[v_ng_idx] = 1
    base_img_ng_h = base_img.copy()
    base_img_ng_h[:, h_ng_idx] = 1
    img_ok = base_img_ok_v * base_img_ok_h
    img_ng = base_img_ng_v * base_img_ng_h
    img_mono = img_ok + img_ng

    img = np.dstack([img_mono, img_mono, img_mono])

    return img


def scroll_image(img, offset_h, offset_v):
    """
    Scroll the input image based on the offset parameters.

    Parameters
    ----------
    img : ndarray
        A image data.
    offset_h : int
        A horizontal offset
    offset_v : int
        A vertical offset

    Returns
    -------
    ndarray
        A scrolled image data
    """
    height, width = img.shape[:2]
    out_img = np.zeros_like(img)
    pos_h = offset_h % width
    pos_v = offset_v % height

    pt1_h = 0
    pt1_v = 0
    pt2_h = pos_h
    pt2_v = 0
    pt3_h = pos_h
    pt3_v = pos_v
    pt4_h = width
    pt4_v = pos_v
    pt5_h = 0
    pt5_v = pos_v
    pt6_h = pos_h
    pt6_v = height
    pt7_h = width
    pt7_v = height

    out_img[:height-pos_v, :width-pos_h] = img[pt3_v:pt7_v, pt3_h:pt7_h]
    out_img[:height-pos_v, width-pos_h:] = img[pt5_v:pt6_v, pt5_h:pt6_h]
    out_img[height-pos_v:, :width-pos_h] = img[pt2_v:pt4_v, pt2_h:pt4_h]
    out_img[height-pos_v:, width-pos_h:] = img[pt1_v:pt3_v, pt1_h:pt3_h]

    return out_img


def create_wrgbmyc_ramp_data(bit_depth=10):
    """
    Examples
    --------
    >>> rgb = create_test_rgb_data(bit_depth=10)
    >>> print(rgb)
    [[[   0    0    0]
      [   1    1    1]
      [   2    2    2]
      ...,
      [1021 1021 1021]
      [1022 1022 1022]
      [1023 1023 1023]]

     [[   0    0    0]
      [   1    0    0]
      [   2    0    0]
      ...,
      [1021    0    0]
      [1022    0    0]
      [1023    0    0]]

     [[   0    0    0]
      [   0    1    0]
      [   0    2    0]
      ...,
      [   0 1021    0]
      [   0 1022    0]
      [   0 1023    0]]

     ...,
     [[   0    0    0]
      [   1    0    1]
      [   2    0    2]
      ...,
      [1021    0 1021]
      [1022    0 1022]
      [1023    0 1023]]

     [[   0    0    0]
      [   1    1    0]
      [   2    2    0]
      ...,
      [1021 1021    0]
      [1022 1022    0]
      [1023 1023    0]]

     [[   0    0    0]
      [   0    1    1]
      [   0    2    2]
      ...,
      [   0 1021 1021]
      [   0 1022 1022]
      [   0 1023 1023]]]
    """
    num_of_cv = 2 ** bit_depth
    gradient = np.arange(num_of_cv, dtype=np.uint16)
    color_mask_list = np.array([
        [1, 1, 1],
        [1, 0, 0], [0, 1, 0], [0, 0, 1],
        [1, 0, 1], [1, 1, 0], [0, 1, 1]
    ], dtype=np.uint16)
    rgb = gradient.reshape(-1, num_of_cv, 1).repeat(3, axis=2)\
        .repeat(color_mask_list.shape[0], axis=0)

    rgb = rgb * color_mask_list.reshape(-1, 1, 3)

    return rgb


def calc_rgb_2_Y(rgb_linear, color_space_name):
    """
    Parameters
    ----------
    rgb_linear : ndarray
        linear rgb data. the unit is nits
    color_space_name : str
        color space name. e.g. `cs.BT709`
    """
    large_xyz = cs.rgb_to_large_xyz(
        rgb=rgb_linear, color_space_name=color_space_name)
    return large_xyz[..., 1]


def calc_max_cll(rgb_linear, color_space_name):
    """
    Parameters
    ----------
    rgb_linear : ndarray
        linear rgb data. the unit is nits
    color_space_name : str
        color space name. e.g. `cs.BT709`
    """
    yy = calc_rgb_2_Y(
        rgb_linear=rgb_linear, color_space_name=color_space_name)
    return int(np.round(np.max(yy)))


def calc_max_fall(rgb_linear, color_space_name):
    """
    Parameters
    ----------
    rgb_linear : ndarray
        linear rgb data. the unit is nits
    color_space_name : str
        color space name. e.g. `cs.BT709`
    """
    yy = calc_rgb_2_Y(
        rgb_linear=rgb_linear, color_space_name=color_space_name)
    return int(np.round(np.average(yy.flatten())))


def png_to_avif(
        png_fname, avif_fname,
        bit_depth=10,
        color_space_name=cs.BT2020,
        transfer_characteristics=tf.ST2084,
        cll=None, pall=None):

    if (cll is None) or (pall is None):
        rgb_non_linear = img_read_as_float(png_fname)
        rgb_linear = tf.eotf_to_luminance(
            rgb_non_linear, transfer_characteristics
        )
        if cll is None:
            cll = calc_max_cll(
                rgb_linear=rgb_linear, color_space_name=color_space_name
            )
        if pall is None:
            pall = calc_max_fall(
                rgb_linear=rgb_linear, color_space_name=color_space_name
            )

    if color_space_name == cs.BT2020:
        cicp_cs = "9"
    elif color_space_name == cs.P3_D65:
        cicp_cs = "12"
    elif color_space_name == cs.BT709:
        cicp_cs = "1"
    else:
        raise ValueError("Error. unknown color space name.")

    if transfer_characteristics == tf.ST2084:
        cicp_tf = "16"
    elif transfer_characteristics == tf.HLG:
        cicp_tf = "18"
    elif transfer_characteristics == tf.SRGB:
        cicp_tf = "13"
    else:
        cicp_tf = "1"

    cmd = [
        "avifenc", png_fname,
        "-d", f"{bit_depth}",
        "-y", "444",
        "--cicp", f"{cicp_cs}/{cicp_tf}/0",
        "-r", "full",
        "--clli", f"{cll},{pall}",
        "--lossless",
        "--ignore-exif",
        avif_fname
    ]
    print(" ".join(cmd))
    subprocess.run(cmd)


def png_to_heif(
        png_fname,
        heif_fname,
        bit_depth=10,
        color_space_name=cs.BT2020,
        transfer_characteristics=tf.ST2084,
):
    if color_space_name == cs.BT2020:
        cicp_cs = "9"
        # cicp_mtx = "9"
    elif color_space_name == cs.P3_D65:
        cicp_cs = "12"
        # cicp_mtx = "1"
    elif color_space_name == cs.BT709:
        cicp_cs = "1"
        # cicp_mtx = "1"
    else:
        raise ValueError("Error. unknown color space name.")

    if transfer_characteristics == tf.ST2084:
        cicp_tf = "16"
    elif transfer_characteristics == tf.HLG:
        cicp_tf = "18"
    elif transfer_characteristics == tf.SRGB:
        cicp_tf = "13"
    else:
        cicp_tf = "1"

    cmd = [
        "heif-enc",
        "--quality", "100",
        "--bit-depth", f"{bit_depth}",
        "-p", "chroma=444",
        "--colour_primaries", cicp_cs,
        "--transfer_characteristic", cicp_tf,
        "--matrix_coefficients", "0",
        "--full_range_flag", "1",
        png_fname,
        '-o', heif_fname
    ]
    print(" ".join(cmd))
    subprocess.run(cmd)


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    # print(calc_rad_patch_idx(outmost_num=9, current_num=1))
    # _plot_same_lstar_radial_color_patch_data(
    #     lstar=58, chroma=32.5, outmost_num=7,
    #     color_space=RGB_COLOURSPACE_BT709,
    #     transfer_function=tf.GAMMA24)
    # calc_rad_patch_idx2(outmost_num=9, current_num=7)
    # print(convert_luminance_to_color_value(100, tf.ST2084))
    # print(generate_color_checker_rgb_value(target_white=[0.3127, 0.3290]))
    # print(calc_st_pos_for_centering(bg_size=(1920, 1080), fg_size=(640, 480)))
    # print(convert_luminance_to_code_value(100, tf.ST2084))
    # make_hue_chroma_pattern(
    #     inner_lut=np.load("/work/src/2021/09_gamut_boundary_lut/lut/lut_sample_1024_1024_32768_ITU-R BT.709.npy"),
    #     outer_lut=np.load("/work/src/2021/09_gamut_boundary_lut/lut/lut_sample_1024_1024_32768_ITU-R BT.2020.npy"),
    #     width=2048, height=1180, hue_num=32)

    # line = np.linspace(0, 1, 5)
    # line_color = tstack([line, line, line])
    # print(line)
    # img = v_mono_line_to_img(line, 4)
    # print(img)

    # line_r = np.linspace(0, 4, 5)
    # line_g = np.linspace(0, 4, 5) * 2
    # line_b = np.linspace(0, 4, 5) * 3
    # line_color = tstack([line_r, line_g, line_b])
    # print(line_color)
    # img = v_color_line_to_img(line_color, 4)
    # print(img)

    # bg_img = np.ones((1080, 1920, 3)) * 0.5
    # fg_img = np.zeros((540, 960, 3), dtype=np.uint8)
    # fg_img = cv2.circle(
    #     fg_img, (200, 100), 40,
    #     color=[0, 192, 192], thickness=-1, lineType=cv2.LINE_AA)
    #     # rmo_list[idx].calc_next_pos()  # マルチスレッド化にともない事前に計算
    # # alpha channel は正規化する。そうしないと中間調合成時に透けてしまう
    # alpha = np.max(fg_img, axis=-1)
    # alpha = alpha / np.max(alpha)
    # fg_img = np.dstack((fg_img / 0xFF, alpha))
    # merge_with_alpha2(bg_img=bg_img, fg_img=fg_img, pos=(200, 100))
    # img_wirte_float_as_16bit_int("fg.png", fg_img)
    # img_wirte_float_as_16bit_int("after_merge.png", bg_img)

    # line = np.linspace(0, 1, 9)
    # print(line)
    # img = h_mono_line_to_img(line, 6)
    # print(img)
