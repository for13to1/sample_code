#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
# Color Space モジュール

## 概要
Primaries および White Point の管理。
以下の情報を取れたりする。

* White Point
* Primaries

## 設計思想
In-Out は原則 [0:1] のレンジで行う。
別途、Global変数として最大輝度をパラメータとして持ち、
輝度⇔信号レベルの相互変換はその変数を使って行う。
"""

import os
import numpy as np
from colour.colorimetry import CCS_ILLUMINANTS as ILLUMINANTS
from colour import RGB_COLOURSPACES
from colour.models import xy_to_XYZ, Jab_to_JCh, JCh_to_Jab
from colour import xy_to_xyY, xyY_to_XYZ, XYZ_to_RGB, RGB_to_XYZ, XYZ_to_Lab, \
    Lab_to_XYZ, Oklab_to_XYZ, XYZ_to_Oklab
from colour.adaptation import matrix_chromatic_adaptation_VonKries as cat02_mtx
from colour.utilities import tstack
from scipy import linalg
from jzazbz import jzazbz_to_large_xyz, large_xyz_to_jzazbz

# Define
CMFS_NAME = 'CIE 1931 2 Degree Standard Observer'
D65 = ILLUMINANTS[CMFS_NAME]['D65']
D50 = ILLUMINANTS[CMFS_NAME]['D50']
D60_ACES = np.array([0.32168, 0.33767])

D65_XYZ = xyY_to_XYZ(xy_to_xyY(D65))
D50_XYZ = xyY_to_XYZ(xy_to_xyY(D50))
D60_ACES_XYZ = xyY_to_XYZ(xy_to_xyY(D60_ACES))

# NAME
BT709 = 'ITU-R BT.709'
BT2020 = 'ITU-R BT.2020'
ACES_AP0 = 'ACES2065-1'
ACES_AP1 = 'ACEScg'
S_GAMUT3 = 'S-Gamut3'
S_GAMUT3_CINE = 'S-Gamut3.Cine'
ALEXA_WIDE_GAMUT = 'ARRI Wide Gamut 3'
ALEXA_WIDE_GAMUT_3 = 'ARRI Wide Gamut 3'
ALEXA_WIDE_GAMUT_4 = "ARRI Wide Gamut 4"
V_GAMUT = 'V-Gamut'
CINEMA_GAMUT = 'Cinema Gamut'
RED_WIDE_GAMUT_RGB = 'REDWideGamutRGB'
DCI_P3 = 'DCI-P3'
SRTB = 'sRGB'
sRGB = "sRGB"
P3_D65 = 'P3-D65'
ADOBE_RGB = 'Adobe RGB (1998)'


def calc_rgb_from_xyY(xyY, color_space_name, white=D65):
    """
    calc rgb from xyY.

    Parameters
    ----------
    xyY : ndarray
        xyY values.
    color_space_name : str
        the name of the target color space.
    white : ndarray
        white point. ex: np.array([0.3127, 0.3290])

    Returns
    -------
    ndarray
        rgb linear value (not clipped, so negative values may be present).

    Examples
    --------
    >>> xyY = np.array(
    ...     [[0.3127, 0.3290, 1.0], [0.64, 0.33, 0.2], [0.30, 0.60, 0.6]])
    >>> calc_rgb_from_xyY(
    ...     xyY=xyY, color_space_name=cs.BT709, white=D65)
    [[  1.00000000e+00   1.00000000e+00   1.00000000e+00]
     [  9.40561207e-01   1.66533454e-16  -1.73472348e-17]
     [ -2.22044605e-16   8.38962916e-01  -6.93889390e-18]]
    """
    rgb = calc_rgb_from_XYZ(
        xyY_to_XYZ(xyY), color_space_name=color_space_name, white=white)

    return rgb


def calc_rgb_from_XYZ(XYZ, color_space_name, white=D65):
    """
    calc rgb from XYZ.

    Parameters
    ----------
    XYZ : ndarray
        XYZ values.
    color_space_name : str
        the name of the target color space.
    white : ndarray
        white point. ex: np.array([0.3127, 0.3290])

    Returns
    -------
    ndarray
        rgb linear value (not clipped, so negative values may be present).

    Examples
    --------
    >>> xyY = np.array(
    ...     [[0.3127, 0.3290, 1.0], [0.64, 0.33, 0.2], [0.30, 0.60, 0.6]])
    >>> calc_rgb_from_XYZ(
    ...     XYZ=xyY_to_XYZ(xyY), color_space_name=cs.BT709, white=D65)
    [[  1.00000000e+00   1.00000000e+00   1.00000000e+00]
     [  9.40561207e-01   1.66533454e-16  -1.73472348e-17]
     [ -2.22044605e-16   8.38962916e-01  -6.93889390e-18]]
    """
    rgb = XYZ_to_RGB(
        XYZ, white, white,
        RGB_COLOURSPACES[color_space_name].XYZ_to_RGB_matrix)

    return rgb


def calc_XYZ_from_rgb(rgb, color_space_name, white=D65):
    """
    calc XYZ from rgb.

    Parameters
    ----------
    rgb : ndarray
        rgb linear values.
    color_space_name : str
        the name of the target color space.
    white : ndarray
        white point. ex: np.array([0.3127, 0.3290])

    Returns
    -------
    ndarray
        rgb linear value (not clipped, so negative values may be present).

    Examples
    --------
    >>> rgb = np.array(
    ...     [[1.0, 1.0, 1.0], [0.18, 0.18, 0.18], [1.0, 0.0, 0.0]])
    >>> XYZ = calc_XYZ_from_rgb(
    ...     rgb=rgb, color_space_name=cs.BT709, white=D65)
    >>> XYZ_to_xyY(XYZ)
    [[ 0.3127      0.329       1.        ]
     [ 0.3127      0.329       0.18      ]
     [ 0.64        0.33        0.21263901]]
    """
    XYZ = RGB_to_XYZ(
        rgb, white, white,
        RGB_COLOURSPACES[color_space_name].RGB_to_XYZ_matrix)

    return XYZ


def split_tristimulus_values(data):
    """
    Examples
    --------
    >>> data = np.array(
    ...     [[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    >>> split_tristimulus_values(data)
    (array([1, 4, 7]), array([2, 5, 8]), array([3, 6, 9]))
    """
    x0 = data[..., 0]
    x1 = data[..., 1]
    x2 = data[..., 2]

    return x0, x1, x2


def xy_to_xyz_internal(xy):
    rz = 1 - (xy[0][0] + xy[0][1])
    gz = 1 - (xy[1][0] + xy[1][1])
    bz = 1 - (xy[2][0] + xy[2][1])

    xyz = [[xy[0][0], xy[0][1], rz],
           [xy[1][0], xy[1][1], gz],
           [xy[2][0], xy[2][1], bz]]

    return xyz


def calc_rgb_to_xyz_matrix(gamut_xy, white_large_xyz):
    """
    RGB2XYZ Matrixを計算する

    Parameters
    ----------
    gamut_xy : ndarray
        gamut. shape should be (3, 2).
    white_large_xyz : ndarray
        large xyz value like [95.047, 100.000, 108.883].

    Returns
    -------
    array_like
        a cms pattern image.

    """

    # まずは xyz 座標を準備
    # ------------------------------------------------
    if np.array(gamut_xy).shape == (3, 2):
        gamut = xy_to_xyz_internal(gamut_xy)
    elif np.array(gamut_xy).shape == (3, 3):
        gamut = gamut_xy.copy()
    else:
        raise ValueError("invalid xy gamut parameter.")

    gamut_mtx = np.array(gamut)

    # 白色点の XYZ を算出。Y=1 となるように調整
    # ------------------------------------------------
    large_xyz = [white_large_xyz[0] / white_large_xyz[1],
                 white_large_xyz[1] / white_large_xyz[1],
                 white_large_xyz[2] / white_large_xyz[1]]
    large_xyz = np.array(large_xyz)

    # Sr, Sg, Sb を算出
    # ------------------------------------------------
    s = linalg.inv(gamut_mtx[0:3]).T.dot(large_xyz)

    # RGB2XYZ 行列を算出
    # ------------------------------------------------
    s_matrix = [[s[0], 0.0,  0.0],
                [0.0,  s[1], 0.0],
                [0.0,  0.0,  s[2]]]
    s_matrix = np.array(s_matrix)
    rgb2xyz_mtx = gamut_mtx.T.dot(s_matrix)

    return rgb2xyz_mtx


def get_rgb_to_xyz_matrix(name):
    """
    RGB to XYZ の Matrix を求める。
    DCI-P3 で D65 の係数を返せるように内部関数化した。
    """
    if name != "DCI-P3":
        rgb_to_xyz_matrix = RGB_COLOURSPACES[name].matrix_RGB_to_XYZ
    else:
        rgb_to_xyz_matrix\
            = calc_rgb_to_xyz_matrix(RGB_COLOURSPACES[DCI_P3].primaries,
                                     xy_to_XYZ(ILLUMINANTS[CMFS_NAME]['D65']))

    return rgb_to_xyz_matrix


def get_xyz_to_rgb_matrix(name):
    """
    XYZ to RGB の Matrix を求める。
    DCI-P3 で D65 の係数を返せるように内部関数化した。
    """
    if name != "DCI-P3":
        xyz_to_rgb_matrix = RGB_COLOURSPACES[name].matrix_XYZ_to_RGB
    else:
        rgb_to_xyz_matrix\
            = calc_rgb_to_xyz_matrix(RGB_COLOURSPACES[DCI_P3].primaries,
                                     xy_to_XYZ(ILLUMINANTS[CMFS_NAME]['D65']))
        xyz_to_rgb_matrix = linalg.inv(rgb_to_xyz_matrix)

    return xyz_to_rgb_matrix


def get_white_point(name=ACES_AP0):
    if name == "DCI-P3":
        return ILLUMINANTS[CMFS_NAME]['D65']
    else:
        illuminant = RGB_COLOURSPACES[name].illuminant
        return ILLUMINANTS[CMFS_NAME][illuminant]


def rgb2rgb_mtx(src_name, dst_name):
    src_white = xy_to_XYZ(get_white_point(src_name))
    dst_white = xy_to_XYZ(get_white_point(dst_name))

    chromatic_adaptation_mtx = cat02_mtx(src_white, dst_white, 'CAT02')
    src_rgb2xyz_mtx = get_rgb_to_xyz_matrix(src_name)
    dst_xyz2rgb_mtx = get_xyz_to_rgb_matrix(dst_name)

    temp = np.dot(chromatic_adaptation_mtx, src_rgb2xyz_mtx)
    mtx = np.dot(dst_xyz2rgb_mtx, temp)

    return mtx


def mtx44_from_mtx33(mtx):
    out_mtx = [[mtx[0][0], mtx[0][1], mtx[0][2], 0],
               [mtx[1][0], mtx[1][1], mtx[1][2], 0],
               [mtx[2][0], mtx[2][1], mtx[2][2], 0],
               [0, 0, 0, 1]]

    return np.array(out_mtx)


def ocio_matrix_transform_mtx(src_name, dst_name):
    """
    OpenColorIO の MatrixTransform に食わせる Matrix を吐く。
    """
    mtx33 = rgb2rgb_mtx(src_name, dst_name)
    mtx44 = mtx44_from_mtx33(mtx33)

    return mtx44.flatten().tolist()


def get_primaries(color_space_name=BT709):
    return RGB_COLOURSPACES[color_space_name].primaries


def lab_to_rgb(lab, color_space_name, xyz_white=D65):
    rgb_linear = large_xyz_to_rgb(
        xyz=Lab_to_XYZ(lab), color_space_name=color_space_name,
        xyz_white=xyz_white)

    return rgb_linear


def oklab_to_rgb(oklab, color_space_name, xyz_white=D65):
    rgb_linear = large_xyz_to_rgb(
        xyz=Oklab_to_XYZ(oklab), color_space_name=color_space_name,
        xyz_white=xyz_white)

    return rgb_linear


def rgb_to_oklab(rgb_linear, color_space_name, xyz_white=D65):
    oklab = XYZ_to_Oklab(
        rgb_to_large_xyz(
            rgb=rgb_linear, color_space_name=color_space_name,
            xyz_white=xyz_white))

    return oklab


def large_xyz_to_rgb(
        xyz, color_space_name, xyz_white=D65):
    colourspace = RGB_COLOURSPACES[color_space_name]
    if np.array_equal(xyz_white, colourspace.whitepoint):
        chromatic_adaptation_transform = None
    else:
        chromatic_adaptation_transform = "CAT02"
    rgb_linear = XYZ_to_RGB(
        XYZ=xyz, colourspace=colourspace,
        illuminant=xyz_white,
        chromatic_adaptation_transform=chromatic_adaptation_transform)

    return rgb_linear


def rgb_to_large_xyz(
        rgb, color_space_name, xyz_white=D65):
    colourspace = RGB_COLOURSPACES[color_space_name]
    if np.array_equal(xyz_white, colourspace.whitepoint):
        chromatic_adaptation_transform = None
    else:
        chromatic_adaptation_transform = "CAT02"
    large_xyz = RGB_to_XYZ(
        RGB=rgb, colourspace=colourspace,
        illuminant=xyz_white,
        chromatic_adaptation_transform=chromatic_adaptation_transform)

    return large_xyz


def rgb_to_lab(
        rgb, color_space_name, xyz_white=D65):
    large_xyz = rgb_to_large_xyz(
        rgb=rgb, color_space_name=color_space_name,
        xyz_white=xyz_white)
    lab = XYZ_to_Lab(large_xyz)

    return lab


def jzazbz_to_rgb(
        jzazbz, color_space_name, xyz_white=D65, luminance=10000):
    """
    Examples
    --------
    >>> from jzazbz import large_xyz_to_jzazbz
    >>> large_xyz_100nits = np.array([95.04559271, 100, 108.90577508])
    >>> large_xyz_10000nits = large_xyz_100nits * 100
    >>> jzazbz_10000nits = large_xyz_to_jzazbz(large_xyz_10000nits)
    [  9.88606961e-01  -2.36258059e-04  -1.72125190e-04]
    >>> jzazbz_100nits = large_xyz_to_jzazbz(large_xyz_100nits)
    [  1.67173428e-01  -1.40335164e-04  -1.02252823e-04]
    >>> large_xyz_10000nits = jzazbz_to_rgb(
    ...     jzazbz=jzazbz_10000nits, color_space_name='ITU-R BT.709',
    ...     luminance=10000)
    [ 1.0  1.0  1.0]
    >>> large_xyz_100nits_on_hdr = jzazbz_to_rgb(
    ...     jzazbz=jzazbz_100nits, color_space_name='ITU-R BT.709',
    ...     luminance=10000)
    [ 0.01  0.01  0.01 ]
    >>> large_xyz_100nits_on_sdr = jzazbz_to_rgb(
    ...     jzazbz=jzazbz_100nits, color_space_name='ITU-R BT.709',
    ...     luminance=100)
    [ 1.0  1.0  1.0]
    """
    large_xyz = jzazbz_to_large_xyz(jzazbz=jzazbz) / luminance
    rgb_linear = large_xyz_to_rgb(
        xyz=large_xyz, color_space_name=color_space_name,
        xyz_white=xyz_white)
    # print(rgb_linear[-4:])
    return rgb_linear


def rgb_to_jzazbz(
        rgb, color_space_name, xyz_white=D65, luminance=10000):
    """
    Parameters
    ----------
    rgb : ndarray
        rgb value (linear).
    color_space_name : str
        color space name
    xyz_white : ndarray
        white point of the XYZ
    rgb_white : ndarray
        white point of the RGB

    Examples
    --------
    >>> rgb_to_jzazbz(
    ...     rgb=np.array([1, 1, 1]), color_space_name="ITU-R BT.709",
    ...     xyz_white=D65, rgb_white=D65, luminance=100)
    [  1.67173428e-01  -1.40335173e-04  -1.02252821e-04]

    >>> rgb_to_jzazbz(
    ...      rgb=np.array([1, 1, 1]), color_space_name="ITU-R BT.709",
    ...      xyz_white=D65, rgb_white=D65, luminance=10000)
    [  9.88606961e-01  -2.36258074e-04  -1.72125186e-04]
    """
    large_xyz = rgb_to_large_xyz(
        rgb=rgb, color_space_name=color_space_name,
        xyz_white=xyz_white) * luminance
    jzazbz = large_xyz_to_jzazbz(xyz=large_xyz)

    return jzazbz


def calc_hue_from_ab(aa, bb):
    """
    calculate hue.
    output range is [0, 2pi).

    Examples
    --------
    >>> aa = np.array([1.0, 0.5, 0.0, -0.5, -1.0, -0.5, 0.0, 0.5, 0.99])
    >>> bb = np.array([0.0, 0.5, 1.0, 0.5, 0.0, -0.5, -1.0, -0.5, -0.001])
    >>> hue = calc_hue_from_ab(aa, bb)
    [0.  45.  90.  135.  180.  225.  270.  315. 359.94212549]
    """
    hue = np.where(aa != 0, np.arctan(bb/aa), np.pi/2*np.sign(bb))
    add_pi_idx = (aa < 0) & (bb >= 0)
    sub_pi_idx = (aa < 0) & (bb < 0)
    hue[add_pi_idx] = hue[add_pi_idx] + np.pi
    hue[sub_pi_idx] = hue[sub_pi_idx] - np.pi

    hue[hue < 0] = hue[hue < 0] + 2 * np.pi

    return np.rad2deg(hue)


def xyY_to_Ych(xyY, white=D65):
    """
    Convert xyY to Ych color space.
    Ych is my original color space expressed in polar coordinate.
    `c` means chroma. `h` means hue-angle.

    Examples
    --------
    >>> rgb = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    >>> large_xyz = rgb_to_large_xyz(rgb, BT2020)
    >>> from colour import XYZ_to_xyY
    >>> xyY = XYZ_to_xyY(large_xyz)
    >>> Ych = xyY_to_Ych(xyY)
    >>> print(Ych)
    [[  2.62700212e-01   3.97027820e-01   3.54652706e+02]
     [  6.77998072e-01   4.89272204e-01   1.06957225e+02]
     [  5.93017165e-02   3.36309218e-01   2.37297530e+02]]
    """
    large_y = xyY[..., 2]
    xx = xyY[..., 0] - white[0]
    yy = xyY[..., 1] - white[1]

    cc = (xx ** 2 + yy ** 2) ** 0.5
    hh = calc_hue_from_ab(xx, yy)

    return tstack([large_y, cc, hh])


def Ych_to_xyY(Ych, white=D65):
    """
    Convert Ych color space to xyY.
    Ych is my original color space expressed in polar coordinate.
    `c` means chroma. `h` means hue-angle.

    Examples
    --------
    >>> rgb = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    >>> large_xyz = rgb_to_large_xyz(rgb, BT2020)
    >>> from colour import XYZ_to_xyY
    >>> xyY = XYZ_to_xyY(large_xyz)
    >>> Ych = xyY_to_Ych(xyY)
    >>> xyY = Ych_to_xyY(Ych)
    >>> print(xyY)
    [[ 0.708       0.292       0.26270021]
     [ 0.17        0.797       0.67799807]
     [ 0.131       0.046       0.05930172]]
    """
    large_y = Ych[..., 0]
    cc = Ych[..., 1]
    hh = Ych[..., 2]

    hh_rad = np.deg2rad(hh)
    xx = cc * np.cos(hh_rad) + white[0]
    yy = cc * np.sin(hh_rad) + white[1]

    return tstack([xx, yy, large_y])


def oklab_to_oklch(oklab):
    return Jab_to_JCh(oklab)


def oklch_to_oklab(oklch):
    return JCh_to_Jab(oklch)


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    # print(rgb2rgb_mtx(DCI_P3, ACES_AP0))
    # print(rgb2rgb_mtx(BT709, DCI_P3))
    # ocio_config_mtx_str(ACES_AP0, DCI_P3)
    # ocio_config_mtx_str(DCI_P3, ACES_AP0)
    # print(get_white_point(SRTB))
    # print(get_xyz_to_rgb_matrix(SRTB))
    # bt709_ap0 = rgb2rgb_mtx(BT709, ACES_AP0)
    # print(bt709_ap0)
    # ap0_bt709 = rgb2rgb_mtx(ACES_AP0, BT709)
    # print(ocio_matrix_transform_mtx(ACES_AP0, BT709))
    # print(ocio_matrix_transform_mtx(BT709, ACES_AP0))

    # print(ocio_matrix_transform_mtx(ACES_AP0, DCI_P3))
    # print(ocio_matrix_transform_mtx(DCI_P3, ACES_AP0))

    # xyY = np.array(
    #     [[0.3127, 0.3290, 1.0], [0.64, 0.33, 0.2], [0.30, 0.60, 0.6]])
    # result = calc_rgb_from_xyY(
    #     xyY=xyY, color_space_name=BT709, white=D65)
    # print(result)

    # xyY = np.array(
    #     [[0.3127, 0.3290, 1.0], [0.64, 0.33, 0.2], [0.30, 0.60, 0.6]])
    # result = calc_rgb_from_XYZ(
    #     XYZ=xyY_to_XYZ(xyY), color_space_name=BT709, white=D65)
    # print(result)

    # data = np.array(
    #     [[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    # print(split_tristimulus_values(data))

    # from colour import XYZ_to_xyY
    # rgb = np.array(
    #     [[1.0, 1.0, 1.0], [0.18, 0.18, 0.18], [1.0, 0.0, 0.0]])
    # XYZ = calc_XYZ_from_rgb(
    #     rgb=rgb, color_space_name=BT709, white=D65)
    # print(XYZ_to_xyY(XYZ))
    # from jzazbz import large_xyz_to_jzazbz
    # print(xy_to_XYZ([0.3127, 0.3290]) * 100)
    # large_xyz_100nits = np.array([95.04559271, 100, 108.90577508])
    # large_xyz_10000nits = large_xyz_100nits * 100
    # jzazbz_10000nits = large_xyz_to_jzazbz(large_xyz_10000nits)
    # print(jzazbz_10000nits)
    # jzazbz_100nits = large_xyz_to_jzazbz(large_xyz_100nits)
    # print(jzazbz_100nits)
    # large_xyz_10000nits = jzazbz_to_rgb(
    #     jzazbz=jzazbz_10000nits, color_space_name='ITU-R BT.709',
    #     luminance=10000)
    # print(large_xyz_10000nits)
    # large_xyz_100nits_on_hdr = jzazbz_to_rgb(
    #     jzazbz=jzazbz_100nits, color_space_name='ITU-R BT.709',
    #     luminance=10000)
    # print(large_xyz_100nits_on_hdr)
    # large_xyz_100nits_on_sdr = jzazbz_to_rgb(
    #     jzazbz=jzazbz_100nits, color_space_name='ITU-R BT.709',
    #     luminance=100)
    # print(large_xyz_100nits_on_sdr)

    # jab = rgb_to_jzazbz(
    #     rgb=np.array([1, 1, 1]), color_space_name="ITU-R BT.709",
    #     xyz_white=D65, rgb_white=D65, luminance=100)
    # print(jab)

    # jab = rgb_to_jzazbz(
    #     rgb=np.array([1, 1, 1]), color_space_name="ITU-R BT.709",
    #     xyz_white=D65, rgb_white=D65, luminance=10000)
    # print(jab)

    # rgb = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    # large_xyz = rgb_to_large_xyz(rgb, BT2020)
    # from colour import XYZ_to_xyY
    # xyY = XYZ_to_xyY(large_xyz)
    # Ych = xyY_to_Ych(xyY)
    # print(Ych)
    # xyY = Ych_to_xyY(Ych)
    # print(xyY)
