#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
ST2084-BT.2020-D65 のデータを 2.4-BT.709-D65 に変換する3DLUTを作る。
ソースが ST2084 なので display reffered な変換とする（system gamma を考慮しない）。
"""

# 外部ライブラリのインポート
import os
import numpy as np
from colour.models import eotf_ST2084
from colour import RGB_to_RGB
from colour.models import BT2020_COLOURSPACE, BT709_COLOURSPACE

# 自作ライブラリのインポート
import lut

NOMINAL_WHITE_LUMINANCE = 100


def make_3dlut_grid(grid_num=33):
    """
    3DLUTの格子点データを作成

    Parameters
    ----------
    x : integer
        A number of grid points.

    Returns
    -------
    ndarray
        An Array of the grid points.
        The shape is (1, grid_num ** 3, 3).

    Examples
    --------
    >>> make_3dlut_grid(grid_num=3)
    array([[[0. , 0. , 0. ],
            [0.5, 0. , 0. ],
            [1. , 0. , 0. ],
            [0. , 0.5, 0. ],
            [0.5, 0.5, 0. ],
            [1. , 0.5, 0. ],
            [0. , 1. , 0. ],
            [0.5, 1. , 0. ],
            [1. , 1. , 0. ],
            [0. , 0. , 0.5],
            [0.5, 0. , 0.5],
            [1. , 0. , 0.5],
            [0. , 0.5, 0.5],
            [0.5, 0.5, 0.5],
            [1. , 0.5, 0.5],
            [0. , 1. , 0.5],
            [0.5, 1. , 0.5],
            [1. , 1. , 0.5],
            [0. , 0. , 1. ],
            [0.5, 0. , 1. ],
            [1. , 0. , 1. ],
            [0. , 0.5, 1. ],
            [0.5, 0.5, 1. ],
            [1. , 0.5, 1. ],
            [0. , 1. , 1. ],
            [0.5, 1. , 1. ],
            [1. , 1. , 1. ]]])
    """
    x = np.linspace(0, 1, grid_num)
    rgb_mesh_array = np.meshgrid(x, x, x)
    rgb_mesh_array = [x.reshape(1, grid_num ** 3, 1) for x in rgb_mesh_array]
    rgb_grid = np.dstack(
        (rgb_mesh_array[2], rgb_mesh_array[0], rgb_mesh_array[1]))

    return rgb_grid


def main(grid_num=65):
    # R, G, B の grid point データを準備
    x = make_3dlut_grid(grid_num=grid_num)

    # linear に戻す
    linear_luminance = eotf_ST2084(x)

    # 単位が輝度(0～10000 nits)になっているので
    # 一般的に使われる 1.0 が 100 nits のスケールに変換
    linear = linear_luminance / NOMINAL_WHITE_LUMINANCE

    # 色域を BT.2020 --> BT.709 に変換
    linear_bt709 = RGB_to_RGB(RGB=linear,
                              input_colourspace=BT2020_COLOURSPACE,
                              output_colourspace=BT709_COLOURSPACE)

    # BT.709 の範囲外の値(linear < 0.0 と linear > 1.0 の領域)をクリップ
    linear_bt709 = np.clip(linear_bt709, 0.0, 1.0)

    # BT.709 のガンマ(OETF)をかける
    non_linear_bt709 = linear_bt709 ** (1 / 2.4)

    # 自作の LUTライブラリのクソ仕様のため shape を変換する
    lut_for_save = non_linear_bt709.reshape((grid_num ** 3, 3))

    # .cube の形式で保存
    lut_fname = "./st2084_bt2020_to_gamma2.4_bt709.cube"
    lut.save_3dlut(
        lut=lut_for_save, grid_num=grid_num, filename=lut_fname)


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    main()
