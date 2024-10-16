# -*- coding: utf-8 -*-
"""
CIELAB色空間の基礎調査
=====================

* XYZ色空間と Lab色空間の順変換・逆変換の数式を確認
* CLELAB a*b* plane (以後は a*b* plane と略す) のプロット(L を 0～100 まで 0.1 step で)
* CIELAB C*L* plane (以後は C*L* plane と略す) のプロット(h を 0～360 まで 0.5 step で)

"""

# import standard libraries
import os
import time

# import third-party libraries
import numpy as np
from sympy import symbols, plotting, sin, cos
from sympy.solvers import solve
from scipy import linalg
from colour.models import BT2020_COLOURSPACE, BT709_COLOURSPACE
from colour import xy_to_XYZ
from numba import jit
import matplotlib.pyplot as plt

# import my libraries
import color_space as cs
import plot_utility as pu

# definition
D65_X = 95.04
D65_Y = 100.0
D65_Z = 108.89
D65_WHITE = [D65_X, D65_Y, D65_Z]
SIGMA = 6/29

# information
__author__ = 'Toru Yoshihara'
__copyright__ = 'Copyright (C) 2019 - Toru Yoshihara'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Toru Yoshihara'
__email__ = 'toru.ver.11 at-sign gmail.com'

__all__ = []


def check_basic_trigonometricfunction():
    print(np.sin(np.pi * -4 / 4))
    print(np.sin(np.pi * -2 / 4))
    print(np.sin(np.pi * 2 / 4))
    print(np.sin(np.pi * 4 / 4))


def get_inv_f_upper():
    """
    t > sigma の f^-1 を返す
    """
    t = symbols('t')
    return t ** 3, t


def get_inv_f_lower():
    """
    t <= sigma の f^-1 を返す
    """
    t = symbols('t')
    sigma = SIGMA
    return 3 * (sigma ** 2) * (t - 4 / 29), t


def plot_inv_f():
    upper, t = get_inv_f_upper()
    plotting.plot(upper, (t, -1, 1))

    lower, t = get_inv_f_lower()
    plotting.plot(lower, (t, -1, 1))


def get_large_xyz_symbol(n, t, upper=True):
    """
    example
    -------
    c, l, h = symbols('c, l, h')
    xt = (l + 16) / 116 + (c * cos(h)) / 500
    yt = (l + 16) / 116
    zt = (l + 16) / 116 - (c * sin(h)) / 200

    x = get_large_xyz_symbol(n=D65_X, t=xt, upper=True)
    y = get_large_xyz_symbol(n=D65_Y, t=yt, upper=True)
    z = get_large_xyz_symbol(n=D65_Z, t=zt, upper=True)
    """
    func, u = get_inv_f_upper() if upper else get_inv_f_lower()
    return n / 100 * func.subs({u: t})


def apply_matrix(src, mtx):
    """
    src: [3]
    mtx: [3][3]
    """
    a = src[0] * mtx[0][0] + src[1] * mtx[0][1] + src[2] * mtx[0][2]
    b = src[0] * mtx[1][0] + src[1] * mtx[1][1] + src[2] * mtx[1][2]
    c = src[0] * mtx[2][0] + src[1] * mtx[2][1] + src[2] * mtx[2][2]

    return a, b, c


def get_xyz_to_rgb_matrix(primaries=cs.REC2020_xy):
    rgb_to_xyz_matrix = cs.calc_rgb_to_xyz_matrix(
        gamut_xy=primaries, white_large_xyz=D65_WHITE)
    xyz_to_rgb_matrix = linalg.inv(rgb_to_xyz_matrix)
    return xyz_to_rgb_matrix


def calc_chroma(h_sample=32):
    l_val = 50
    h = np.linspace(0, 1, h_sample) * 2 * np.pi

    chroma = []
    for h_val in h:
        chroma.append(lab_to_xyz_formla(l_val, h_val))
    return np.array(chroma)


def plot_ab_plane():
    h_sample = 32
    h = np.linspace(0, 1, h_sample) * 2 * np.pi
    chroma = calc_chroma(h_sample)
    a = chroma * np.cos(h)
    b = chroma * np.sin(h)

    ax1 = pu.plot_1_graph(
        fontsize=20,
        figsize=(10, 8),
        graph_title="Title",
        graph_title_size=None,
        xlabel="X Axis Label", ylabel="Y Axis Label",
        axis_label_size=None,
        legend_size=17,
        xlim=None,
        ylim=None,
        xtick=None,
        ytick=None,
        xtick_size=None, ytick_size=None,
        linewidth=3,
        minor_xtick_num=None,
        minor_ytick_num=None)
    ax1.plot(a, b, label="ab plane")
    plt.legend(loc='upper left')
    plt.show()


def lab_to_xyz_formla():
    """
    数式を取得
    """
    matrix = get_xyz_to_rgb_matrix(primaries=cs.REC2020_xy)

    # base formula
    c, l, h = symbols('c, l, h', real=True)
    xt = (l + 16) / 116 + (c * cos(h)) / 500
    yt = (l + 16) / 116
    zt = (l + 16) / 116 - (c * sin(h)) / 200
    xyz_t = [xt, yt, zt]

    # upper
    upper_xyzt = [
        get_large_xyz_symbol(n=D65_WHITE[idx], t=xyz_t[idx], upper=True)
        for idx in range(3)]
    upper_rgb = apply_matrix(upper_xyzt, matrix)

    # lower
    lower_xyzt = [
        get_large_xyz_symbol(n=D65_WHITE[idx], t=xyz_t[idx], upper=False)
        for idx in range(3)]
    lower_rgb = apply_matrix(lower_xyzt, matrix)

    # chroma = solve_chroma(upper_rgb, lower_rgb, xyz_t, l, h, l_val, h_val, c)

    return upper_rgb, lower_rgb, xyz_t, l, h, c


def solve_chroma(
        l_vals=[50], h_vals=np.linspace(0, 2*np.pi, 64)):
    upper_rgb, lower_rgb, xyz_t, l, h, c = lab_to_xyz_formla()
    chroma_return = []
    for l_val in l_vals:
        chroma = []
        for h_val in h_vals:
            temp = solve_chroma_sub(
                upper_rgb, lower_rgb, xyz_t, l, h, l_val, h_val, c)
            chroma.append(temp)
        chroma_return.append(chroma)

    return np.array(chroma_return)


def solve_chroma_sub(upper_rgb, lower_rgb, xyz_t, l, h, l_val, h_val, c):
    """
    与えられた条件下での Chroma の限界値を算出する。


    """
    start = time.time()

    upper_rgb = [
        upper_rgb[idx].subs({l: l_val, h: h_val}) for idx in range(3)]
    lower_rgb = [
        lower_rgb[idx].subs({l: l_val, h: h_val}) for idx in range(3)]
    xyz_t = [
        xyz_t[idx].subs({l: l_val, h: h_val}) for idx in range(3)]
    end = time.time()
    # print("intro = {}".format(end - start))

    # まず解く
    start = time.time()
    upper_solution_zero = [solve(upper_rgb[idx] + 0) for idx in range(3)]
    upper_solution_one = [solve(upper_rgb[idx] - 1) for idx in range(3)]
    lower_solution_zero = [solve(lower_rgb[idx] + 0) for idx in range(3)]
    lower_solution_one = [solve(lower_rgb[idx] - 1) for idx in range(3)]
    end = time.time()
    # print("mazutoku = {}".format(end - start))

    # それぞれの解が \sigma の条件を満たしているか確認
    start = time.time()
    solve_list = []
    for idx in range(3):
        for solve_val in upper_solution_zero[idx]:
            t_val = xyz_t[idx].subs({c: solve_val})
            if t_val > SIGMA:
                solve_list.append(solve_val)
        for solve_val in upper_solution_one[idx]:
            t_val = xyz_t[idx].subs({c: solve_val})
            if t_val > SIGMA:
                solve_list.append(solve_val)

        for solve_val in lower_solution_zero[idx]:
            t_val = xyz_t[idx].subs({c: solve_val})
            if t_val <= SIGMA:
                solve_list.append(solve_val)
        for solve_val in lower_solution_one[idx]:
            t_val = xyz_t[idx].subs({c: solve_val})
            if t_val <= SIGMA:
                solve_list.append(solve_val)

    end = time.time()
    # print("sorekai = {}".format(end - start))

    # 出揃った全てのパターンの中から最小値を選択する
    start = time.time()
    solve_list = np.array(solve_list)
    chroma = np.min(solve_list[solve_list >= 0.0])
    end = time.time()
    # print("get_minimum = {}".format(end - start))

    return chroma


def experimental_functions():
    # check_basic_trigonometricfunction()
    # plot_inv_f()
    # plot_ab_plane()
    solve_list = solve_chroma()
    l_len = len(solve_list)
    l_val = np.linspace(0, 100, l_len)
    p_str = "{}, {}, {}"
    print(50)
    for idx in range(len(solve_list[0])):
        print(solve_list[0][idx])


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    experimental_functions()
