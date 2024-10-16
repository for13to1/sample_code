# -*- coding: utf-8 -*-
"""
plot gamut boundary
"""

# import standard libraries
import os

# import third-party libraries
import numpy as np
from colour import XYZ_to_RGB, RGB_COLOURSPACES, xy_to_XYZ
from multiprocessing import Pool, cpu_count

# import my libraries
import color_space as cs
import transfer_functions as tf
from jzazbz import jzazbz_to_large_xyz, large_xyz_to_jzazbz, st2084_oetf_like, st2084_eotf_like,\
    jzczhz_to_jzazbz
from create_gamut_booundary_lut import is_out_of_gamut_rgb
from test_pattern_generator2 import img_wirte_float_as_16bit_int
import plot_utility as pu

# information
__author__ = 'Toru Yoshihara'
__copyright__ = 'Copyright (C) 2021 - Toru Yoshihara'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Toru Yoshihara'
__email__ = 'toru.ver.11 at-sign gmail.com'

__all__ = []


def create_valid_ab_plane_image_st2084(
        j_val=0.5, ab_max=1.0, ab_sample=512, color_space_name=cs.BT2020,
        bg_rgb_luminance=np.array([50, 50, 50])):
    """
    Create an image that indicates the valid area of the ab plane.

    Parameters
    ----------
    j_val : float
        A Lightness value. range is 0.0 - 1.0
    ab_max : float
        A maximum value of the a, b range.
    ab_sapmle : int
        A number of samples in the image resolution.
    color_space_name : str
        color space name for colour.RGB_COLOURSPACES
    """
    aa_base = np.linspace(-ab_max, ab_max, ab_sample)
    bb_base = np.linspace(-ab_max, ab_max, ab_sample)
    aa = aa_base.reshape((1, ab_sample))\
        * np.ones_like(bb_base).reshape((ab_sample, 1))
    bb = bb_base.reshape((ab_sample, 1))\
        * np.ones_like(aa_base).reshape((1, ab_sample))
    jj = np.ones_like(aa) * j_val
    jzazbz = np.dstack((jj, aa, bb[::-1])).reshape((ab_sample, ab_sample, 3))
    large_xyz = jzazbz_to_large_xyz(jzazbz)
    rgb_luminance = XYZ_to_RGB(
        large_xyz, cs.D65, cs.D65,
        RGB_COLOURSPACES[color_space_name].matrix_XYZ_to_RGB)

    ng_idx = is_out_of_gamut_rgb(rgb=rgb_luminance/10000)
    rgb_luminance[ng_idx] = bg_rgb_luminance
    rgb_st2084 = tf.oetf_from_luminance(
        np.clip(rgb_luminance, 0.0, 10000), tf.ST2084)

    return rgb_st2084


def create_valid_cj_plane_image_st2084(
        h_val=50, c_max=1, c_sample=1024, j_sample=1024,
        color_space_name=cs.BT2020, bg_rgb_luminance=np.array([50, 50, 50]),
        maximum_luminance=10000):
    """
    Create an image that indicates the valid area of the ab plane.

    Parameters
    ----------
    h_val : float
        A Hue value. range is 0.0 - 360.0
    c_max : float
        A maximum value of the chroma.
    c_sapmle : int
        A number of samples for the chroma.
    l_sample : int
        A number of samples for the lightness.
    color_space_name : str
        color space name for colour.RGB_COLOURSPACES
    bg_lightness : float
        background lightness value.
    maximum_luminance : float
        maximum luminance of the target display device.
    """
    l_max = 1

    cc_base = np.linspace(0, c_max, c_sample)
    jj_base = np.linspace(0, l_max, j_sample)
    cc = cc_base.reshape(1, c_sample)\
        * np.ones_like(jj_base).reshape(j_sample, 1)
    jj = jj_base.reshape(j_sample, 1)\
        * np.ones_like(cc_base).reshape(1, c_sample)
    hh = np.ones_like(cc) * h_val

    jczhz = np.dstack([jj[::-1], cc, hh]).reshape((j_sample, c_sample, 3))
    jzazbz = jzczhz_to_jzazbz(jczhz)
    large_xyz = jzazbz_to_large_xyz(jzazbz)
    rgb_luminance = XYZ_to_RGB(
        large_xyz, cs.D65, cs.D65,
        RGB_COLOURSPACES[color_space_name].matrix_XYZ_to_RGB)
    print(f"maximum_luminance = {maximum_luminance}")
    ng_idx = is_out_of_gamut_rgb(rgb=rgb_luminance/maximum_luminance)

    rgb_luminance[ng_idx] = bg_rgb_luminance

    rgb_st2084 = tf.oetf_from_luminance(
        np.clip(rgb_luminance, 0.0, 10000), tf.ST2084)

    return rgb_st2084


def plot_czjz_plane_st2084(
        h_idx, h_val, color_space_name, maximum_luminance=10000):
    """
    Parameters
    ----------
    h_idx : int
        A Hue index for ty_ch_lut
    h_val : float
        Hue Value of the ab plane
    color_space_name : str
        color space name for colour.RGB_COLOURSPACES
    maximum_luminance : float
        maximum luminance of the target display device.
    """
    c_max = 0.5
    c_sample = 1024
    j_max = 1
    j_sample = 1024
    rgb_st2084 = create_valid_cj_plane_image_st2084(
        h_val=h_val, c_max=c_max, c_sample=c_sample, j_sample=j_sample,
        color_space_name=color_space_name,
        bg_rgb_luminance=np.array([100, 100, 100]),
        maximum_luminance=maximum_luminance)
    graph_title = f"CzJz Plane,  peak {maximum_luminance} nits,  "
    graph_title += f"hz={h_val:.1f}°"

    fig, ax1 = pu.plot_1_graph(
        fontsize=20,
        figsize=(10, 10),
        bg_color=None,
        graph_title=graph_title,
        graph_title_size=None,
        xlabel="Cz", ylabel="Jz",
        axis_label_size=None,
        legend_size=17,
        xlim=[0, c_max],
        ylim=[0, j_max],
        xtick=None,
        ytick=None,
        xtick_size=None, ytick_size=None,
        linewidth=4,
        minor_xtick_num=None,
        minor_ytick_num=None)
    ax1.imshow(
        rgb_st2084, extent=(0, c_max, 0, j_max), aspect='auto')
    prefix = "/work/overuse/2021/10_jzazbz/img_seq_cj"
    fname = f"{prefix}/cj_plane_max-{maximum_luminance}nits_{h_idx:04d}.png"
    print(f"save file = {fname}")
    pu.show_and_save(
        fig=fig, legend_loc=None, show=False, save_fname=fname)


def plot_ab_plane_st2084(
        j_idx=0, j_val=0.5, ab_max=1.0, ab_sample=1536,
        color_space_name=cs.BT2020):
    rgb_st2084 = create_valid_ab_plane_image_st2084(
        j_val=j_val, ab_max=ab_max, ab_sample=ab_sample,
        color_space_name=color_space_name,
        bg_rgb_luminance=np.array([100, 100, 100]))
    luminance = int(
        round(st2084_eotf_like(j_val)) + 0.5)
    graph_title = f"azbz plane,  Jz={j_val:.2f},  Luminance={luminance} nits"

    fig, ax1 = pu.plot_1_graph(
        fontsize=20,
        figsize=(10, 10),
        bg_color=(0.96, 0.96, 0.96),
        graph_title=graph_title,
        graph_title_size=None,
        xlabel="az", ylabel="bz",
        axis_label_size=None,
        legend_size=17,
        xlim=[-ab_max, ab_max],
        ylim=[-ab_max, ab_max],
        xtick=None,
        ytick=None,
        xtick_size=None, ytick_size=None,
        linewidth=3,
        minor_xtick_num=None,
        minor_ytick_num=None)
    # ax1.plot([0], [0], '.')
    ax1.imshow(
        rgb_st2084, extent=(-ab_max, ab_max, -ab_max, ab_max), aspect='auto')
    fname = "/work/overuse/2021/10_jzazbz/img_seq_ab/"
    fname += f"azbz_plane_{color_space_name}_{j_idx:04d}.png"
    print(fname)
    pu.show_and_save(
        fig=fig, legend_loc=None, show=False, save_fname=fname)


def thread_wrapper_plot_ab_plane_st2084(args):
    plot_ab_plane_st2084(**args)


def plot_ab_plane_seq(color_space_name):
    """
    Parameters
    ----------
    ty_lch_lut : ndarray
        gamut boundary data.
        shape is (Lightness_num, Hue_num, 3).
        the data order is L*, C*, Hab
    """
    j_num = 501

    total_process_num = j_num
    block_process_num = cpu_count() // 2
    block_num = int(round(total_process_num / block_process_num + 0.5))

    for b_idx in range(block_num):
        args = []
        for p_idx in range(block_process_num):
            j_idx = b_idx * block_process_num + p_idx              # User
            print(f"b_idx={b_idx}, p_idx={p_idx}, l_idx={j_idx}")  # User
            if j_idx >= total_process_num:                         # User
                break
            d = dict(
                j_idx=j_idx, j_val=j_idx/(j_num-1), ab_max=0.5, ab_sample=1536,
                color_space_name=color_space_name)
            # plot_ab_plane_st2084(**d)
            args.append(d)
        #     break
        # break
        with Pool(cpu_count()) as pool:
            pool.map(thread_wrapper_plot_ab_plane_st2084, args)


def thread_wrapper_plot_czjz_plane_st2084(args):
    plot_czjz_plane_st2084(**args)


def plot_cj_plane_seq(color_space_name, maximum_luminance=10000):
    h_num = 721

    total_process_num = h_num
    block_process_num = cpu_count() // 2
    block_num = int(round(total_process_num / block_process_num + 0.5))

    for b_idx in range(block_num):
        args = []
        for p_idx in range(block_process_num):
            h_idx = b_idx * block_process_num + p_idx              # User
            print(f"b_idx={b_idx}, p_idx={p_idx}, h_idx={h_idx}")  # User
            if h_idx >= total_process_num:                         # User
                break
            d = dict(
                h_idx=h_idx, h_val=h_idx/(h_num-1)*360,
                color_space_name=color_space_name,
                maximum_luminance=maximum_luminance)
            # plot_czjz_plane_st2084(**d)
            args.append(d)
        #     break
        # break
        with Pool(cpu_count()) as pool:
            pool.map(thread_wrapper_plot_czjz_plane_st2084, args)


def check_jz_val_for_luminance():
    luminance_num = 1024
    x = np.linspace(1/255, 1, luminance_num)
    luminance = tf.eotf_to_luminance(x, tf.ST2084).reshape(-1, 1)
    d65_xyz = xy_to_XYZ(cs.D65).reshape(1, 3).repeat(
        luminance_num, axis=0) * luminance
    jzazbz = large_xyz_to_jzazbz(xyz=d65_xyz)
    jz = jzazbz[..., 0]

    fig, ax1 = pu.plot_1_graph(
        fontsize=20,
        figsize=(10, 7),
        bg_color=(0.96, 0.96, 0.96),
        graph_title="Relationship between Luminance and Jz (D65)",
        graph_title_size=None,
        xlabel="Luminance [nits]", ylabel="Jz",
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
    ax1.set_xscale('log', base=10)
    ax1.plot(luminance, jz, color=(0.2, 0.2, 0.2))
    fname = "./img/lumi_vs_jz.png"
    pu.show_and_save(
        fig=fig, legend_loc=None, show=False, save_fname=fname)


def thread_wrapper_plot_ab_plane_gm24(args):
    plot_ab_plane_gm24(**args)


def plot_ab_plane_seq_sdr(color_space_name):
    """
    Parameters
    ----------
    ty_lch_lut : ndarray
        gamut boundary data.
        shape is (Lightness_num, Hue_num, 3).
        the data order is L*, C*, Hab
    """
    j_num = 1001

    total_process_num = j_num
    block_process_num = cpu_count() // 2
    block_num = int(round(total_process_num / block_process_num + 0.5))

    for b_idx in range(block_num):
        args = []
        for p_idx in range(block_process_num):
            j_idx = b_idx * block_process_num + p_idx              # User
            print(f"b_idx={b_idx}, p_idx={p_idx}, l_idx={j_idx}")  # User
            if j_idx >= total_process_num:                         # User
                break
            d = dict(
                j_idx=j_idx, j_val=j_idx/(j_num-1), ab_max=0.5, ab_sample=1536,
                color_space_name=color_space_name)
            # plot_ab_plane_gm24(**d)
            args.append(d)
        #     break
        # break
        with Pool(cpu_count()) as pool:
            pool.map(thread_wrapper_plot_ab_plane_gm24, args)


def plot_ab_plane_gm24(
        j_idx=0, j_val=0.5, ab_max=1.0, ab_sample=1536,
        color_space_name=cs.BT2020):
    rgb_img = create_valid_ab_plane_image_gm24(
        j_val=j_val, ab_max=ab_max, ab_sample=ab_sample,
        color_space_name=color_space_name,
        bg_rgb_luminance=np.array([5000, 5000, 5000]))
    luminance = int(
        round(st2084_eotf_like(j_val)) + 0.5)
    graph_title = f"azbz plane,  {color_space_name},  Jz={j_val:.2f},  "
    graph_title += f"Luminance={luminance} nits"

    fig, ax1 = pu.plot_1_graph(
        fontsize=20,
        figsize=(10, 10),
        bg_color=(0.96, 0.96, 0.96),
        graph_title=graph_title,
        graph_title_size=None,
        xlabel="az", ylabel="bz",
        axis_label_size=None,
        legend_size=17,
        xlim=[-ab_max, ab_max],
        ylim=[-ab_max, ab_max],
        xtick=None,
        ytick=None,
        xtick_size=None, ytick_size=None,
        linewidth=3,
        minor_xtick_num=None,
        minor_ytick_num=None)
    # ax1.plot([0], [0], '.')
    ax1.imshow(
        rgb_img, extent=(-ab_max, ab_max, -ab_max, ab_max), aspect='auto')
    fname = "/work/overuse/2021/10_jzazbz/img_seq_ab/"
    fname += f"azbz_plane_{color_space_name}_{j_idx:04d}.png"
    print(fname)
    pu.show_and_save(
        fig=fig, legend_loc=None, show=False, save_fname=fname)


def create_valid_ab_plane_image_gm24(
        j_val=0.5, ab_max=0.5, ab_sample=512, color_space_name=cs.BT2020,
        bg_rgb_luminance=np.array([50, 50, 50])):
    """
    Create an image that indicates the valid area of the ab plane.

    Parameters
    ----------
    j_val : float
        A Lightness value. range is 0.0 - 1.0
    ab_max : float
        A maximum value of the a, b range.
    ab_sapmle : int
        A number of samples in the image resolution.
    color_space_name : str
        color space name for colour.RGB_COLOURSPACES
    """
    aa_base = np.linspace(-ab_max, ab_max, ab_sample)
    bb_base = np.linspace(-ab_max, ab_max, ab_sample)
    aa = aa_base.reshape((1, ab_sample))\
        * np.ones_like(bb_base).reshape((ab_sample, 1))
    bb = bb_base.reshape((ab_sample, 1))\
        * np.ones_like(aa_base).reshape((1, ab_sample))
    jj = np.ones_like(aa) * j_val
    jzazbz = np.dstack((jj, aa, bb[::-1])).reshape((ab_sample, ab_sample, 3))
    large_xyz = jzazbz_to_large_xyz(jzazbz)
    rgb_luminance = XYZ_to_RGB(
        large_xyz, cs.D65, cs.D65,
        RGB_COLOURSPACES[color_space_name].matrix_XYZ_to_RGB)

    ng_idx = is_out_of_gamut_rgb(rgb=rgb_luminance/10000)
    rgb_luminance[ng_idx] = bg_rgb_luminance
    rgb_gm24 = tf.oetf_from_luminance(
        np.clip(rgb_luminance/100, 0.0, 10000), tf.GAMMA24)

    return rgb_gm24


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    luminance = 1000

    plot_ab_plane_seq_sdr(color_space_name=cs.BT709)
    # plot_cj_plane_seq(color_space_name=cs.BT2020, maximum_luminance=100)
    # plot_cj_plane_seq(color_space_name=cs.BT2020, maximum_luminance=1000)
    # plot_cj_plane_seq(color_space_name=cs.BT2020, maximum_luminance=4000)
    # plot_cj_plane_seq(color_space_name=cs.BT2020, maximum_luminance=10000)
