# -*- coding: utf-8 -*-
"""
動くタマをプロットする
=====================

要は ReflectiveMovingObject の動作確認

"""

# import standard libraries
import os
from pathlib import Path

# import third-party libraries
import numpy as np
from numpy.random import rand, seed
import cv2
import test_pattern_generator2 as tpg
from multiprocessing import Pool, cpu_count

# import my libraries
from reflective_moving_object import ReflectiveMovingObject
import transfer_functions as tf
from common import MeasureExecTime

# information
__author__ = 'Toru Yoshihara'
__copyright__ = 'Copyright (C) 2020 - Toru Yoshihara'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Toru Yoshihara'
__email__ = 'toru.ver.11 at-sign gmail.com'

__all__ = []

mtime = MeasureExecTime()


def main_func():
    radius = 30
    bg_level = 96
    velocity_rate = 20
    seed(0)
    velocity_list = np.array([rand(2), rand(2), rand(2), rand(2)])
    velocity_list = np.int16(np.round(velocity_list * velocity_rate))
    color_list = [(192, 192, 192), (192, 0, 0), (0, 192, 0), (0, 0, 192)]

    rmo_list = [ReflectiveMovingObject(
        velocity_init=velocity_list[idx], radius=radius)
        for idx in range(len(velocity_list))]
    trial_num = 720
    img_base = np.ones((1080, 1920, 3), dtype=np.uint8) * bg_level
    for idx in range(trial_num):
        fname = "./img/test_ball_bg_{:04d}_lv_{:04d}.png".format(bg_level, idx)
        pos_list = [
            rmo_list[idx].get_pos() for idx in range(len(velocity_list))]
        img = img_base.copy()
        for idx, pos in enumerate(pos_list):
            img = cv2.circle(
                img, pos_list[idx], radius,
                color=color_list[idx], thickness=-1)
            rmo_list[idx].calc_next_pos()
        cv2.imwrite(fname, img[:, :, ::-1])


def theread_wrapper_draw_main(kwargs):
    draw_main(**kwargs)


def draw_main(
        idx, radius, pos_list_total, bg_image_name, color_list,
        width, height, vr, fps):
    basename = Path(bg_image_name).stem
    fname = "/work/overuse/2020/003_moving_ball/img_seq_vr/"
    fname += f"size_{radius}_{basename}_vr{int(vr*10)}_"
    fname += f"{int(fps)}fps_{idx:04d}.png"
    bg_image = cv2.imread(
        bg_image_name, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_COLOR) / 0xFFFF
    # pos_list = [
    #     rmo_list[idx].get_pos() for idx in range(len(velocity_list))]
    img = np.zeros((height, width, 3), dtype=np.uint8)
    for c_idx, pos in enumerate(pos_list_total[idx]):
        img = cv2.circle(
            img, pos, radius,
            color=color_list[c_idx], thickness=-1, lineType=cv2.LINE_AA)
        # rmo_list[idx].calc_next_pos()  # マルチスレッド化にともない事前に計算
    # alpha channel は正規化する。そうしないと中間調合成時に透けてしまう
    alpha = np.max(img, axis=-1)
    alpha = alpha / np.max(alpha)
    img = np.dstack((img / 0xFF, alpha))

    bg_temp = bg_image.copy()
    tpg.merge_with_alpha2(bg_temp, img)
    print(fname)
    # cv2.imwrite(
    #     fname, np.uint16(np.round(bg_temp[:, :, ::-1] * 0xFFFF)),
    #     [cv2.IMWRITE_PNG_COMPRESSION, 7])
    tpg.img_wirte_float_as_16bit_int(fname, bg_temp, comp_val=7)


def plot_with_bg_image(
        width=1920, height=1080, radius=20,
        bg_image_name="./img/bg_img_5x3.png", vr=1.0, fps=60):
    velocity_rate = 20 * height / 1080 * vr
    color_list = [
        [192, 192, 192], [192, 0, 0], [0, 192, 0], [0, 0, 192],
        [192, 0, 192], [192, 192, 0], [0, 192, 192]]
    seed(0)
    velocity_list = np.array(
        [rand(2) for x in range(len(color_list))])
    velocity_list = np.int32(np.round(velocity_list * velocity_rate))

    rmo_list = [ReflectiveMovingObject(
        velocity_init=velocity_list[idx], radius=radius,
        outline_size=(width, height))
        for idx in range(len(velocity_list))]
    sec = 15
    trial_num = sec * fps

    # 先に全部座標を計算してしまう
    pos_list_total = []
    for idx in range(trial_num):
        pos_list = [
            rmo_list[idx].get_pos() for idx in range(len(velocity_list))]
        pos_list_total.append(pos_list)
        for idx, pos in enumerate(pos_list):
            rmo_list[idx].calc_next_pos()

    total_process_num = trial_num
    block_process_num = int(cpu_count() * 0.4)
    block_num = int(round(total_process_num / block_process_num + 0.5))
    mtime.start()
    for b_idx in range(block_num):
        args = []
        for p_idx in range(block_process_num):
            h_idx = b_idx * block_process_num + p_idx              # User
            print(f"b_idx={b_idx}, p_idx={p_idx}, h_idx={h_idx}")  # User
            if h_idx >= total_process_num:                         # User
                break
            d = dict(
                idx=h_idx, radius=radius, pos_list_total=pos_list_total.copy(),
                bg_image_name=bg_image_name, color_list=color_list,
                width=width, height=height, vr=vr, fps=fps)
            args.append(d)
            # draw_main(**d)
            mtime.lap("p loop")
        with Pool(cpu_count()) as pool:
            pool.map(theread_wrapper_draw_main, args)
    mtime.end()


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    # main_func()
    bg_h_name = "./img/h_inc_grad_3840x2160_1000-nits.png"
    bg_v_name = "./img/v_inc_grad_3840x2160_1000_nits.png"
    c_6x4_name = "./img/bg_img_6x4.png"
    # plot_with_bg_image(
    #     radius=5 * 2, width=3840, height=2160, bg_image_name=bg_h_name)
    # plot_with_bg_image(
    #     radius=10 * 2, width=3840, height=2160, bg_image_name=bg_h_name)
    # plot_with_bg_image(
    #     radius=20 * 2, width=3840, height=2160, bg_image_name=bg_h_name)
    # plot_with_bg_image(
    #     radius=40 * 2, width=3840, height=2160, bg_image_name=bg_h_name)

    # plot_with_bg_image(
    #     radius=5 * 2, width=3840, height=2160, bg_image_name=bg_v_name)
    # plot_with_bg_image(
    #     radius=10 * 2, width=3840, height=2160, bg_image_name=bg_v_name)
    # plot_with_bg_image(
    #     radius=20 * 2, width=3840, height=2160, bg_image_name=bg_v_name)
    # plot_with_bg_image(
    #     radius=40 * 2, width=3840, height=2160, bg_image_name=bg_v_name)

    plot_with_bg_image(
        radius=20*4, width=3840, height=2160, bg_image_name=c_6x4_name, vr=1.0,
        fps=60)
    plot_with_bg_image(
        radius=20*4, width=3840, height=2160, bg_image_name=c_6x4_name, vr=1.5,
        fps=60)
    plot_with_bg_image(
        radius=20*4, width=3840, height=2160, bg_image_name=c_6x4_name, vr=2.0,
        fps=60)

    plot_with_bg_image(
        radius=20*4, width=3840, height=2160, bg_image_name=c_6x4_name, vr=1.0,
        fps=120)
    plot_with_bg_image(
        radius=20*4, width=3840, height=2160, bg_image_name=c_6x4_name, vr=1.5,
        fps=120)
    plot_with_bg_image(
        radius=20*4, width=3840, height=2160, bg_image_name=c_6x4_name, vr=2.0,
        fps=120)

    # debug
    # plot_with_bg_image(
    #     radius=15 * 2, width=3840, height=2160, bg_image_name=bg_h_name)
