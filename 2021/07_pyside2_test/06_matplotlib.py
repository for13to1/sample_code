# -*- coding: utf-8 -*-
"""
improve the 3dlut.
"""

# import standard libraries
import os
import sys
from matplotlib.backend_bases import Event

# import third-party libraries
import numpy as np
import matplotlib.pyplot as plt
from colour import RGB_to_XYZ, XYZ_to_RGB, CCT_to_xy
from colour.models import RGB_COLOURSPACE_BT709

from PySide2.QtWidgets import QApplication, QHBoxLayout, QWidget, QSlider,\
    QLabel, QVBoxLayout
from PySide2.QtCore import Qt
from PySide2 import QtWidgets
from PySide2.QtGui import QPixmap, QImage
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg\
    import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# import my libraries
import plot_utility as pu
import test_pattern_generator2 as tpg

# information
__author__ = 'Toru Yoshihara'
__copyright__ = 'Copyright (C) 2021 - Toru Yoshihara'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Toru Yoshihara'
__email__ = 'toru.ver.11 at-sign gmail.com'

__all__ = []


class GammaSlider(QWidget):
    def __init__(self, slot_func=None):
        super().__init__()
        self.int_float_rate = 10
        self.defalut_value = 2.2
        min_value = 0.1
        max_value = 3.0
        interval = 0.1
        step = 0.1
        self.slider = QSlider(orientation=Qt.Horizontal)
        self.slider.setMinimum(int(min_value * self.int_float_rate))
        self.slider.setMaximum(int(max_value * self.int_float_rate))
        self.slider.setValue(int(self.defalut_value * self.int_float_rate))
        self.slider.setTickInterval(int(interval * self.int_float_rate))
        self.slider.setTickPosition(QSlider.TicksBelow)
        self.slider.setSingleStep(int(step * self.int_float_rate))
        # print(f"step={int(step * self.int_float_rate)}")

    def change_value(self):
        print("changed")
        value = self.get_value()
        print(value)

    def set_value(self, value=2.2):
        self.slider.setValue(int(value * self.int_float_rate))

    def get_value(self):
        return self.slider.value() / self.int_float_rate

    def get_widget(self):
        return self.slider

    def get_default(self):
        return self.defalut_value

    def set_slot(self, slot_func):
        self.slot_func = slot_func
        self.slot_func()
        self.slider.valueChanged.connect(self.slot_func)


class ColorTempSlider(QWidget):
    def __init__(self, slot_func=None):
        super().__init__()
        self.int_float_rate = 1/100
        self.defalut_value = 6500
        min_value = 3000
        max_value = 10000
        self.slider = QSlider(orientation=Qt.Horizontal)
        self.slider.setMinimum(int(min_value * self.int_float_rate))
        self.slider.setMaximum(int(max_value * self.int_float_rate))
        self.slider.setValue(int(self.defalut_value * self.int_float_rate))
        self.slider.setTickPosition(QSlider.TicksBelow)

    def change_value(self):
        print("color temp changed")
        value = self.get_value()
        print(value)

    def set_value(self, value=6500):
        self.slider.setValue(int(value * self.int_float_rate))

    def get_value(self):
        return self.slider.value() / self.int_float_rate

    def get_widget(self):
        return self.slider

    def get_default(self):
        return self.defalut_value

    def set_slot(self, slot_func):
        self.slot_func = slot_func
        self.slot_func()
        self.slider.valueChanged.connect(self.slot_func)


class GammaLabel(QWidget):
    def __init__(self, default=2.2):
        super().__init__()
        self.label = QLabel(str(default))

    def get_widget(self):
        return self.label

    def set_label(self, value):
        self.label.setText(str(value))


class ColorTempLabel(QWidget):
    def __init__(self, default=6500):
        super().__init__()
        self.label = QLabel(str(default))

    def get_widget(self):
        return self.label

    def set_label(self, value):
        self.label.setText(str(int(value)))


class MatplotlibTest():
    def __init__(
            self, figsize=(10, 8), sample_num=1024, init_gamma=2.2):
        super().__init__()
        self.fig = Figure(figsize=figsize)
        self.ax1 = self.fig.add_subplot(111)
        self.plot_init_plot(sample_num=sample_num, gamma=init_gamma)
        self.canvas = FigureCanvas(self.fig)

    def plot_init_plot(self, sample_num, gamma=2.2):
        self.x = np.linspace(0, 1, sample_num)
        self.y = self.x ** gamma
        self.plot_obj, =\
            self.ax1.plot(self.x, self.y, label=f"gamma={gamma}")

    def get_widget(self):
        return self.canvas

    def update_plot(self, gamma=2.4):
        y = self.x ** (1/gamma)
        self.plot_obj.set_data(self.x, y)
        self.plot_obj.figure.canvas.draw()


class MyLayout():
    def __init__(self, parent) -> None:
        # self.base_layout = QHBoxLayout(parent)
        self.base_layout = QHBoxLayout()
        parent.setLayout(self.base_layout)

    def add_mpl_widget(self, canvas, gm_slider, gm_label, ct_slider, ct_label):
        mpl_layout = QVBoxLayout()
        gm_ctrl_layout = QHBoxLayout()
        ct_ctrl_layout = QHBoxLayout()
        gm_ctrl_layout.addWidget(gm_label.get_widget())
        gm_ctrl_layout.addWidget(gm_slider.get_widget())
        ct_ctrl_layout.addWidget(ct_label.get_widget())
        ct_ctrl_layout.addWidget(ct_slider.get_widget())
        mpl_layout.addWidget(canvas.get_widget())
        mpl_layout.addLayout(gm_ctrl_layout)
        mpl_layout.addLayout(ct_ctrl_layout)
        self.base_layout.addLayout(mpl_layout)

    def add_image_widget(self, img):
        self.base_layout.addWidget(img)


class EventControl():
    def __init__(self) -> None:
        pass

    def mpl_slider_changed(
            self, canvas, gm_slider, ct_slider, gm_label, ct_label, img):
        self.mpl_canvas = canvas
        self.gm_slider = gm_slider
        self.ct_slider = ct_slider
        self.mpl_label = gm_label
        self.ct_label = ct_label
        self.img = img
        gm_slider.set_slot(slot_func=self.slider_change_slot)
        ct_slider.set_slot(slot_func=self.slider_change_slot)

    def slider_change_slot(self):
        gm_value = self.gm_slider.get_value()
        color_temp = self.ct_slider.get_value()
        print(f"gamma = {gm_value}, color_temp = {color_temp}")
        self.mpl_label.set_label(gm_value)
        self.ct_label.set_label(color_temp)
        self.mpl_canvas.update_plot(gamma=gm_value)
        self.img.image_change(gamma=gm_value, color_temp=color_temp)


class MyImage():
    def __init__(
            self, numpy_img=np.ones((512, 512, 3), dtype=np.float32)) -> None:
        self.label = QLabel()
        self.np_img = numpy_img
        self.ndarray_to_qimage(self.np_img)
        self.label.setPixmap(QPixmap.fromImage(self.qimg))

    def ndarray_to_qimage(self, img):
        """
        Parameters
        ----------
        img : ndarray(float)
            image data. data range is from 0.0 to 1.0
        """
        self.uint8_img = np.uint8(np.round(np.clip(img, 0.0, 1.0) * 255))
        height, width = self.uint8_img.shape[:2]
        print(f"h={height}, w={width}")
        self.qimg = QImage(
            self.uint8_img.data, width, height, QImage.Format_RGB888)

        return self.qimg

    def image_change(self, gamma=2.2, color_temp=6500):
        linear_img = (self.np_img ** 2.4)
        dst_temperature_xy = CCT_to_xy(color_temp)
        # dst_temperature_xy = tpg.D65_WHITE
        xyz = RGB_to_XYZ(
            linear_img, tpg.D65_WHITE, dst_temperature_xy,
            RGB_COLOURSPACE_BT709.RGB_to_XYZ_matrix)
        linear_img = XYZ_to_RGB(
            xyz, dst_temperature_xy, dst_temperature_xy,
            RGB_COLOURSPACE_BT709.XYZ_to_RGB_matrix)
        img = np.clip(linear_img, 0.0, 1.0) ** (1/gamma)
        self.ndarray_to_qimage(img)
        self.label.setPixmap(QPixmap.fromImage(self.qimg))

    def get_widget(self):
        return self.label


class MyWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.resize(960, 540)
        layout = MyLayout(self)
        img = tpg.img_read_as_float("./img/sozai.png")
        # img = np.ones((512, 512, 3)) * np.array([1.0, 1.0, 0.0])
        my_image = MyImage(numpy_img=img)
        event_control = EventControl()
        mpl_gm_slider = GammaSlider()
        mpl_ct_slider = ColorTempSlider()
        mpl_gm_label = GammaLabel(mpl_gm_slider.get_default())
        mpl_ct_label = ColorTempLabel(mpl_ct_slider.get_default())
        mpl_canvas = MatplotlibTest(
            init_gamma=mpl_gm_slider.get_default())

        event_control.mpl_slider_changed(
            canvas=mpl_canvas, gm_slider=mpl_gm_slider,
            ct_slider=mpl_ct_slider, gm_label=mpl_gm_label,
            ct_label=mpl_ct_label, img=my_image)

        layout.add_mpl_widget(
            canvas=mpl_canvas, gm_slider=mpl_gm_slider, gm_label=mpl_gm_label,
            ct_slider=mpl_ct_slider, ct_label=mpl_ct_label)

        layout.add_image_widget(my_image.get_widget())


def main_func():
    app = QApplication([])
    widget = MyWidget()
    widget.show()
    sys.exit(app.exec_())


def matplot_qt5_test():
    app = QtWidgets.QApplication(sys.argv)
    wid = QtWidgets.QWidget()
    wid.resize(250, 150)
    grid = QtWidgets.QGridLayout(wid)
    fig = Figure(
        figsize=(7, 5), dpi=65, facecolor=(1, 1, 1), edgecolor=(0, 0, 0))
    ax1 = fig.add_subplot(111)
    x = np.linspace(0, 1, 1024)
    y = x ** 2.4
    ax1.plot(x, y)
    canvas = FigureCanvas(fig)
    grid.addWidget(canvas, 0, 0)
    wid.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    main_func()
    # a = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.uint8)
    # print(a.nbytes)
    # print(tpg.D65_WHITE)