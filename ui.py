import sys
import os
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QComboBox,
                             QTextEdit, QFileDialog, QLabel, QMessageBox)
from PyQt5.QtCore import Qt, QThread, pyqtSignal,QObject
from pre import Pre, Config
from PyQt5.QtGui import QFont
from hmm.main import pt_model_use
model_dic = {"bert_crf":"ptm_crf","bert_idcnn_crf":"ptm_idcnn_crf","bilstm_crf":"bilstm_crf","bert_bilstm_crf":"ptm_bilstm_crf","bilstm_crf":"bilstm_crf","bert_bp":"ptm_bp","bert_gp":"ptm_gp","hmm":"hmm"}


# 模型检测接口函数
def model_detection(text, model_name):
    model_name = model_dic[model_name]
    print(model_name)
    if model_name == "hmm":
        return pt_model_use(text)
    cfg = Config(model_name)
    cfg1 = cfg.configure
    if model_name == "ptm_gp" or model_name == "ptm_bp":
        cfg1["method"] = "span"
    predictor = Pre(cfg1)
    result = predictor.predict_one(text)
    return result


# 模型性能测试函数
def test_model_performance(test_path, model_name):
    model_name = model_dic[model_name]
    print(model_name)
    if model_name == "hmm":
        return "hmm模型性能测试功能暂未开放"
    cfg = Config(model_name, test_path)
    cfg1 = cfg.configure
    if model_name == "ptm_gp" or model_name == "ptm_bp":
        cfg1["method"] = "span"
    predictor = Pre(cfg1)
    return predictor.predict_test()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("命名实体识别系统")
        self.setGeometry(100, 100, 800, 600)

        self.central_widget = QWidget(self)
        self.setCentralWidget(self.central_widget)

        self.layout = QVBoxLayout(self.central_widget)

        # 菜单按钮
        self.menu_button_layout = QVBoxLayout()
        self.named_entity_detection_button = QPushButton("命名实体检测")
        self.model_performance_detection_button = QPushButton("模型性能检测")

        # 设置按钮字体和颜色（如果之前未设置）
        font = QFont()
        font.setPointSize(14)
        self.named_entity_detection_button.setFont(font)
        self.model_performance_detection_button.setFont(font)
        self.named_entity_detection_button.setStyleSheet("background-color: #4CAF50; color: white;")
        self.model_performance_detection_button.setStyleSheet("background-color: #2196F3; color: white;")

        self.menu_button_layout.addWidget(self.named_entity_detection_button)
        self.menu_button_layout.addWidget(self.model_performance_detection_button)
        self.layout.addLayout(self.menu_button_layout)

        # 连接按钮点击信号到相应槽函数
        self.named_entity_detection_button.clicked.connect(self.show_model_selection)
        self.model_performance_detection_button.clicked.connect(self.show_model_selection_for_performance)

        # 初始化其他界面相关变量
        self.model_name = None
        self.model_selection_widget = None
        self.detection_widget = None
        self.performance_widget = None

    def show_model_selection(self):
        if self.model_selection_widget is None:
            self.model_selection_widget = ModelSelectionWidget(self, self.show_detection_interface)
        self.model_selection_widget.show()
        self.named_entity_detection_button.hide()
        self.model_performance_detection_button.hide()

    def show_detection_interface(self, model_name):
        self.model_name = model_name
        print(self.model_name)
        if self.detection_widget is None:
            self.detection_widget = DetectionWidget(self, model_name)
        self.detection_widget.show()
        self.model_selection_widget.hide()

    def show_model_selection_for_performance(self):
        if self.model_selection_widget is None:
            self.model_selection_widget = ModelSelectionWidget(self, self.show_performance_interface)
        self.model_selection_widget.show()
        self.named_entity_detection_button.hide()
        self.model_performance_detection_button.hide()

    def show_performance_interface(self, model_name):
        self.model_name = model_name
        print(self.model_name)
        if self.performance_widget is None:
            self.performance_widget = PerformanceWidget(self, model_name)
        self.performance_widget.show()
        self.model_selection_widget.hide()

    def show_menu_buttons(self):
        self.model_selection_widget = None
        self.detection_widget = None
        self.performance_widget = None
        self.model_name = None
        self.named_entity_detection_button.show()
        self.model_performance_detection_button.show()


class ModelSelectionWidget(QWidget):
    def __init__(self, parent, next_interface_callback):
        super().__init__(parent)
        self.setWindowTitle("模型选择")
        self.setGeometry(200, 200, 400, 200)

        self.layout = QVBoxLayout(self)

        # 模型选择下拉框
        self.model_combo_box = QComboBox(self)
        self.model_combo_box.addItems(["bert_bilstm_crf", "bert_crf", "bilstm_crf", "bert_idcnn_crf", "bert_bp", "bert_gp", "hmm"])
        self.layout.addWidget(self.model_combo_box)

        # 确定按钮
        self.ok_button = QPushButton("确定")
        self.ok_button.clicked.connect(lambda: next_interface_callback(self.model_combo_box.currentText()))
        self.layout.addWidget(self.ok_button)

        # 返回按钮
        self.back_button = QPushButton("返回菜单")
        self.back_button.clicked.connect(self.return_to_menu)
        self.layout.addWidget(self.back_button)

    def return_to_menu(self):
        self.close()
        self.parent().show_menu_buttons()


class DetectionWidget(QWidget):
    def __init__(self, parent, model_name):
        super().__init__(parent)
        self.setWindowTitle("命名实体检测 - {}".format(model_name))
        self.setGeometry(100, 100, 600, 400)

        self.layout = QVBoxLayout(self)

        # 输入框
        self.input_text_edit = QTextEdit(self)
        self.layout.addWidget(self.input_text_edit)

        # 检测按钮
        self.detect_button = QPushButton("检测")
        self.detect_button.setFont(QFont("Arial", 12))
        self.detect_button.setStyleSheet("background-color: #4CAF50; color: white;")
        self.detect_button.clicked.connect(self.perform_detection)
        self.layout.addWidget(self.detect_button)

        # 输出框
        self.output_text_edit = QTextEdit(self)
        self.output_text_edit.setReadOnly(True)
        self.layout.addWidget(self.output_text_edit)

        # 返回按钮
        self.back_button = QPushButton("返回菜单")
        self.back_button.setFont(QFont("Arial", 12))
        self.back_button.setStyleSheet("background-color: #2196F3; color: white;")
        self.back_button.clicked.connect(self.return_to_menu)
        self.layout.addWidget(self.back_button)

        self.model_name = model_name

        # 序列标注模板展示（假设模板是固定的，这里简单显示一个示例）
        self.sequence_label_template_label = QLabel("序列标注模板示例：[B-PER, I-PER, O, B-LOC, I-LOC]", self)
        self.layout.addWidget(self.sequence_label_template_label)

        # 序列标注方法展示（根据选择的模型显示相应的标注方法）
        self.sequence_label_method_label = QLabel("", self)
        self.layout.addWidget(self.sequence_label_method_label)
        self.show_sequence_label_method(model_name)

    def perform_detection(self):
        text = self.input_text_edit.toPlainText()
        result = model_detection(text, self.model_name)
        self.output_text_edit.setPlainText(str(result))

    def return_to_menu(self):
        self.close()
        self.parent().show_menu_buttons()

    def show_sequence_label_method(self, model_name):
        if model_name == "bert_bilstm_crf" or model_name == "bilstm_crf":
            method_text = "使用BiLSTM - CRF进行序列标注"
        elif model_name == "bert_crf":
            method_text = "使用CRF进行序列标注"
        elif model_name == "bert_idcnn_crf":
            method_text = "使用IDCNN - CRF进行序列标注"
        elif model_name == "bert_gp" or model_name == "bert_bp":
            method_text = "使用特定方法（ptm_gp或ptm_bp相关）进行序列标注"
        else:
            method_text = "hmm的序列标注方法"
        self.sequence_label_method_label.setText(method_text)


class PerformanceWidget(QWidget):
    def __init__(self, parent, model_name):
        super().__init__(parent)
        self.setWindowTitle("模型性能检测 - {}".format(model_name))
        self.setGeometry(100, 100, 600, 400)

        self.layout = QVBoxLayout(self)

        # 模型选择显示
        self.model_label = QLabel("模型: {}".format(model_name), self)
        font = QFont()
        font.setPointSize(14)
        self.model_label.setFont(font)
        self.layout.addWidget(self.model_label)

        # 测试集选择按钮
        self.test_set_button = QPushButton("选择测试集")
        self.test_set_button.setFont(QFont("Arial", 12))
        self.test_set_button.setStyleSheet("background-color: #2196F3; color: white;")
        self.test_set_button.clicked.connect(self.select_test_set)
        self.layout.addWidget(self.test_set_button)

        # 测试集路径显示
        self.test_set_path_label = QLabel("", self)
        self.layout.addWidget(self.test_set_path_label)

        # 测试按钮
        self.test_button = QPushButton("测试")
        self.test_button.setFont(QFont("Arial", 12))
        self.test_button.setStyleSheet("background-color: #4CAF50; color: white;")
        self.test_button.clicked.connect(self.start_performance_test)
        self.layout.addWidget(self.test_button)

        # 性能结果显示
        self.performance_text_edit = QTextEdit(self)
        self.performance_text_edit.setReadOnly(True)
        self.layout.addWidget(self.performance_text_edit)

        # 返回按钮
        self.back_button = QPushButton("返回菜单")
        self.back_button.setFont(QFont("Arial", 12))
        self.back_button.setStyleSheet("background-color: #2196F3; color: white;")
        self.back_button.clicked.connect(self.return_to_menu)
        self.layout.addWidget(self.back_button)

        self.model_name = model_name
        self.test_set_path = ""
        # 用于接收测试结果的信号
        self.result_signal = pyqtSignal(str)
        # 测试线程
        self.test_thread = None

    def select_test_set(self):
        self.test_set_path, _ = QFileDialog.getOpenFileName(self, "选择测试集", "")
        if self.test_set_path:
            self.test_set_path_label.setText(self.test_set_path)

    def start_performance_test(self):
        if self.test_set_path:
            # 创建新线程并启动测试
            self.test_thread = QThread()
            self.worker = Worker(self.test_set_path, self.model_name)
            self.worker.moveToThread(self.test_thread)
            # 连接信号和槽
            self.worker.result_signal.connect(self.handle_result)
            self.test_thread.started.connect(self.worker.run)
            self.test_thread.start()

    def handle_result(self, result):
        # 处理测试结果并显示
        print(result, type(result))
        formatted_result = result.replace(";", "\n")
        if isinstance(result, str):
            # 保持原来的格式展示性能结果
            self.performance_text_edit.setPlainText(formatted_result)
        else:
            # 如果结果格式不符合预期，显示错误消息
            QMessageBox.critical(self, "错误", "模型性能测试结果格式不正确。")
        # 测试完成后清理线程资源
        self.test_thread.quit()
        self.test_thread.wait()

    def return_to_menu(self):
        self.close()
        self.parent().show_menu_buttons()


# 工作线程类
class Worker(QObject):
    result_signal = pyqtSignal(str)

    def __init__(self, test_path, model_name):
        super().__init__()
        self.test_path = test_path
        self.model_name = model_name

    def run(self):
        result = test_model_performance(self.test_path, self.model_name)
        self.result_signal.emit(result)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())