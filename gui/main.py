from PyQt5.QtWidgets import QMainWindow, QApplication

from gui.designer import Ui_MainWindow


class MainClass(Ui_MainWindow, QMainWindow):
    def __init__(self):
        super(MainClass, self).__init__()
        self.setupUi(self)


if __name__ == '__main__':
    app = QApplication([])
    vp = MainClass()
    vp.show()
    app.exec_()
