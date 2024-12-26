from PyQt5.QtWidgets import QApplication
from gui import MainWindow
import sys

def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    # This will only be called by the launcher after dependencies are installed
    main()