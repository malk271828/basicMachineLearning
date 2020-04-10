from glob import glob

BASE_DIR = "./"

class fileSelector:
    def __init__(self,
                 base_dir:str = BASE_DIR) -> list:
        self.base_dir = base_dir

    def getFileList(self):
        path = self.base_dir + "/*"
        return glob(path)