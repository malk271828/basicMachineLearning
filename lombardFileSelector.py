from glob import glob

BASE_DIR = "./"

class fileSelector:
    def __init__(self,
                 base_dir:str = BASE_DIR) -> list:
        self.base_dir = base_dir

    def getFileList(self):
        path = self.base_dir + "/*"
        return glob(path)

class lombardFileSelector(fileSelector):
    def __init__(self,
                 base_dir:str = BASE_DIR):
        super().__init__(base_dir)

    def getFileList(self,
                    domain:str,
                    verbose:int = 1):
        if domain == "visual":
            domain = "front"
            ext = "mov"
        else:
            ext = "wav"
        path = self.base_dir + domain + "/*." + ext
        g = glob(path)

        if verbose > 0:
            print("search pattern: {0}".format(path))
            print("{0} files have been detected.".format(len(g)))

        return g