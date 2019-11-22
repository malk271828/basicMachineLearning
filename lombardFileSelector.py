from glob import glob

BASE_DIR = "./"

class lombardFileSelector:
    def __init__(self, base_dir=BASE_DIR):
        self.base_dir = base_dir

    def getFileList(self,
                    domain:str,
                    verbose:int=0):
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