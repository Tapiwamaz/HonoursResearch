# Recreate the my_spectra array
class SpectrumArray():
    def __init__(self,intensities,width,height,mzs=None):

        if width * height != len(intensities):
            raise ValueError(f"Dimension mismatch: width({width}) * height({height}) = {width*height} != len(intensities)({len(intensities)})")
        
 
        self.intensities = intensities
        self.width = width
        self.height = height
        self.mzs = mzs if mzs is not None else list(range(100,1501,1))

        # CREATE A COORDINATES ARRAY
        self.pixels = []
        for y in height:
            for x in width:
                self.pixels.append((x,y))
        pass

    def __call__(self, *args, **kwds):
        pass