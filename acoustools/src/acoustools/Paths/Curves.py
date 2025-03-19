class CubicBezier():

    def __init__(self, start, end, offset_1, offset_2):
        self.start = start
        self.end = end
        self.offset_1 = offset_1
        self.offset_2 = offset_2

        self.index = -1
    
    def __iter__(self):
        return self #Is this right?

    def __next__(self):
        self.index += 1

        if self.index == 0:
            return self.start
        elif self.index == 1:
            return self.end
        elif self.index == 2:
            return self.offset_1
        elif self.index == 3:
            return self.offset_2
        if self.index == 4:
            raise StopIteration
        

    def get_data(self):
        return self.start, self.end, self.offset_1, self.offset_2

    def __getitem__(self, i):
        return self.get_data()[i]
    
    def __len__(self):
        return 4
    
    def get_OptiSpline_parameters(self,start=False):
        params = [self.offset_1, self.offset_2]
        if start:
            params.append(self.end)
        return params
    

class Spline():

    def __init__(self,curves:list = None):
        if curves is None:
            curves = []
        self.curves = curves
        self.index = -1
    
    def __iter__(self):
        return iter(self.curves)
    
    def __next__(self):
        self.index += 1
        if self.index >= len(self.curves):
            raise StopIteration
        else:
            return self.curves[self.index]
        
    def __getitem__(self, i):
        return self.curves[i]
    
    def __len__(self):
        return len(self.curves)

    def add_curve(self,curve):
        self.curves.append(curve)
    
    def __setitem__(self, i, value):
        self.curves[i] = value
    
    def __str__(self):
        return "Spline: "  + str(self.curves)