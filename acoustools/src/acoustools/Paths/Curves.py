# CURVES MUST DEFINE A get_data THAT RETURNS THE PARAMETERS FOR ITS CONSTRUCTOR SUCH THAT `curve==type(curve)(curve.get_data())`

class CubicBezier():
    '''
    Cublic Bezier class defined by `start, end, offset_1, offset_2` \n
    See https://www.desmos.com/calculator/gtngeffijw for interactive Bezier demo
    
    '''

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
    
    def __str__(self):
        return f"Bezier: Start{self.start}, End{self.end}, Offset_1{self.offset_1}, Offset_2{self.offset_2}"
    
    

class Spline():
    '''
    Container for combining curve objects into a spline
    '''

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
    
    def clone(self):
        new_curves = []
        for curve in self.curves:
            curve_type = type(curve)
            data = curve.get_data()
            data = [d.clone() for d in data]
            new = curve_type(*data)
            new_curves.append(new)
        return Spline(new_curves)