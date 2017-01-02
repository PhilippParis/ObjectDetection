class Rect:
    x = 0
    y = 0
    width = 0
    height = 0    
    
    # ============================================================= #

    @staticmethod
    def overlap(lhs, rhs):
        return max(0, min(lhs.x2(), rhs.x2()) - max(lhs.x, rhs.x)) * max(0, min(lhs.y2(), rhs.y2()) - max(lhs.y, rhs.y))
    
    
    # ============================================================= #

    @staticmethod
    def compare(lhs, rhs):
        delta = 0.2 * 0.5 * (min(lhs.width, rhs.width) + min(lhs.height, rhs.height))
    
        return  abs(lhs.x - rhs.x) <= delta and \
                abs(lhs.y - rhs.y) <= delta and \
                abs(lhs.x + lhs.width - rhs.x - rhs.width) <= delta and \
                abs(lhs.y + lhs.height - rhs.y - rhs.width) <= delta
        
    # ============================================================= #

    def __init__(self, x, y, width, height):
        self.x = x 
        self.y = y
        self.width = width
        self.height = height
        
    # ============================================================= #
    
    def center(self):
        return (self.x + self.width / 2, self.y + self.height / 2)
        
    # ============================================================= #
    
    def x2(self):
        return self.x + self.width
        
    # ============================================================= #
    
    def y2(self):
        return self.y + self.height
    
    # ============================================================= #
    
    def __add__(self, other):
        return Rect(self.x + other.x, self.y + other.y, self.width + other.width, self.height + other.height)
    
    # ============================================================= #

    def __iadd__(self, other):
        self.x += other.x
        self.y += other.y
        self.width += other.width
        self.height += other.height
        return self
    
    # ============================================================= #
    
    def __div__(self, other):
        return Rect(self.x / other, self.y / other, self.width / other, self.height / other)
    
    # ============================================================= #
    
    def __idiv__(self, other):
        self.x /= other
        self.y /= other
        self.width /= other
        self.height /= other
        return self
    # ============================================================= #
    
    def __str__(self):
        return str(self.x) + "/" + str(self.y) + " " + str(self.width) + "x" + str(self.height)
    
    # ============================================================= #

    def contains(self, other):
        return self.x <= other.x and self.x2() >= other.x2() and self.y <= other.y and self.y2() >= other.y2()
        
    
