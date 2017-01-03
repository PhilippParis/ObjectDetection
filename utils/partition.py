class TreeNode:
    parent = None
    label = -1
    children = []
    
    def __init__(self, parent):
        self.parent = parent
        self.label = -1
        
    def root(self):
        if self.parent != None:
            return self.parent.root()
        return self
    
    def rank(self):
        if self.parent == None:
            return 0
        return self.parent.rank() + 1
    
    def compress(self):
        if self.parent != None:
            self.parent = self.parent.compress()
            return self.parent
        return self

# ============================================================= #

def partition(objects, compare):
    labels = []
    forest = []
    nclasses = 0
    N = len(objects)
    
    # create forest of single element trees
    for i in xrange(N):
        forest.append(TreeNode(None))
    
    # compare elements
    for i in xrange(N):
        root1 = forest[i].root()
    
        for j in xrange(N):
            if i == j or not compare(objects[i], objects[j]):
                continue
        
            root2 = forest[j].root()
            
            # merge trees
            if root1 != root2:
                if root1.rank() < root2.rank():
                    root1.parent = root2
                else:
                    root2.parent = root1
                    
                # compress trees / reduce depth
                forest[i].compress()
                forest[j].compress()
          
    # create labels
    for i in xrange(N):
        root = forest[i].root()
        
        if root.label == -1:
            root.label = nclasses
            nclasses += 1
        
        labels.append(root.label)
    return labels, nclasses
                
                

    
    

