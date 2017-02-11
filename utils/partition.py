class TreeNode:
    parent = None
    label = -1
    children = []
    rank = 0
    
    def __init__(self, parent):
        self.parent = parent
        self.label = -1
        
    def root(self):
        if self.parent != None:
            return self.parent.root()
        return self
    
    
    def compress(self):
        if self.parent != None:
            self.parent = self.parent.compress()
            return self.parent
        return self

# ============================================================= #

def partition(objects, compare):
    """
    partitions the elements in 'objects' into n classes each class
    containing all elements which are equivalent (= function 'compare' returns true)
    
    Args:
        objects: list of objects to partition
        compare: comparator method
        
    Returns:
        labels: list of size len(objects); labels[i] = class label of object[i]
        nclasses: number of classes
    """
    
    labels = []
    forest = []
    nclasses = 0
    N = len(objects)
    
    # create forest of single element trees
    for i in xrange(N):
        forest.append(TreeNode(None))
    
    t = 0
    # compare elements
    for i in xrange(N):
        
        for j in xrange(N):
            if i == j or not compare(objects[i], objects[j]):
                continue
        
            root1 = forest[i].root()
            root2 = forest[j].root()
            
            # merge trees
            if root1 != root2:
                if root1.rank < root2.rank:
                    root1.parent = root2
                elif root1.rank > root2.rank:
                    root2.parent = root1
                else:
                    root2.parent = root1
                    root1.rank += 1
                    
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
                
                

    
    

