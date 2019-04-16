
class Mydict(dict,):
    def __init__(self,RBM):
        self.RBM = RBM
    def v_update(self, key, value):
        self.update({key:value})
        setattr(self.RBM, key, value)