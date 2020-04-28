class a:
    def __init__(self):
        super().__init__()
        self.n = None


aa = a()
bb = a()
print(aa,bb)
bb.n = aa
bb.n=None
aa.n=None
print(aa,bb)