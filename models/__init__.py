from .tfr import build,  build_BubbleFormer_all #Transfloormer

def build_model(args):
    return build(args)

def build_BubbleFormer(args):
    return build_BubbleFormer_all(args)
