def update_namespace(args, cfg_dict):
    for k, v in cfg_dict.items():
        setattr(args, k, v)


class CMAPSS():
    def __init__(self):
        super(CMAPSS, self)
        self.sequence_len = 50
        self.input_channels = 14

        self.shuffle = True
        self.drop_last = False


class NCMAPSS():
    def __init__(self):
        super(NCMAPSS, self)
        self.sequence_len = 50
        self.input_channels = 20

        self.shuffle = True
        self.drop_last = False
        self.normalize = False
