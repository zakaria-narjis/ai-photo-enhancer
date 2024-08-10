ENCODING_DEVICE = "cuda"


class Extractor:
    """
    Features extractor boilerplate
    """

    def __init__(
        self,
    ):
        self.name = ""
        self.input_shape = None
        self.output_shape = None
        self.model = None
        self.preprocess = None
        self.device = None


class ResnetEncoder(Extractor):
    def __init__(self):
        super().__init__()
        pass
