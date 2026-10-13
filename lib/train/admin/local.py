class EnvironmentSettings:
    """Project paths; this class is also serialized in the supplied checkpoint."""

    def __init__(self):
        self.workspace_dir = '/home/gaob/Dual-RGBT'
        self.tensorboard_dir = self.workspace_dir + '/runs/tensorboard'
        self.pretrained_networks = '/home/gaob'
        self.lasher_dir = self.workspace_dir + '/data/LasHeR/train'
