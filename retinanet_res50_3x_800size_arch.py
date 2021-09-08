from detection import models

class CustomRetinaNetConfig(models.RetinaNetConfig):
    def __init__(self):
        super().__init__()

        # ------------------------ data cfg -------------------------- #
        self.train_dataset = dict(
            name="arch",
            root="arch",
            ann_file="arch_detection/train_arch.json",
            remove_images_without_annotations=True,
        )
        self.test_dataset = dict(
            name="arch",
            root="arch",
            ann_file="arch_detection/test_arch.json",
            remove_images_without_annotations=False,
        )
        self.num_classes = 1
        
        # ------------------------ training cfg ---------------------- #
        self.train_image_short_size = 800
        self.max_epoch = 36
        self.lr_decay_stages = [24, 32]
        self.nr_images_epoch = 1000
        self.warm_iters = 100
        self.log_interval = 10



Net = models.RetinaNet
Cfg = CustomRetinaNetConfig