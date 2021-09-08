from detection import models

class CustomRetinaNetConfig(models.RetinaNetConfig):
    def __init__(self):
        super().__init__()

        # ------------------------ data cfg -------------------------- #
        self.train_dataset = dict(
            name="cocomini",
            root="train",
            ann_file="annotations/cocomini.json",
            remove_images_without_annotations=True,
        )
        self.test_dataset = dict(
            name="cocomini",
            root="val2017",
            ann_file="annotations/instances_val2017.json",
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

print(__file__)
print("version", sys.version)
Net = models.RetinaNet
Cfg = CustomRetinaNetConfig