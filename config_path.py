
import socket
import getpass

class PathConfig:
    """
    A simple class to detect the current cluster and user,
    then provide the correct data path for each scenario.
    """
    def __init__(self):
        self.username = getpass.getuser() # e.g. saiyuel; toolkit
        self.hostname = socket.gethostname().lower() # e.g. nerds06; nerds07
        self.config={
            "saiyuel":{
                "nerds06":{
                    "imagenet_val": "/scratch/saiyuel/data/cache/autoencoders/data/ILSVRC2012_validation/data",
                    "model_ckpt": "/scratch/saiyuel/github/carlini-drs/imagenet/256x256_diffusion_uncond.pt"
                },
                "nerds07":{
                    "imagenet_val": "/data1/saiyuel/cache/autoencoders/data/ILSVRC2012_validation/data",
                    "model_ckpt": "/data1/saiyuel/projects/diffusion-ars/imagenet/256x256_diffusion_uncond.pt"
                },
                "nerds10":{
                    "imagenet_val": "/data1/saiyuel/cache/autoencoders/data/ILSVRC2012_validation/data",
                    "model_ckpt": "/data1/saiyuel/github/DensePure/256x256_diffusion_uncond.pt"
                },
                "aurora4.cs.ubc.ca":{
                    "imagenet_val": "/scratch1/data/cache/autoencoders/data/ILSVRC2012_validation/data",
                    "model_ckpt": "/scratch1/diffusion-ars/models/256x256_diffusion_uncond.pt"
                }
            },
            "toolkit":{
                "imagenet_val": "/mnt/home/ILSVRC2012_validation/data",
                "model_ckpt": "/mnt/home/diffusion-ars/imagenet/256x256_diffusion_uncond.pt"
            },
            "ubuntu":{
                "imagenet_val": "/home/ubuntu/imagenet_val",
                "model_ckpt": "/home/ubuntu/256x256_diffusion_uncond.pt"
            }
        }

    def get_imagenet_val_path(self):
        if self.username == "toolkit": 
            return self.config["toolkit"]["imagenet_val"]
        elif self.username == "ubuntu":
            return self.config["ubuntu"]["imagenet_val"]
        elif self.username == "saiyuel": 
            return self.config["saiyuel"][self.hostname]["imagenet_val"]
        else :
            # submit non-interactive jobs to toolkit 
            return self.config["toolkit"]["imagenet_val"]
            # raise Exception("username or hostname not found in config")
    
    def get_model_path(self):
        if self.username == "toolkit": 
            return self.config["toolkit"]["model_ckpt"]
        elif self.username == "ubuntu":
            return self.config["ubuntu"]["model_ckpt"]
        elif self.username == "saiyuel": 
            return self.config["saiyuel"][self.hostname]["model_ckpt"]
        else :
            # submit non-interactive jobs to toolkit 
            return self.config["toolkit"]["model_ckpt"]
            # raise Exception("username or hostname not found in config")


# def main():
#     config = PathConfig()
#     imagenet_path = config.get_imagenet_path()
#     model_path = config.get_model_path()

#     print(imagenet_path)
#     print(model_path)

# if __name__ == "__main__":
#     main()
