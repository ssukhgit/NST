import os
import torch
import torchvision.models as models

def main():
    models_dir = "models"
    
    os.makedirs(models_dir, exist_ok=True)

    print("Downloading VGG19 model...")
    vgg19 = models.vgg19(pretrained=True)

    model_path = os.path.join(models_dir, "vgg19.pth")

    print(f"Saving model to {model_path}...")
    torch.save(vgg19.state_dict(), model_path)
    
    print("VGG19 model has been saved successfully.")

if __name__ == "__main__":
    main()