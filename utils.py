import torch
import torchvision.transforms as custom_transforms
from PIL import Image

def display_examples(model, device, vocab_source):
    preprocess = custom_transforms.Compose(
        [
            custom_transforms.Resize((299, 299)),
            custom_transforms.ToTensor(),
            custom_transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    model.eval()
    # Example images for model testing
    sample_imgs = [
        ("dog.jpg", "Dog on a beach by the ocean"),
        
    ]
    
    for idx, (img_path, correct_caption) in enumerate(sample_imgs, 1):
        img = preprocess(Image.open(img_path).convert("RGB")).unsqueeze(0)
        generated_caption = " ".join(model.caption_image(img.to(device), vocab_source.vocab))
        print(f"Example {idx} CORRECT: {correct_caption}")
        print(f"Example {idx} OUTPUT: {generated_caption}")
    
    model.train()


def checkpoint_save(state, filepath="checkpoint.pth.tar"):
    print("=> Saving model state to file")
    torch.save(state, filepath)


def checkpoint_load(checkpoint, model, optimizer):
    print("=> Loading model state from file")
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    current_step = checkpoint["step"]
    return current_step
