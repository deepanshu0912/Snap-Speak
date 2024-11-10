import torch
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as img_transforms
from torch.utils.tensorboard import SummaryWriter
from utils import save_checkpoint, load_checkpoint, print_examples
from get_loader import get_loader
from model import CNNtoRNN


def train_model():
    transform = img_transforms.Compose(
        [
            img_transforms.Resize((356, 356)),
            img_transforms.RandomCrop((299, 299)),
            img_transforms.ToTensor(),
            img_transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    train_loader, dataset = get_loader(
        root_folder="flickr8k/images",
        annotation_file="flickr8k/captions.txt",
        transform=transform,
        num_workers=2,
    )

    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    load_prev_model = False
    save_current_model = False
    fine_tune_CNN = False

    # Hyperparameters
    embed_dim = 256
    hidden_dim = 256
    vocab_len = len(dataset.vocab)
    num_lstm_layers = 1
    learning_rate = 3e-4
    epochs = 100

    # Initialize TensorBoard for monitoring
    writer = SummaryWriter("runs/flickr_captioning")
    global_step = 0

    # Initialize model, loss function, and optimizer
    model = CNNtoRNN(embed_dim, hidden_dim, vocab_len, num_lstm_layers).to(device)
    loss_fn = nn.CrossEntropyLoss(ignore_index=dataset.vocab.stoi["<PAD>"])
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Set CNN layers to be trainable or frozen based on fine-tuning flag
    for name, param in model.encoderCNN.inception.named_parameters():
        param.requires_grad = fine_tune_CNN if "fc" not in name else True

    # Load model checkpoint if specified
    if load_prev_model:
        global_step = load_checkpoint(torch.load("my_checkpoint.pth.tar"), model, optimizer)

    model.train()

    for epoch in range(epochs):
        # Uncomment the line below to test model captioning examples
        # print_examples(model, device, dataset)

        if save_current_model:
            checkpoint = {
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "step": global_step,
            }
            save_checkpoint(checkpoint)

        for idx, (images, captions) in tqdm(
            enumerate(train_loader), total=len(train_loader), leave=False
        ):
            images, captions = images.to(device), captions.to(device)

            # Forward pass
            output = model(images, captions[:-1])
            loss = loss_fn(output.reshape(-1, output.shape[2]), captions.reshape(-1))

            # Log training loss
            writer.add_scalar("Training Loss", loss.item(), global_step=global_step)
            global_step += 1

            # Backpropagation and optimization step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


if __name__ == "__main__":
    train_model()
