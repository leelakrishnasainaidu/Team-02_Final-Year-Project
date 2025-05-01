import os
import torch
import torch.nn as nn
from torch.optim import AdamW
from torchvision.utils import save_image
from datetime import datetime
from utils.helpers import unnormalize
from utils.loss import ContentLoss, AdversialLoss
from utils.transforms import get_pair_transforms
from utils.datasets import get_dataloader
from models.discriminator import Discriminator
from models.generator import Generator

torch.backends.cudnn.benchmark = True

# Total Variation Loss for smoothness
class TVLoss(nn.Module):
    def __init__(self, weight=1.0):
        super(TVLoss, self).__init__()
        self.weight = weight

    def forward(self, x):
        batch_size = x.size()[0]
        h_tv = torch.pow(x[:, :, 1:, :] - x[:, :, :-1, :], 2).sum()
        w_tv = torch.pow(x[:, :, :, 1:] - x[:, :, :, :-1], 2).sum()
        return self.weight * 2 * (h_tv + w_tv) / batch_size

def train():
    torch.manual_seed(1337)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Configs
    batch_size = 16
    image_size = 256
    learning_rate = 1e-4
    beta1, beta2 = (0.5, 0.999)
    weight_decay = 1e-5
    epochs = 1000

    # Models
    netD = Discriminator().to(device)
    netG = Generator().to(device)
    netG.load_state_dict(torch.load("./checkpoints/pretrained_netG.pth", map_location=device))

    optimizerD = AdamW(netD.parameters(), lr=learning_rate, betas=(beta1, beta2), weight_decay=weight_decay)
    optimizerG = AdamW(netG.parameters(), lr=learning_rate, betas=(beta1, beta2), weight_decay=weight_decay)
    scaler = torch.cuda.amp.GradScaler()

    cartoon_labels = torch.ones(batch_size, 1, image_size // 4, image_size // 4).to(device)
    fake_labels = torch.zeros(batch_size, 1, image_size // 4, image_size // 4).to(device)

    # Loss functions
    content_loss = ContentLoss().to(device)
    adv_loss = AdversialLoss(cartoon_labels, fake_labels).to(device)
    bce_loss = nn.BCEWithLogitsLoss().to(device)
    tv_loss = TVLoss(weight=1e-5).to(device)  # Adjust weight as needed

    real_dataloader = get_dataloader("D:/cartoon-gan/datasets/real_images/flickr30k_images/", image_size, batch_size)
    cartoon_dataloader = get_dataloader("./datasets/cartoon_images_smoothed/Studio Ghibli", image_size, batch_size, get_pair_transforms(image_size))

    tracked_images = next(iter(real_dataloader)).to(device)
    os.makedirs("images", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)

    G_losses, D_losses = [], []
    iters = 0

    print("Starting Training Loop...")
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        for i, (cartoon_edge_data, real_data) in enumerate(zip(cartoon_dataloader, real_dataloader)):
            netD.zero_grad()
            cartoon_data = cartoon_edge_data[:, :, :, :image_size].to(device)
            edge_data = cartoon_edge_data[:, :, :, image_size:].to(device)
            real_data = real_data.to(device)

            for param in netD.parameters():
                param.requires_grad = True

            with torch.cuda.amp.autocast():
                fake_images = netG(real_data)
                cartoon_pred = netD(cartoon_data)
                edge_pred = netD(edge_data)
                fake_pred = netD(fake_images.detach())
                errD = adv_loss(cartoon_pred, fake_pred, edge_pred)

            scaler.scale(errD).backward()
            scaler.step(optimizerD)

            netG.zero_grad()
            for param in netD.parameters():
                param.requires_grad = False

            with torch.cuda.amp.autocast():
                fake_pred = netD(fake_images)
                content = content_loss(fake_images, real_data)
                adversarial = bce_loss(fake_pred, cartoon_labels)
                smoothness = tv_loss(fake_images)
                errG = 1.0 * content + 1.0 * adversarial + smoothness

            scaler.scale(errG).backward()
            scaler.step(optimizerG)
            scaler.update()

            G_losses.append(errG.item())
            D_losses.append(errD.item())

            if iters % 200 == 0:
                with torch.no_grad():
                    preview = netG(tracked_images)
                    save_path = f"images/{epoch}_{i}.png"
                    save_image(unnormalize(preview), save_path, normalize=True)
                    print(f"Saved preview to {save_path}")

            if iters % 1000 == 0:
                torch.save(netG.state_dict(), f"checkpoints/netG_epoch{epoch}_iter{iters}.pth")
                torch.save(netD.state_dict(), f"checkpoints/netD_epoch{epoch}_iter{iters}.pth")

            iters += 1

if __name__ == "__main__":
    train()
