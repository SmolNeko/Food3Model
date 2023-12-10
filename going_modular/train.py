"""
Train a PyTorch image classification model using device-agnostic code.
"""
import os
import argparse
import torch
import data_setup, engine, model_builder, utils

from torchvision import transforms

# Create a parser
parser = argparse.ArgumentParser(description="Get some hyperparameters.")

# Create an arg for num_epochs
parser.add_argument("--num_epochs",
                    default=10,
                    type=int,
                    help="number of epochs to train for")
# Create an arg for batch_size
parser.add_argument("--batch_size",
                     default=32,
                     type=int,
                     help="number of samples per batch")

# Create an arg for hidden_units
parser.add_argument("--hidden_units",
                    default=10,
                    type=int,
                    help="number of hidden units in hidden layers")

# Create an arg for learning_rate
parser.add_argument("--learning_rate",
                    default=0.001,
                    type=float,
                    help="learning rate to use for model")

# Create an arg for training directory
parser.add_argument("--train_dir",
                    default="data/pizza_steak_sushi/train",
                    type=str,
                    help="directory file path to training data in standard image classification format")

# Create an arg for testing directory
parser.add_argument("--test_dir",
                    default="data/pizza_steak_sushi/test",
                    type=str,
                    help="directory file path to testing data in standard image classification format")


# Create an arg for early_stopper_patience
parser.add_argument("--early_stopper_patience",
                    default=3,
                    type=int,
                    help="integer indicate how many try before early stopping")

# Create an arg for early_stopper_rate
parser.add_argument("--early_stopper_rate",
                    default=0.01,
                    type=float,
                    help="float indicate the wiggle room for early stopping decision")

# Get arguments from the parser
args = parser.parse_args()

# Setup hyperparameters
NUM_EPOCHS = args.num_epochs
BATCH_SIZE = args.batch_size
HIDDEN_UNITS = args.hidden_units
LEARNING_RATE = args.learning_rate
EARLY_STOPPER_PATIENCE = args.early_stopper_patience
EARLY_STOPPER_RATE = args.early_stopper_rate
print(f"[INFO] Training a model for {NUM_EPOCHS} epochs with batch size {BATCH_SIZE} using {HIDDEN_UNITS} hidden units and a learning rate of {LEARNING_RATE}")
print(f"[INFO] EarlyStopper setting: patience={EARLY_STOPPER_PATIENCE}, min_delta={EARLY_STOPPER_RATE}")

# Setup directories
train_dir = args.train_dir
test_dir = args.test_dir
print(f"[INFO] Training data file: {train_dir}\n[INFO] Testing data file: {test_dir}")

# Setup target device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Create transforms
data_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])

# Create DataLoaders with help from data_setup.py
train_dataloader, test_dataloader, class_names = data_setup.create_dataloaders(
    train_dir=train_dir,
    test_dir=test_dir,
    transform=data_transform,
    batch_size=BATCH_SIZE
)

# Create model with help from model_builder.py
model = model_builder.TinyVGG(
    input_shape=3,
    hidden_units=HIDDEN_UNITS,
    output_shape=len(class_names)
).to(device)

# Setup loss function and optimizer
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),
                             lr=LEARNING_RATE)

# Start training with help from engine.py
engine.train(model=model,
             train_dataloader=train_dataloader,
             test_dataloader=test_dataloader,
             loss_fn=loss_fn,
             optimizer=optimizer,
             epochs=NUM_EPOCHS,
             device=device,
             early_stopper_patience=EARLY_STOPPER_PATIENCE,
             early_stopper_rate=EARLY_STOPPER_RATE)

# Save the model with help from utils.py
utils.save_model(model=model,
                 target_dir="models",
                 model_name="tinyvgg_model.pth")

