"""
Contain function to load a model and make a prediction on target image.
"""
import torch
import torchvision
import argparse
import matplotlib.pyplot as plt
import model_builder

# Creating a parser
parser = argparse.ArgumentParser()

# Get an image path
parser.add_argument("--image_path",
                    type=str,
                    help="target image filepath to predict on")
# Get a model path
parser.add_argument("--model_path",
                    default="models/tinyvgg_model.pth",
                    type=str,
                    help="target model filepath to use for prediction")
args = parser.parse_args()

# Setup class names
class_names = ["pizza", "steak", "sushi"]

# Setup device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Get the image path
IMG_PATH = args.image_path
MODEL_PATH = args.model_path
print(f"[INFO] Predicting on {IMG_PATH}")

# Function to load in the model
def load_model(filepath=MODEL_PATH):
  # Need to use the same hyperparameter as saved model
  model = model_builder.TinyVGG(input_shape=3,
                                hidden_units=128,
                                output_shape=3).to(device)

  print(f"[INFO] Loading in model from: {filepath}")
  # Load in the saved model state dictionary from file
  model.load_state_dict(torch.load(filepath))
  return model

# Function to load in model and predict on target image
def predict_on_image(image=IMG_PATH, filepath=MODEL_PATH):
  # Load the model
  model = load_model(filepath)

  # Load in the image and turn it into type torch.float32
  image = torchvision.io.read_image(str(IMG_PATH)).type(torch.float32)

  # Preprocess the image to get it between 0 and 1
  image = image / 255.

  # Resize the image to be the same size as the model
  transform = torchvision.transforms.Resize(size=(64, 64), antialias=True)
  image = transform(image)

  # Add batch dimension to image
  image = image.unsqueeze(dim=0)

  # Predict on image
  model.eval()
  with torch.inference_mode():
    # Put image to target device
    image = image.to(device)

    # Get pred logits
    logits = model(image)

    # Get pred probs
    pred_probs = logits.softmax(dim=1)

    # Get pred labels
    pred_label = torch.argmax(pred_probs, dim=1)
    pred_label_class = class_names[pred_label]

  # Put image to cpu
  image = image.cpu()
  # Plot the image alongsize the prediction and prediction probability

  plt.imshow(image.squeeze().permute(1, 2, 0)) # Remove batch dimension and rearrage dimension

  if (class_names):
    title = f"Pred: {class_names[pred_label.cpu()]} | Prob: {pred_probs.max().cpu():.3f}"
  else:
    title = f"Pred: {pred_label.cpu()} | Prob: {pred_probs.max().cpu():.3f}"
  plt.title(title)
  plt.axis(False)
  plt.show()
  #print(f"[INFO] Pred class: {pred_label_class}, Pred prob: {pred_probs.max():.3f}")

if __name__ == "__main__":
  predict_on_image()
