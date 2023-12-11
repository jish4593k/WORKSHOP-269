import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from tensorflow.keras.preprocessing.text import Tokenizer

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.fc = nn.Linear(28 * 28, 10)  # Example linear layer for image classification

    def forward(self, x):
        x = x.view(-1, 28 * 28)  # Flatten the input image
        x = self.fc(x)
        return x

class GUIApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Neural Network GUI")

        # Neural network initialization
        self.neural_network = NeuralNetwork()
        self.optimizer = optim.SGD(self.neural_network.parameters(), lr=0.01)
        self.loss_criterion = nn.CrossEntropyLoss()

        # GUI components
        self.label = ttk.Label(root, text="Sample Image:")
        self.label.grid(row=0, column=0, padx=10, pady=10)

        self.canvas = tk.Canvas(root, width=200, height=200)
        self.canvas.grid(row=1, column=0, padx=10, pady=10)

        self.predict_button = ttk.Button(root, text="Predict", command=self.predict)
        self.predict_button.grid(row=2, column=0, padx=10, pady=10)

    def load_image(self, image_path):
        image = Image.open(image_path)
        image = image.resize((200, 200), Image.ANTIALIAS)
        photo = ImageTk.PhotoImage(image)
        self.canvas.config(width=200, height=200)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=photo)
        self.canvas.image = photo

    def predict(self):
        
        image_path = 'sample_image.jpg'
        self.load_image(image_path)

        
        transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
        ])
        image = Image.open(image_path)
        image = transform(image)

      
        with torch.no_grad():
            output = self.neural_network(image.unsqueeze(0))
            predicted_class = torch.argmax(output).item()

        
        result_text = f"Predicted Class: {predicted_class}"
        print(result_text)

if __name__ == "__main__":
    root = tk.Tk()
    app = GUIApp(root)
    root.mainloop()
