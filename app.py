from flask import Flask, request, jsonify
import torch, torch.nn as nn
from torchvision import transforms
import sqlite3
from PIL import Image
import io
from categories_config import categories
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

class SimpleCNN(nn.Module):
   def __init__(self, num_classes=5):
       super(SimpleCNN, self).__init__()
      
       self.conv_layers = nn.Sequential(
           nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
           nn.ReLU(),
           nn.MaxPool2d(kernel_size=2, stride=2),
           nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
           nn.ReLU(),
           nn.MaxPool2d(kernel_size=2, stride=2),
       )
      
       self.fc_layers = nn.Sequential(
           nn.Linear(32 * 56 * 56, 512),  # 56 is half of 224, assuming max pooling is applied twice
           nn.ReLU(),
           nn.Linear(512, num_classes)
       )
  
   def forward(self, x):
       x = self.conv_layers(x)
       x = x.view(x.size(0), -1)  # Flatten
       x = self.fc_layers(x)
       return x

# instantiate model
model = SimpleCNN()
# Load your trained model
model.load_state_dict(torch.load('recycling_model.pth'))
model.eval()  # Set the model to evaluation mode

def preprocess_image(image_bytes):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    
    image = Image.open(io.BytesIO(image_bytes))
    tensor = transform(image)
    return tensor.unsqueeze(0)

@app.route('/identify_waste', methods=['POST'])
def identify_waste():
    if 'file' not in request.files:
        return jsonify({'error': 'no file'}), 400

    file = request.files['file']

    # Convert image bytes to tensor
    img_bytes = file.read()

    # Safety check and exception handling
    try:
        tensor = preprocess_image(img_bytes)
        outputs = model(tensor)
        _, predicted = outputs.max(1)
        class_id = predicted.item()

    except Exception as e:
        return jsonify({'error': f"An error occurred during prediction: {str(e)}"}), 500

    # Adjust the index to match your database
    database_id = class_id + 1
    item_name = categories[class_id]

    try:
        item_id, steps = get_recycling_info(item_name)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

    return jsonify({
        'item_name': item_name,
        'steps': steps
    })



def get_recycling_info(item_name):
    with sqlite3.connect('recycling_info.db') as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT item_id FROM Items WHERE item_name=?", (item_name,))
        result = cursor.fetchone()
        
        if not result:
            raise ValueError(f"No item found with name: {item_name}")
        
        item_id = result[0]
        cursor.execute("SELECT step_description FROM RecyclingSteps WHERE item_id=?", (item_id,))
        steps = [step[0] for step in cursor.fetchall()]

    return item_id, steps

if __name__ == '__main__':
    app.run(debug=True)
