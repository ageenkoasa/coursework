from flask import Flask, render_template, request
from src.model_rnn import MusicRNN
from src.music_generator import generate_music
import torch

app = Flask(__name__)

# Загрузка модели
model_path = "models/music_rnn_model.pth"
input_size = 2
hidden_size = 128
output_size = 2
num_layers = 2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = MusicRNN(input_size, hidden_size, output_size, num_layers).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

@app.route('/', methods=['GET', 'POST'])
def index():
    generated_music_path = None
    
    if request.method == 'POST':
        # Получение данных из формы
        initial_sequence = request.form['initial_sequence']
        
        # Генерация музыки
        generated_music_path = generate_music(model, initial_sequence)
        
    return render_template('index.html', generated_music_path=generated_music_path)

if __name__ == '__main__':
    app.run(debug=True)

