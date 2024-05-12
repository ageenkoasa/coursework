import torch
import torch.nn as nn
import numpy as np
import src.midi_processing as mp
import os

class MusicRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=2):
        super(MusicRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x, hidden):
        out, hidden = self.lstm(x, hidden)
        out = self.fc(out[:, -1, :])
        return out, hidden

    def init_hidden(self, batch_size):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return (torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device),
                torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device))

    def generate_sequence(self, initial_sequence, length=200):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        sequence = torch.tensor([initial_sequence], dtype=torch.float32).to(device)
        sequence = sequence.unsqueeze(0)
        
        hidden = self.init_hidden(1)
        
        generated_sequence = []
        for _ in range(length):
            output, hidden = self(sequence, hidden)
            generated_note = output.squeeze().detach().cpu().numpy()
            generated_sequence.append(generated_note)
            sequence = torch.tensor([generated_note], dtype=torch.float32).to(device)
            sequence = sequence.unsqueeze(0)
        
        return generated_sequence

def train_model(data_directory, num_epochs=100, batch_size=32, sequence_length=100):
    input_sequences, output_sequences = mp.load_data(data_directory, sequence_length)

    input_size = 2
    output_size = 2
    hidden_size = 128
    num_layers = 2

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = MusicRNN(input_size, hidden_size, output_size, num_layers).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(num_epochs):
        indices = np.random.randint(0, len(input_sequences), batch_size)
        input_batch = torch.tensor(input_sequences[indices], dtype=torch.float32).to(device)
        output_batch = torch.tensor(output_sequences[indices], dtype=torch.float32).to(device)
        
        hidden = model.init_hidden(batch_size)
        optimizer.zero_grad()
        
        outputs, _ = model(input_batch, hidden)
        loss = criterion(outputs, output_batch)
        
        loss.backward()
        optimizer.step()
        
        if (epoch+1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

    # Сохранение обученной модели
    save_path = 'models/music_rnn_model.pth'
    if not os.path.exists(save_path):
        torch.save(model.state_dict(), save_path)
    else:
        print("Файл уже существует. Перезапись невозможна.")


if __name__ == "__main__":
    train_model("C:/Users/ageenko/music_generator_app/audio_dataset")
