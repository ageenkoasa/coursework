import os
import numpy as np
import mido

def read_midi(file_path):
    midi = mido.MidiFile(file_path)
    notes = []
    for msg in midi:
        if msg.type == 'note_on':
            note = msg.note
            velocity = msg.velocity
            notes.append((note, velocity))
    return notes

def prepare_sequences(notes, sequence_length=100):
    sequence_input = []
    sequence_output = []
    for i in range(len(notes) - sequence_length):
        sequence_input.append(notes[i:i + sequence_length])
        sequence_output.append(notes[i + sequence_length])
    return np.array(sequence_input), np.array(sequence_output)

def load_data(directory, sequence_length=100):
    notes = []
    for file in os.listdir(directory):
        if file.endswith(".mid") or file.endswith(".midi"):
            file_path = os.path.join(directory, file)
            notes += read_midi(file_path)
    sequence_input, sequence_output = prepare_sequences(notes, sequence_length)
    return sequence_input, sequence_output

def encode_midi_notes(notes):
    # Преобразование списка нот в одномерный массив индексов
    encoded_notes = []
    for note in notes:
        encoded_notes.append(note[0])  # Берем только ноту, игнорируем скорость
    return np.array(encoded_notes)

def decode_midi_notes(encoded_notes):
    decoded_notes = []
    for note in encoded_notes:
        decoded_notes.append((note, 64))  # Устанавливаем произвольную скорость
    return decoded_notes
