import torch
import torch.nn as nn
import src.midi_processing as mp
import os

def generate_music(model, initial_sequence_str):
    initial_sequence = [int(note) for note in initial_sequence_str.split()]
    generated_sequence = model.generate_sequence(initial_sequence, length=200)
    decoded_sequence = mp.decode_midi_notes(generated_sequence)  # Используем функцию декодирования
    return mp.save_midi(decoded_sequence)  # Сохраняем MIDI файл




