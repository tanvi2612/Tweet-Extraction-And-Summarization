from imports import *
import pickle

model = torch.load("../../models/best_model_state.bin")
model.eval()