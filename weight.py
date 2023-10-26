import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = torch.load('.\\checkpointsteds_2\\bogie_2\wres50_bogie_2199.pth')

print(model)
