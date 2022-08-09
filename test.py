from model import test
import torch
import torch.nn as nn

device = "mps" if torch.has_mps else "cpu"
print(f"Using {device} device")

encoder = test().to(device)
print(encoder)
[w, b] = encoder.parameters()
print(w, b)

X = torch.tensor([[2], [3], [4]], dtype=torch.float32, device=device)
Y = torch.tensor([[4], [6], [8]], dtype=torch.float32, device=device)

loss = nn.MSELoss()
optimizater = torch.optim.SGD(encoder.linear.parameters(), lr=0.01)
# for name, param in encoder.named_parameters():
# print(f"Layer: {name} | Size: {param.size()} | Values : {param[:2]} \n")
for i in range(1000):
    y_est = encoder(X)
    l = loss(Y, y_est)
    l.backward()
    optimizater.step()
    optimizater.zero_grad()
    print(i)
    [w, b] = encoder.parameters()
    print(w,b)
