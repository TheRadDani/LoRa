import torch
import torch.nn.functional as F

n = 10000
d_in = 1001
d_out = 1000

thethas = torch.randn(d_in,d_out)

X = torch.randn(n,d_in)
y = torch.matmul(X,thethas)
print(y.size())

# defining base model
model = torch.nn.Linear(d_in,d_out,bias=False)
#model parameters
print(model.parameters())


#function to train the model 

def train(model,X,y,batch_size = 128, epochs=100):
    
    # Adam optimizer
    optimizer = torch.optim.Adam(model.parameters())
    # loss function class defined as mse criterion
    criterion = torch.nn.MSELoss()
    
    n = len(X)  # Total number of samples

    for epoch in range(epochs):
        for i in range(0, n, batch_size):
            # Generate mini-batches
            indices = torch.randint(0, n, (batch_size,))
            mini_batch_X = X[indices]
            mini_batch_y = y[indices]

            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass
            predictions = model(mini_batch_X)

            # Compute the MSE loss
            loss = criterion(predictions, mini_batch_y)

            # Backpropagation and optimization
            loss.backward()
            optimizer.step()


        if (epoch + 1) % 10 == 0:
            # Compute and print the loss for the entire dataset
            with torch.no_grad():
                predictions = model(X)
                loss = criterion(predictions, y)
                print(f"{epoch+1} Loss: {loss.item()}\n")

train(model,X,y)

# modify parameter distribution
thethas2 = thethas + 1

X2 = torch.randn(n,d_in)
y2 = torch.matmul(X2,thethas2)

# apply out base model to this distribution 
criterion = torch.nn.MSELoss()
modified_loss = criterion(model(X2),y2)
print(modified_loss.item())

# fine tune initial model

# Implementation of LoRa logic

class LoRALinear(torch.nn.Module):
    def __init__(self, linear, r=16, alpha=1):
        super(LoRALinear, self).__init__()
        self.linear = linear
        
        # Freeze parameters of the original linear layer
        for param in self.linear.parameters():
            param.requires_grad = False
        
        # Initialize low-rank parameters A and B
        self.A = torch.nn.Parameter(torch.randn(linear.in_features, r))
        self.B = torch.nn.Parameter(torch.zeros(r, linear.out_features))
        
        # Scaling constant
        self.scaling = alpha / r

    def forward(self, x):
        # Original linear transformation
        linear_result = self.linear(x)
        
        # Low-rank adaptation
        adaptation = torch.matmul(x, torch.matmul(self.A, self.B)) * self.scaling
        
        # Combine original and adapted results
        output = linear_result + adaptation
        return output

lora = LoRALinear(model, r = 1)
train(lora,X2,y2)

delta_theta = torch.matmul(lora.A,lora.B)*lora.scaling
print(delta_theta[1:5,1:5])

with torch.no_grad():
    model.weight.add_(delta_theta.t())

print(criterion(model(X2),y2))