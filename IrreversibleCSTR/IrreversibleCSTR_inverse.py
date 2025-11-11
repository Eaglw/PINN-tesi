""" 
qui voglio trovare i parametri del modello fisico a partire dal training, ma non ho ancora capito
benissimo come contestualizzarlo nel caso del CSTR

provvisorio copiato da oscillatore armonico
"""

torch.manual_seed(123)

# I am picking data on the entire domain
x_data_new = x[::2]
y_data_new = y[::2]

model = FCN(1, 1, 64, 5, nn.Tanh)

x_physics = torch.linspace(0, 1, 500).view(-1, 1).requires_grad_(True)

mu_train = nn.Parameter(torch.tensor(1.0, dtype=torch.float32, requires_grad=True))
k_train = nn.Parameter(torch.tensor(300.0, dtype=torch.float32, requires_grad=True))

#boundary conditions
single_point = x[0].unsqueeze(1)
single_point.requires_grad = True

# Optimizer for pretraining (only updates model parameters)
optimizer_model = torch.optim.Adam(model.parameters(), lr=1e-4)

# Lists to store the parameter updates
mu_history = []
k_history = []
files = []
# Pretraining loop
for epoch in range(6000):
    optimizer_model.zero_grad()

    # Compute losses (solution loss, boundary loss, etc.)
    loss1 = torch.mean((model(x_data_new) - y_data_new) ** 2)  # Data loss
    bc1 = model(single_point)
    bc2 = torch.autograd.grad(bc1, single_point, torch.ones_like(bc1), create_graph=True)[0]
    loss3 = torch.mean((bc1 - 1) ** 2)  # Boundary conditions (if any)
    loss4 = torch.mean(bc2 ** 2)

    loss_pretrain = loss1 + loss3 + loss4  # Pretraining loss

    loss_pretrain.backward()
    optimizer_model.step()

    if epoch % 100 == 0:
        print(f"Pretrain Epoch {epoch}, Loss: {loss_pretrain.item()}")

print("Pretraining finished. Now optimizing parameters...")

optimizer_full = torch.optim.Adam([
    {'params': model.parameters(), 'lr': 1e-4},
    {'params': [mu_train, k_train], 'lr': 1e-2}  # Higher learning rate for parameters
])

for i in range(10000000):
    optimizer_full.zero_grad()

    # Solution loss (already trained)
    loss1 = torch.mean((model(x_data_new) - y_data_new) ** 2)
    bc1 = model(single_point)
    bc2 = torch.autograd.grad(bc1, single_point, torch.ones_like(bc1), create_graph=True)[0]
    loss3 = torch.mean((bc1 - 1) ** 2)  # Boundary conditions (if any)
    loss4 = torch.mean(bc2 ** 2)

    # Compute physics loss
    yhp = model(x_physics)  # Predicted solution at physics points
    dx = torch.autograd.grad(yhp, x_physics, torch.ones_like(yhp), create_graph=True, retain_graph=True)[0]
    dx2 = torch.autograd.grad(dx, x_physics, torch.ones_like(dx), create_graph=True, retain_graph=True)[0]
    physics = dx2 + mu_train * dx + k_train * yhp
    loss2 = torch.mean(physics ** 2)  # Physics loss

    # the model tends to go into an overdumped state, therefore I am applying a soft constraint by regularization
    # this approach do not avoid entirely the overdamped state, because it strongly depends on the lambda_reg
    damping_constraint = torch.relu(mu_train - 2 * torch.sqrt(k_train)) ** 2
    lambda_reg = 0.1


    """
    plt.figure()
    plt.plot(x, y, label="Exact solution")
    plt.scatter(x_data_new, y_data_new, color="tab:orange", label="Training data")
    plt.scatter(x_data_new, model(x_data_new).detach().numpy(), color='red', label="Prediction")
    plt.legend()
    
    
    plt.show()
    """


    # Total loss
    loss = loss1 + loss2 + loss3 + loss4 + lambda_reg * damping_constraint

    loss.backward()
    optimizer_full.step()

    # Store the values of mu and k
    mu_history.append(mu_train.item())
    k_history.append(k_train.item())

    if i % 100 == 0:
        print(f"Epoch {i}, Loss: {loss.item()}, mu_train: {mu_train.item()}, k_train: {k_train.item()}"
                f", Solution Loss: {loss1.item()}, PDE Loss: {loss2.item()}, Dirichlet Loss: {loss3.item()}, Neumann Loss: {loss4.item()}"
                f", Damping Constraint: {damping_constraint.item()}")

    # plot the result as training progresses
    if (i + 1) % 2000 == 0:

        yh = model(x).detach()
        xp = x_physics.detach()

        plot_result(x, y, x_data_new, y_data_new, yh)

        file = "plots/pinn_inverse_%.8i.png" % (i + 1)
        plt.savefig(file, bbox_inches='tight', pad_inches=0.1, dpi=100, facecolor="white")
        files.append(file)

        if (i + 1) % 6000 == 0:
            plt.show()
        else:
            plt.close("all")




#save_gif_PIL("pinn_inverse.gif", files, fps=20, loop=0)


# Plot parameter evolution
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(mu_history, label="mu_train")
plt.axhline(y=2 * d, color="r", linestyle="--", label="True mu")
plt.xlabel("Iteration")
plt.ylabel("mu_train")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(k_history, label="k_train")
plt.axhline(y=w ** 2, color="r", linestyle="--", label="True k")
plt.xlabel("Iteration")
plt.ylabel("k_train")
plt.legend()

plt.tight_layout()
plt.show()

print("Training complete.")
