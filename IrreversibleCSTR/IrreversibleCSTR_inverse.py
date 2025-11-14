"""
In questo caso do alla rete tutti i dati simulati possibili, e cerco di stimare i parametri fisici che mi 
permettono di rispettare il mass balance. 
Non so se è necessario ma sto raddoppiando la dimensione della rete, e anche step di train
In questo caso voglio stimare k reaction rate e tau=F/V residence time.
"""
# I am picking data on the entire domain
x_data_new = x[::2]
y_data_new = y[::2]
# Aumento i punti della fisica rispetto ai casi precedenti
x_physics = torch.linspace(0,5,100).view(-1,1).requires_grad_(True)# sample locations over the problem domain
#creo gli oggetti torch per i parametri fisici, con valore iniziale
#considerando che k è 1 e tau=F/V=400/2000=0.2, scelgo valori diversi ma stessa magnitude
tau_train = nn.Parameter(torch.tensor(0.5, dtype=torch.float32, requires_grad=True))
k_train = nn.Parameter(torch.tensor(5.0, dtype=torch.float32, requires_grad=True))

torch.manual_seed(123)
#boundary conditions, servono davvero in questo caso?
single_point = x[0].unsqueeze(1)
single_point.requires_grad = True

#aumento i layer e neuroni per layer, uso anche tanh come activation
model = FCN(1,1,64,5) #gli ho tolto nn.Tanh ma non so perchè non dovrebbe fungere 
optimizer = torch.optim.Adam(model.parameters(),lr=1e-4)
files = []
tau_history = []
k_history = []
#per il pretraining uso solo dati simulati e BC
for i in tqdm(range(step), desc="Pretraining on data"):
    optimizer.zero_grad()
    loss1 = torch.mean((model(x_data_new) - y_data_new) ** 2)
    # niente physic loss in questa fase
    # BC loss
    bc1 = model(single_point)
    loss3 = torch.mean((bc1-10) ** 2) #credo che questo sia cA(t=0)=10
    # backpropagate joint loss
    loss_pretrain = loss1+loss3 #add two loss terms together
    loss_pretrain.backward()
    optimizer.step()

optimizer_full = torch.optim.Adam([
    {'params': model.parameters(), 'lr': 1e-4},
    {'params': [tau_train, k_train], 'lr': 1e-2}  # Higher learning rate for parameters
])
# train per i parametri
for i in tqdm(range(step*5), desc="Train for parameters"):
    optimizer_full.zero_grad()
    loss1 = torch.mean((model(x_data_new) - y_data_new) ** 2)
    # compute the "physics loss"
    yhp = model(x_physics)
    dx  = torch.autograd.grad(yhp, x_physics, torch.ones_like(yhp), create_graph=True)[0]# computes dy/dx
    physics = dx + k_train*yhp - tau_train*(cAin-yhp)# computes the residual of the CSTR irreversible mass balance equation
    loss2 = (1e-4)*torch.mean(physics**2)
    # BC loss
    bc1 = model(single_point)
    loss3 = torch.mean((bc1-10) ** 2)

    #per adesso non ci voglio mettere costanti di dumping
    loss = loss1 + loss2 + loss3
    loss.backward()
    optimizer_full.step()
    tau_history.append(tau_train.item())
    k_history.append(k_train.item())
    # plot the result as training progresses
    if (i + 1) % 2000 == 0:
        yh = model(x).detach()
        xp = x_physics.detach()
        plot_result(i, x, y, x_data_new, y_data_new, yh)
        file = "plots/CSTRpinn_inverse_%.8i.png" % (i + 1)
        plt.savefig(file, bbox_inches='tight', pad_inches=0.1, dpi=100, facecolor="white")
        files.append(file)
        if (i + 1) % 6000 == 0:
            plt.show()
        else:
            plt.close("all")
save_gif_PIL("CSTRpinn_inverse.gif", files, fps=20, loop=0)
# Plot parameter evolution
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(tau_history, label="tau_train")
plt.axhline(F/V, color="r", linestyle="--", label="True tau")
plt.xlabel("Iteration")
plt.ylabel("tau_train")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(k_history, label="k_train")
plt.axhline(k, color="r", linestyle="--", label="True k")
plt.xlabel("Iteration")
plt.ylabel("k_train")
plt.legend()

plt.tight_layout()
plt.show()

print("Training complete.")