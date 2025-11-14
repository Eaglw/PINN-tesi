# train standard neural network to fit training data
torch.manual_seed(123)
model = FCN(1,1,32,3)
optimizer = torch.optim.Adam(model.parameters(),lr=1e-3)
files = []
for i in tqdm(range(step), desc="Training NN"):
    optimizer.zero_grad()
    yh = model(x_data)
    loss = torch.mean((yh-y_data)**2)# use mean squared error
    loss.backward()
    optimizer.step()
    # plot the result as training progresses
    if (i+1) % 100 == 0: 
        yh = model(x).detach()
        plot_result(i,x,y,x_data,y_data,yh)
        file = "plots/CSTRnn_%.8i.png"%(i+1)
        plt.savefig(file, bbox_inches='tight', pad_inches=0.1, dpi=100, facecolor="white")
        files.append(file)
        if (i+1) % 5000 == 0: plt.show()# cambiato per non vedere sempre
        else: plt.close("all") 
save_gif_PIL("IrreversibleCSTR/CSTRnn.gif", files, fps=20, loop=0,delete_files=True)



# train physics informed neural network (PINN)  
x_physics = torch.linspace(0,5,50).view(-1,1).requires_grad_(True)# sample locations over the problem domain
torch.manual_seed(123)
model = FCN(1,1,32,3)
optimizer = torch.optim.Adam(model.parameters(),lr=1e-4)
files = []
for i in tqdm(range(step), desc="Training PINN"):
    optimizer.zero_grad()
    # compute the "data loss"
    yh = model(x_data)
    loss1 = torch.mean((yh-y_data)**2)# use mean squared error
    # compute the "physics loss"
    yhp = model(x_physics)
    dx  = torch.autograd.grad(yhp, x_physics, torch.ones_like(yhp), create_graph=True)[0] #computes dy/dx
    physics = dx + k*yhp - F*(cAin-yhp)/V #computes the residual of the CSTR irreversible mass balance equation
    loss2 = (1e-4)*torch.mean(physics**2) #penso che qua sotto yhp è cA(t), dx sarebbe dcA/dt
    # backpropagate joint loss
    loss = loss1 + loss2 #add two loss terms together
    loss.backward()
    optimizer.step()
    # plot the result as training progresses
    if (i+1) % 100 == 0: 
        yh = model(x).detach()
        xp = x_physics.detach()
        plot_result(i,x,y,x_data,y_data,yh,xp)
        file = "plots/CSTRpinn_%.8i.png"%(i+1)
        plt.savefig(file, bbox_inches='tight', pad_inches=0.1, dpi=100, facecolor="white")
        files.append(file)
        if (i+1) % 4500 == 0: plt.show()
        else: plt.close("all")  
save_gif_PIL("IrreversibleCSTR/CSTRpinn.gif", files, fps=20, loop=0)