"""
in this case I want to solve the problem without training data, but using only the equation
and the boundary condition
"""
# train physics informed neural network (PINN)  
x_physics = torch.linspace(0,5,50).view(-1,1).requires_grad_(True)# sample locations over the problem domain


torch.manual_seed(123)
model = FCN(1,1,32,3)
optimizer = torch.optim.Adam(model.parameters(),lr=1e-4)
files = []
for i in tqdm(range(step), desc="Training nodata"):
    optimizer.zero_grad()

    # compute the "physics loss"
    yhp = model(x_physics)
    dx  = torch.autograd.grad(yhp, x_physics, torch.ones_like(yhp), create_graph=True)[0]# computes dy/dx
    physics = dx + k*yhp - F*(cAin-yhp)/V# computes the residual of the CSTR irreversible mass balance equation
    loss2 = (1e-4)*torch.mean(physics**2)
    # penso che qua sotto yhp è cA(t), dx sarebbe dcA/dt
    
    # now I need to add boundary and initial conditions (in some points)
    # since it is  1d problem I just need 1 colocation points (at time t=0)!
    single_point = x[0].unsqueeze(1)
    #questo single point non andrebbe fuori dal loop?
    # questo step a quanto pare è necessario!
    single_point.requires_grad = True
    bc1 = model(single_point)
    # questa loss limita il training ----> non sta imparando questa condizione!
    loss3 = torch.mean((bc1-10) ** 2)
    #credo che questo sia cA(t=0)=10
    # backpropagate joint loss
    loss = loss2+loss3# add two loss terms together
    loss.backward()
    optimizer.step()
    
    x_data=[]
    y_data=[]
    # plot the result as training progresses
    if (i+1) % 2000 == 0: 
        #print("Iteration: {}  | Physics Loss: {} | Dirichlet Loss: {}".format(i,loss2,loss3))
        
        yh = model(x).detach()
        xp = x_physics.detach()
        
        plot_result(i,x,y,x_data,y_data,yh,xp)
        
        file = "plots/CSTRpinn_nodata%.8i.png"%(i+1)
        plt.savefig(file, bbox_inches='tight', pad_inches=0.1, dpi=100, facecolor="white")
        files.append(file)
        
        if (i+1) % 500 == 0: plt.show()
        else: plt.close("all")
            
save_gif_PIL("IrreversibleCSTR/CSTRpinn_nodata.gif", files, fps=20, loop=0)