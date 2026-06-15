import sys, os
sys.path.append('c:/Users/eaglw/Documents/PINN tesi')
from Viscoelastic.src.load_comsol import load_comsol_csv
params = {'mu_s': 0.005, 'mu_p': 0.005, 'lam': 1.0, 'eps': 0.0, 'alpha': 0.0, 'rho': 1.0}
try:
    ds = load_comsol_csv('c:/Users/eaglw/Documents/PINN tesi/Viscoelastic/data/Oldroyd_geom.csv', params)
except Exception as e:
    print(e)
