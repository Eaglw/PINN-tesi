# Literature: Oldroyd-B Model (Wikipedia)

## Summary
The Oldroyd-B model is a constitutive model used to describe the flow of viscoelastic fluids. It is an extension of the upper-convected Maxwell model and can be visualized as a fluid containing elastic bead-and-spring dumbbells.

## Key Methodology
- **Constitutive Equation**: 
  \[ \mathbf{T} + \lambda_1 \stackrel{\nabla}{\mathbf{T}} = 2\eta_0 (\mathbf{D} + \lambda_2 \stackrel{\nabla}{\mathbf{D}}) \]
  where \(\lambda_1\) is the relaxation time and \(\lambda_2\) is the retardation time.
- **Upper-Convected Time Derivative**: 
  \[ \stackrel{\nabla}{\mathbf{T}} = \frac{\partial}{\partial t}\mathbf{T} + \mathbf{v} \cdot \nabla \mathbf{T} - ((\nabla \mathbf{v})^T \cdot \mathbf{T} + \mathbf{T} \cdot (\nabla \mathbf{v})) \]
- **Split Formulation**: The stress tensor \(\mathbf{T}\) is often split into solvent and polymeric parts:
  \[ \mathbf{T} = 2\eta_s \mathbf{D} + \mathbf{\tau} \]
  \[ \mathbf{\tau} + \lambda_1 \stackrel{\nabla}{\mathbf{\tau}} = 2\eta_p \mathbf{D} \]

## Key Findings
- Effectively describes shear flow in viscoelastic fluids.
- Contains an unphysical singularity in idealized extensional flow (infinite stretching of dumbbells).
- Reduces to the Upper-Convected Maxwell (UCM) model if solvent viscosity is zero.

## Related
- **Topics**: [[Viscoelasticity]], [[Fluid_Dynamics]]
- **Methods**: [[ViscoelasticNet]]
- **Systems**: [[Viscoelastic_Fluids]]
