#########################################################
# 2D Dynamic Stokes FLow
# 20180529
#########################################################

# In[1]:

# Import the libraries (OpenCMISS,python,numpy,scipy)
import numpy,math,cmath,csv,time,sys,os,pdb
from scipy.fftpack import fft,ifft
from scipy.sparse  import linalg
from scipy.linalg  import inv,eig
from scipy.special import jn
from opencmiss.iron import iron

#------------------------------
#Set problem parameters
#------------------------------
height = 3.0
width = 1.0
length = 1.0

mu=1.0
rho=1.0
initial_conc=0.0
screen_output_freq=2

CoordinateSystemUserNumber=1
RegionUserNumber=2
MeshUserNumber=3
DecompositionUserNumber=4
GeometricFieldUserNumber=5
EquationsSetFieldUserNumber=6
DependentFieldUserNumberStokes=7
MaterialsFieldUserNumberStokes=8
IndependentFieldUserNumberStokes=9
EquationsSetUserNumberStokes=10
ProblemUserNumber=11
GeneratedMeshUserNumber=12
AnalyticFieldUserNumber=13
BasisUserNumber=14

DomainUserNumber=1
SolverStokesUserNumber=1
MaterialsFieldUserNumberStokesMu=1
MaterialsFieldUserNumberStokesRho=2

numberGlobalXElements = 3
numberGlobalYElements = 3
numberGlobalZElements = 3

#boundary condition nodes
InletWallNodes=[2,3]
FixedWallNodes=[1,4,5,8,9,12,13,16]

DYNAMIC_SOLVER_STOKES_OUTPUT_FREQUENCY = 1

timeIncrement   = 1.0
startTime       = 0.0
stopTime  = 5.0
dynamicSolverStokesTheta = [1.0]

#---------------------------------
# Get the computational nodes info
#---------------------------------
numberOfComputationalNodes = iron.ComputationalNumberOfNodesGet()
computationalNodeNumber    = iron.ComputationalNodeNumberGet()

#------------------
#Coordinate system
#------------------
coordinateSystem = iron.CoordinateSystem()
coordinateSystem.CreateStart(CoordinateSystemUserNumber)
coordinateSystem.dimension = 2
coordinateSystem.CreateFinish()

#------------------
#Regin
#------------------
# Create a region
region = iron.Region()
region.CreateStart(RegionUserNumber,iron.WorldRegion)
region.label = "StokesRegion"
region.coordinateSystem = coordinateSystem
region.CreateFinish()


# In[2]:

#--------
#Basis
#--------
# Create a tri-linear lagrange basis
basis = iron.Basis()
basis.CreateStart(BasisUserNumber)
basis.type = iron.BasisTypes.LAGRANGE_HERMITE_TP
basis.numberOfXi = 2
basis.interpolationXi = [iron.BasisInterpolationSpecifications.LINEAR_LAGRANGE]*2
basis.quadratureNumberOfGaussXi = [2]*2
#basis.quadratureLocalFaceGaussEvaluate = True
basis.CreateFinish()


# In[3]:

#----------
#Mesh
#----------
# Create a generated mesh
generatedMesh = iron.GeneratedMesh()
generatedMesh.CreateStart(GeneratedMeshUserNumber,region)
generatedMesh.type = iron.GeneratedMeshTypes.REGULAR
generatedMesh.basis = [basis]
generatedMesh.extent = [width,height]
generatedMesh.numberOfElements = [numberGlobalXElements,numberGlobalYElements]

mesh = iron.Mesh()
generatedMesh.CreateFinish(MeshUserNumber,mesh)

numberOfElements = mesh.numberOfElements
print("number of elements: " + str(numberOfElements))


# In[4]:

#------------------
#Mesh decomposition
#------------------
decomposition = iron.Decomposition()
decomposition.CreateStart(DecompositionUserNumber,mesh)
decomposition.type = iron.DecompositionTypes.CALCULATED
decomposition.numberOfDomains = numberOfComputationalNodes
decomposition.CreateFinish()


# In[5]:

#--------------
#Geometry field
#--------------

#Create a field for the geometry
geometricField = iron.Field()
geometricField.CreateStart(GeometricFieldUserNumber,region)
geometricField.meshDecomposition = decomposition
geometricField.ComponentMeshComponentSet(iron.FieldVariableTypes.U,1,1)
geometricField.ComponentMeshComponentSet(iron.FieldVariableTypes.U,2,1)
#geometricField.ComponentMeshComponentSet(iron.FieldVariableTypes.U,3,1)
geometricField.CreateFinish()

# Set geometry from the generated mesh
generatedMesh.GeometricParametersCalculate(geometricField)


# In[6]:

#-------------
#Equation sets
#-------------

#Create Stokes equations set
equationsSetField = iron.Field()
equationsSet = iron.EquationsSet()
equationsSetSpecification = [iron.EquationsSetClasses.FLUID_MECHANICS,
        iron.EquationsSetTypes.STOKES_EQUATION,
        iron.EquationsSetSubtypes.TRANSIENT_STOKES]
equationsSet.CreateStart(EquationsSetUserNumberStokes,region,geometricField,
        equationsSetSpecification,EquationsSetFieldUserNumber,equationsSetField)
equationsSet.CreateFinish()


# In[7]:

#---------------
#Dependent field
#---------------
# Create dependent field
dependentField = iron.Field()
equationsSet.DependentCreateStart(DependentFieldUserNumberStokes,dependentField)
dependentField.VariableLabelSet(iron.FieldVariableTypes.U,'U')
dependentField.ComponentMeshComponentSet(iron.FieldVariableTypes.U,1,1)
dependentField.ComponentMeshComponentSet(iron.FieldVariableTypes.U,2,1)
dependentField.ComponentMeshComponentSet(iron.FieldVariableTypes.U,3,1)
equationsSet.DependentCreateFinish()


# In[8]:

#--------------
#Material field
#--------------
# Create material field
materialField = iron.Field()
equationsSet.MaterialsCreateStart(MaterialsFieldUserNumberStokes,materialField)
materialField.VariableLabelSet(iron.FieldVariableTypes.U, "Material")
equationsSet.MaterialsCreateFinish()

# Mu and Rho?
materialField.ComponentValuesInitialiseDP(iron.FieldVariableTypes.U,iron.FieldParameterSetTypes.VALUES,1,mu)
materialField.ComponentValuesInitialiseDP(iron.FieldVariableTypes.U,iron.FieldParameterSetTypes.VALUES,2,rho)

# Initialise dependent field
dependentField.ComponentValuesInitialiseDP(iron.FieldVariableTypes.U,iron.FieldParameterSetTypes.VALUES,1,initial_conc)


# In[9]:

#-----------
# EQUATIONS
#-----------
# Create equations
equations = iron.Equations()
equationsSet.EquationsCreateStart(equations)
equations.sparsityType = iron.EquationsSparsityTypes.SPARSE
equations.lumpingType = iron.EquationsLumpingTypes.UNLUMPED
equations.outputType = iron.EquationsOutputTypes.NONE
equationsSet.EquationsCreateFinish()


# In[10]:

#-----------------------------------------------------------------------------------------------------------
#PROBLEM
#-----------------------------------------------------------------------------------------------------------

# Create Diffusion equation problem
problem = iron.Problem()
problemSpecification = [iron.ProblemClasses.FLUID_MECHANICS,
        iron.ProblemTypes.STOKES_EQUATION,
        iron.ProblemSubtypes.TRANSIENT_STOKES]
problem.CreateStart(ProblemUserNumber, problemSpecification)
problem.CreateFinish()

# Create control loops
#problem.ControlLoopCreateStart()
#problem.ControlLoopCreateFinish()
#===============
#  Control Loops
#===============
TimeLoop = iron.ControlLoop()
problem.ControlLoopCreateStart()
problem.ControlLoopGet([iron.ControlLoopIdentifiers.NODE],TimeLoop)
TimeLoop.LabelSet('Time Loop')
TimeLoop.TimesSet(startTime,stopTime,timeIncrement)
TimeLoop.TimeOutputSet(DYNAMIC_SOLVER_STOKES_OUTPUT_FREQUENCY)
TimeLoop.OutputTypeSet(iron.ControlLoopOutputTypes.NONE)
problem.ControlLoopCreateFinish()

# In[11]:

#-------
#SOLVER
#-------
DYNAMIC_SOLVER_STOKES_OUTPUT_TYPE    = iron.SolverOutputTypes.NONE
LINEAR_SOLVER_STOKES_OUTPUT_TYPE    = iron.SolverOutputTypes.NONE
# Create problem solver
solverLinear = iron.Solver()
solverDynamic = iron.Solver()
problem.SolversCreateStart()
problem.SolverGet([iron.ControlLoopIdentifiers.NODE],SolverStokesUserNumber,solverDynamic)
solverDynamic.OutputTypeSet(DYNAMIC_SOLVER_STOKES_OUTPUT_TYPE)
solverDynamic.DynamicThetaSet(dynamicSolverStokesTheta)
solverDynamic.DynamicLinearSolverGet(solverLinear)
solverLinear.outputType = iron.SolverOutputTypes.SOLVER
solverLinear.linearType = iron.LinearSolverTypes.ITERATIVE
solverLinear.linearIterativeAbsoluteTolerance = 1.0E-10
solverLinear.linearIterativeRelativeTolerance = 1.0E-10
problem.SolversCreateFinish()


# In[12]:

#-----------------------------------------------------------------------------------------------------------
#SOLVER EQUATIONS
#-----------------------------------------------------------------------------------------------------------
## Create solver equations and add equations set to solver equations
solver = iron.Solver()
solverEquations = iron.SolverEquations()
problem.SolverEquationsCreateStart()
problem.SolverGet([iron.ControlLoopIdentifiers.NODE],1,solver)
solver.SolverEquationsGet(solverEquations)
solverEquations.sparsityType = iron.SolverEquationsSparsityTypes.SPARSE
equationsSetIndex = solverEquations.EquationsSetAdd(equationsSet)
problem.SolverEquationsCreateFinish()


# In[13]:

#-------------------
#Boundary conditions
#-------------------
# Create boundary conditions and set first and last nodes to 0.0 and 1.0
boundaryConditions = iron.BoundaryConditions()
solverEquations.BoundaryConditionsCreateStart(boundaryConditions)

"""
firstNodeNumber=1
nodes = iron.Nodes()
region.NodesGet(nodes)
lastNodeNumber = nodes.numberOfNodes
firstNodeDomain = decomposition.NodeDomainGet(firstNodeNumber,1)
lastNodeDomain = decomposition.NodeDomainGet(lastNodeNumber,1)
"""
#Set velocity boundary conditions
for i in InletWallNodes:
    print i
    boundaryConditions.SetNode(dependentField,iron.FieldVariableTypes.U,1,1,i,1,iron.BoundaryConditionsTypes.FIXED,0.0)
    boundaryConditions.SetNode(dependentField,iron.FieldVariableTypes.U,1,1,i,2,iron.BoundaryConditionsTypes.FIXED,1.0)
    #boundaryConditions.SetNode(dependentField,iron.FieldVariableTypes.U,1,1,i,3,iron.BoundaryConditionsTypes.FIXED,0.0)

#Set fixed wall nodes
for i in FixedWallNodes:
    print i
    boundaryConditions.SetNode(dependentField,iron.FieldVariableTypes.U,1,1,i,1,iron.BoundaryConditionsTypes.FIXED_WALL,0.0)
    boundaryConditions.SetNode(dependentField,iron.FieldVariableTypes.U,1,1,i,2,iron.BoundaryConditionsTypes.FIXED_WALL,0.0)

solverEquations.BoundaryConditionsCreateFinish()


# In[ ]:
#-----------------------------------------------------------------------------------------------------------
#SOLVE
#-----------------------------------------------------------------------------------------------------------
#problem.Solve()
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Solve the problem
print "Solving problem..."
start = time.time()
problem.Solve()
end = time.time()
elapsed = end - start
print "Calculation Time = %3.4f" %elapsed
print "Problem solved!"
print "#"

#-----------------------------------------------------------------------------------------------------------
#OUTPUT
#-----------------------------------------------------------------------------------------------------------
# Ensure output directories exist
#if not os.path.exists('./output'):
#    os.makedirs('./output')

# Export results
#fields = iron.Fields()
#fields.CreateRegion(region)
#fields.NodesExport("output/DynamicStokes","FORTRAN")
#fields.ElementsExport("output/DynamicStokes","FORTRAN")
#fields.Finalise()

iron.Finalise()


# In[ ]:
