PROGRAM stokes_dynamic

  !
  !================================================================================================================================
  !

  USE OpenCMISS
  USE OpenCMISS_Iron
#ifndef NOMPIMOD
  USE MPI
#endif
  IMPLICIT NONE
#ifdef NOMPIMOD
#include "mpif.h"
#endif

  !Test program parameters
  REAL(CMISSRP), PARAMETER :: HEIGHT=3.0_CMISSRP
  REAL(CMISSRP), PARAMETER :: WIDTH=1.0_CMISSRP
  REAL(CMISSRP), PARAMETER :: LENGTH=1.0_CMISSRP

  INTEGER(CMISSIntg), PARAMETER :: CoordinateSystemUserNumber=1
  INTEGER(CMISSIntg), PARAMETER :: RegionUserNumber=2
  INTEGER(CMISSIntg), PARAMETER :: MeshUserNumber=3
  INTEGER(CMISSIntg), PARAMETER :: DecompositionUserNumber=4
  INTEGER(CMISSIntg), PARAMETER :: GeometricFieldUserNumber=5
  INTEGER(CMISSIntg), PARAMETER :: EquationsSetFieldUserNumber=6
  INTEGER(CMISSIntg), PARAMETER :: DependentFieldUserNumberStokes=7
  INTEGER(CMISSIntg), PARAMETER :: MaterialsFieldUserNumberStokes=8
  INTEGER(CMISSIntg), PARAMETER :: EquationsSetUserNumberStokes=9
  INTEGER(CMISSIntg), PARAMETER :: ProblemUserNumber=10
  INTEGER(CMISSIntg), PARAMETER :: GeneratedMeshUserNumber=11
  INTEGER(CMISSIntg), PARAMETER :: AnalyticFieldUserNumber=12

  INTEGER(CMISSIntg), PARAMETER :: DomainUserNumber=2
  INTEGER(CMISSIntg), PARAMETER :: SolverStokesUserNumber=1
  INTEGER(CMISSIntg), PARAMETER :: MaterialsFieldUserNumberStokesMu=1
  INTEGER(CMISSIntg), PARAMETER :: MaterialsFieldUserNumberStokesRho=2

  !Program types
  !Program variables

  INTEGER(CMISSIntg) :: NUMBER_OF_DIMENSIONS
  INTEGER(CMISSIntg) :: NUMBER_GLOBAL_X_ELEMENTS,NUMBER_GLOBAL_Y_ELEMENTS,NUMBER_GLOBAL_Z_ELEMENTS
  INTEGER(CMISSIntg) :: BASIS_TYPE
  INTEGER(CMISSIntg) :: BASIS_NUMBER_SPACE
  INTEGER(CMISSIntg) :: BASIS_NUMBER_VELOCITY
  INTEGER(CMISSIntg) :: BASIS_NUMBER_PRESSURE
  INTEGER(CMISSIntg) :: BASIS_XI_GAUSS_SPACE
  INTEGER(CMISSIntg) :: BASIS_XI_GAUSS_VELOCITY
  INTEGER(CMISSIntg) :: BASIS_XI_GAUSS_PRESSURE
  INTEGER(CMISSIntg) :: BASIS_XI_INTERPOLATION_SPACE
  INTEGER(CMISSIntg) :: BASIS_XI_INTERPOLATION_VELOCITY
  INTEGER(CMISSIntg) :: BASIS_XI_INTERPOLATION_PRESSURE
  INTEGER(CMISSIntg) :: MESH_NUMBER_OF_COMPONENTS
  INTEGER(CMISSIntg) :: MESH_COMPONENT_NUMBER_SPACE
  INTEGER(CMISSIntg) :: MESH_COMPONENT_NUMBER_VELOCITY
  INTEGER(CMISSIntg) :: MESH_COMPONENT_NUMBER_PRESSURE
  INTEGER(CMISSIntg) :: NUMBER_OF_NODES_SPACE
  INTEGER(CMISSIntg) :: NUMBER_OF_NODES_VELOCITY
  INTEGER(CMISSIntg) :: NUMBER_OF_NODES_PRESSURE
  INTEGER(CMISSIntg) :: NUMBER_OF_ELEMENT_NODES_SPACE
  INTEGER(CMISSIntg) :: NUMBER_OF_ELEMENT_NODES_VELOCITY
  INTEGER(CMISSIntg) :: NUMBER_OF_ELEMENT_NODES_PRESSURE
  INTEGER(CMISSIntg) :: TOTAL_NUMBER_OF_NODES
  INTEGER(CMISSIntg) :: TOTAL_NUMBER_OF_ELEMENTS
  INTEGER(CMISSIntg) :: MAXIMUM_ITERATIONS
  INTEGER(CMISSIntg) :: RESTART_VALUE
  !   INTEGER(CMISSIntg) :: MPI_IERROR
  INTEGER(CMISSIntg) :: NUMBER_OF_FIXED_WALL_NODES_STOKES
  INTEGER(CMISSIntg) :: NUMBER_OF_INLET_WALL_NODES_STOKES

  INTEGER(CMISSIntg) :: EQUATIONS_STOKES_OUTPUT
  INTEGER(CMISSIntg) :: COMPONENT_NUMBER
  INTEGER(CMISSIntg) :: NODE_NUMBER
  INTEGER(CMISSIntg) :: ELEMENT_NUMBER
  INTEGER(CMISSIntg) :: NODE_COUNTER
  INTEGER(CMISSIntg) :: CONDITION

  INTEGER(CMISSIntg) :: DYNAMIC_SOLVER_STOKES_OUTPUT_FREQUENCY
  INTEGER(CMISSIntg) :: DYNAMIC_SOLVER_STOKES_OUTPUT_TYPE
  INTEGER(CMISSIntg) :: LINEAR_SOLVER_STOKES_OUTPUT_TYPE

  INTEGER, ALLOCATABLE, DIMENSION(:):: FIXED_WALL_NODES_STOKES
  INTEGER, ALLOCATABLE, DIMENSION(:):: INLET_WALL_NODES_STOKES

  REAL(CMISSRP) :: INITIAL_FIELD_STOKES(3)
  REAL(CMISSRP) :: BOUNDARY_CONDITIONS_STOKES(3)
  REAL(CMISSRP) :: DIVERGENCE_TOLERANCE
  REAL(CMISSRP) :: RELATIVE_TOLERANCE
  REAL(CMISSRP) :: ABSOLUTE_TOLERANCE
  REAL(CMISSRP) :: LINESEARCH_ALPHA
  REAL(CMISSRP) :: VALUE
  REAL(CMISSRP) :: MU_PARAM_STOKES
  REAL(CMISSRP) :: RHO_PARAM_STOKES

  REAL(CMISSRP) :: DYNAMIC_SOLVER_STOKES_START_TIME
  REAL(CMISSRP) :: DYNAMIC_SOLVER_STOKES_STOP_TIME
  REAL(CMISSRP) :: DYNAMIC_SOLVER_STOKES_THETA
  REAL(CMISSRP) :: DYNAMIC_SOLVER_STOKES_TIME_INCREMENT

  LOGICAL :: EXPORT_FIELD_IO
  LOGICAL :: LINEAR_SOLVER_STOKES_DIRECT_FLAG
  LOGICAL :: FIXED_WALL_NODES_STOKES_FLAG
  LOGICAL :: INLET_WALL_NODES_STOKES_FLAG

  !CMISS variables

  TYPE(cmfe_RegionType) :: Region
  TYPE(cmfe_RegionType) :: WorldRegion
  TYPE(cmfe_ComputationEnvironmentType) :: computationEnvironment
  TYPE(cmfe_ContextType) :: context
  TYPE(cmfe_CoordinateSystemType) :: CoordinateSystem
  TYPE(cmfe_BasisType) :: BasisGeometry
  TYPE(cmfe_BasisType) :: BasisVelocity
  TYPE(cmfe_BasisType) :: BasisPressure
  TYPE(cmfe_NodesType) :: Nodes
  TYPE(cmfe_MeshElementsType) :: MeshElementsSpace
  TYPE(cmfe_MeshElementsType) :: MeshElementsVelocity
  TYPE(cmfe_MeshElementsType) :: MeshElementsPressure
  TYPE(cmfe_MeshType) :: Mesh
  TYPE(cmfe_GeneratedMeshType) :: GeneratedMesh
  TYPE(cmfe_DecompositionType) :: Decomposition
  TYPE(cmfe_FieldsType) :: Fields
  TYPE(cmfe_FieldType) :: GeometricField,AnalyticField
  TYPE(cmfe_FieldType) :: EquationsSetField
  TYPE(cmfe_FieldType) :: DependentFieldStokes
  TYPE(cmfe_FieldType) :: MaterialsFieldStokes
  TYPE(cmfe_BoundaryConditionsType) :: BoundaryConditionsStokes
  TYPE(cmfe_EquationsSetType) :: EquationsSetStokes
  TYPE(cmfe_EquationsType) :: EquationsStokes
  TYPE(cmfe_ProblemType) :: Problem
  TYPE(cmfe_ControlLoopType) :: ControlLoop
  TYPE(cmfe_SolverType) :: DynamicSolverStokes
  TYPE(cmfe_SolverType) :: LinearSolverStokes
  TYPE(cmfe_SolverEquationsType) :: SolverEquationsStokes

  !Generic CMISS variables

  INTEGER(CMISSIntg) :: NumberOfComputationalNodes,ComputationalNodeNumber,BoundaryNodeDomain
  INTEGER(CMISSIntg) :: EquationsSetIndex,Err

  !
  !================================================================================================================================
  !

  !INITIALISE OPENCMISS

  CALL cmfe_Context_Initialise(context,err)
  CALL cmfe_Initialise(context,err)
  CALL cmfe_ErrorHandlingModeSet(CMFE_ERRORS_TRAP_ERROR,err)
  CALL cmfe_Region_Initialise(worldRegion,err)
  CALL cmfe_Context_WorldRegionGet(context,worldRegion,err)

  !
  !================================================================================================================================
  !

  !CHECK COMPUTATIONAL NODE

  !Get the computational nodes information
  CALL cmfe_ComputationEnvironment_Initialise(computationEnvironment,err)
  CALL cmfe_Context_ComputationEnvironmentGet(context,computationEnvironment,err)
  CALL cmfe_ComputationEnvironment_NumberOfWorldNodesGet(computationEnvironment,numberOfComputationalNodes,err)
  CALL cmfe_ComputationEnvironment_WorldNodeNumberGet(computationEnvironment,computationalNodeNumber,err)

  !
  !================================================================================================================================
  !

  !PROBLEM CONTROL PANEL
  NUMBER_GLOBAL_X_ELEMENTS=3
  NUMBER_GLOBAL_Y_ELEMENTS=3
  NUMBER_GLOBAL_Z_ELEMENTS=3
  MESH_COMPONENT_NUMBER_SPACE=1
  MESH_COMPONENT_NUMBER_VELOCITY=1
  MESH_COMPONENT_NUMBER_PRESSURE=1
  NUMBER_OF_DIMENSIONS=2
  BASIS_TYPE=CMFE_BASIS_LINEAR_LAGRANGE_INTERPOLATION
  BASIS_XI_INTERPOLATION_SPACE=CMFE_BASIS_LINEAR_LAGRANGE_INTERPOLATION
  BASIS_XI_INTERPOLATION_VELOCITY=CMFE_BASIS_LINEAR_LAGRANGE_INTERPOLATION
  BASIS_XI_INTERPOLATION_PRESSURE=CMFE_BASIS_LINEAR_LAGRANGE_INTERPOLATION

  !Set initial values
  INITIAL_FIELD_STOKES(1)=0.0_CMISSRP
  INITIAL_FIELD_STOKES(2)=0.0_CMISSRP
  INITIAL_FIELD_STOKES(3)=0.0_CMISSRP
  !Set boundary conditions
  FIXED_WALL_NODES_STOKES_FLAG=.TRUE.
  INLET_WALL_NODES_STOKES_FLAG=.TRUE.
  IF(FIXED_WALL_NODES_STOKES_FLAG) THEN
     NUMBER_OF_FIXED_WALL_NODES_STOKES=8
     ALLOCATE(FIXED_WALL_NODES_STOKES(NUMBER_OF_FIXED_WALL_NODES_STOKES))
     FIXED_WALL_NODES_STOKES=[1,4,5,8,9,12,13,16]
  ENDIF
  IF(INLET_WALL_NODES_STOKES_FLAG) THEN
     NUMBER_OF_INLET_WALL_NODES_STOKES=2
     ALLOCATE(INLET_WALL_NODES_STOKES(NUMBER_OF_INLET_WALL_NODES_STOKES))
     INLET_WALL_NODES_STOKES=[2,3]
     !Set initial boundary conditions
     BOUNDARY_CONDITIONS_STOKES(1)=0.0_CMISSRP
     BOUNDARY_CONDITIONS_STOKES(2)=1.0_CMISSRP
     BOUNDARY_CONDITIONS_STOKES(3)=0.0_CMISSRP
  ENDIF
  !Set material parameters
  MU_PARAM_STOKES=1.0_CMISSRP
  RHO_PARAM_STOKES=1.0_CMISSRP
  !Set interpolation parameters
  BASIS_XI_GAUSS_SPACE=3
  BASIS_XI_GAUSS_VELOCITY=3
  BASIS_XI_GAUSS_PRESSURE=3
  !Set output parameter
  !(NoOutput/ProgressOutput/TimingOutput/SolverOutput/SolverMatrixOutput)
  DYNAMIC_SOLVER_STOKES_OUTPUT_TYPE=CMFE_SOLVER_NO_OUTPUT
  LINEAR_SOLVER_STOKES_OUTPUT_TYPE=CMFE_SOLVER_NO_OUTPUT
  !(NoOutput/TimingOutput/MatrixOutput/ElementOutput)
  EQUATIONS_STOKES_OUTPUT=CMFE_EQUATIONS_NO_OUTPUT
  !Set time parameter
  DYNAMIC_SOLVER_STOKES_START_TIME=0.0_CMISSRP
  DYNAMIC_SOLVER_STOKES_STOP_TIME=5.0_CMISSRP
  DYNAMIC_SOLVER_STOKES_TIME_INCREMENT=1.0_CMISSRP
  DYNAMIC_SOLVER_STOKES_THETA=1.0_CMISSRP
  !Set result output parameter
  DYNAMIC_SOLVER_STOKES_OUTPUT_FREQUENCY=1
  !Set solver parameters
  LINEAR_SOLVER_STOKES_DIRECT_FLAG=.FALSE.
  RELATIVE_TOLERANCE=1.0E-10_CMISSRP !default: 1.0E-05_CMISSRP
  ABSOLUTE_TOLERANCE=1.0E-10_CMISSRP !default: 1.0E-10_CMISSRP
  DIVERGENCE_TOLERANCE=1.0E20 !default: 1.0E5
  MAXIMUM_ITERATIONS=100000 !default: 100000
  RESTART_VALUE=3000 !default: 30
  LINESEARCH_ALPHA=1.0

  !
  !================================================================================================================================
  !

  !COORDINATE SYSTEM

  !Start the creation of a new RC coordinate system
  CALL cmfe_CoordinateSystem_Initialise(CoordinateSystem,Err)
  CALL cmfe_CoordinateSystem_CreateStart(CoordinateSystemUserNumber,context,CoordinateSystem,Err)
  !Set the coordinate system dimension
  CALL cmfe_CoordinateSystem_DimensionSet(CoordinateSystem,NUMBER_OF_DIMENSIONS,Err)
  !Finish the creation of the coordinate system
  CALL cmfe_CoordinateSystem_CreateFinish(CoordinateSystem,Err)

  !
  !================================================================================================================================
  !

  !REGION

  !Start the creation of a new region
  CALL cmfe_Region_Initialise(Region,Err)
  CALL cmfe_Region_CreateStart(RegionUserNumber,WorldRegion,Region,Err)
  CALL cmfe_Region_LabelSet(Region,"StokesRegion",Err)
  !Set the regions coordinate system as defined above
  CALL cmfe_Region_CoordinateSystemSet(Region,CoordinateSystem,Err)
  !Finish the creation of the region
  CALL cmfe_Region_CreateFinish(Region,Err)

  !
  !================================================================================================================================
  !

  !BASES

  !Start the creation of new bases
  MESH_NUMBER_OF_COMPONENTS=1
  CALL cmfe_Basis_Initialise(BasisGeometry,Err)
  CALL cmfe_Basis_CreateStart(BASIS_NUMBER_SPACE,context,BasisGeometry,Err)
  !Set the basis type (Lagrange/Simplex)
  CALL cmfe_Basis_TypeSet(BasisGeometry,BASIS_TYPE,Err)
  !Set the basis xi number
  CALL cmfe_Basis_NumberOfXiSet(BasisGeometry,NUMBER_OF_DIMENSIONS,Err)
  !Set the basis xi interpolation and number of Gauss points
  IF(NUMBER_OF_DIMENSIONS==2) THEN
     CALL cmfe_Basis_InterpolationXiSet(BasisGeometry,[BASIS_XI_INTERPOLATION_SPACE,BASIS_XI_INTERPOLATION_SPACE],Err)
     CALL cmfe_Basis_QuadratureNumberOfGaussXiSet(BasisGeometry,[BASIS_XI_GAUSS_SPACE,BASIS_XI_GAUSS_SPACE],Err)
  ELSE
     CALL cmfe_Basis_InterpolationXiSet(BasisGeometry,[BASIS_XI_INTERPOLATION_SPACE,BASIS_XI_INTERPOLATION_SPACE, &
          & BASIS_XI_INTERPOLATION_SPACE],Err)
     CALL cmfe_Basis_QuadratureNumberOfGaussXiSet(BasisGeometry,[BASIS_XI_GAUSS_SPACE,BASIS_XI_GAUSS_SPACE,BASIS_XI_GAUSS_SPACE], &
          & Err)
  ENDIF
  !Finish the creation of the basis
  CALL cmfe_Basis_CreateFinish(BasisGeometry,Err)

  !Start the creation of another basis
  IF(BASIS_XI_INTERPOLATION_VELOCITY==BASIS_XI_INTERPOLATION_SPACE) THEN
     BasisVelocity=BasisGeometry
  ELSE
     MESH_NUMBER_OF_COMPONENTS=MESH_NUMBER_OF_COMPONENTS+1
     !Initialise a new velocity basis
     CALL cmfe_Basis_Initialise(BasisVelocity,Err)
     !Start the creation of a basis
     CALL cmfe_Basis_CreateStart(BASIS_NUMBER_VELOCITY,context,BasisVelocity,Err)
     !Set the basis type (Lagrange/Simplex)
     CALL cmfe_Basis_TypeSet(BasisVelocity,BASIS_TYPE,Err)
     !Set the basis xi number
     CALL cmfe_Basis_NumberOfXiSet(BasisVelocity,NUMBER_OF_DIMENSIONS,Err)
     !Set the basis xi interpolation and number of Gauss points
     IF(NUMBER_OF_DIMENSIONS==2) THEN
        CALL cmfe_Basis_InterpolationXiSet(BasisVelocity,[BASIS_XI_INTERPOLATION_VELOCITY,BASIS_XI_INTERPOLATION_VELOCITY],Err)
        CALL cmfe_Basis_QuadratureNumberOfGaussXiSet(BasisVelocity,[BASIS_XI_GAUSS_VELOCITY,BASIS_XI_GAUSS_VELOCITY],Err)
     ELSE
        CALL cmfe_Basis_InterpolationXiSet(BasisVelocity,[BASIS_XI_INTERPOLATION_VELOCITY,BASIS_XI_INTERPOLATION_VELOCITY, &
             & BASIS_XI_INTERPOLATION_VELOCITY],Err)
        CALL cmfe_Basis_QuadratureNumberOfGaussXiSet(BasisVelocity,[BASIS_XI_GAUSS_VELOCITY,BASIS_XI_GAUSS_VELOCITY, &
             & BASIS_XI_GAUSS_VELOCITY],Err)
     ENDIF
     !Finish the creation of the basis
     CALL cmfe_Basis_CreateFinish(BasisVelocity,Err)
  ENDIF

  !Start the creation of another basis
  IF(BASIS_XI_INTERPOLATION_PRESSURE==BASIS_XI_INTERPOLATION_SPACE) THEN
     BasisPressure=BasisGeometry
  ELSE IF(BASIS_XI_INTERPOLATION_PRESSURE==BASIS_XI_INTERPOLATION_VELOCITY) THEN
     BasisPressure=BasisVelocity
  ELSE
     MESH_NUMBER_OF_COMPONENTS=MESH_NUMBER_OF_COMPONENTS+1
     !Initialise a new pressure basis
     CALL cmfe_Basis_Initialise(BasisPressure,Err)
     !Start the creation of a basis
     CALL cmfe_Basis_CreateStart(BASIS_NUMBER_PRESSURE,context,BasisPressure,Err)
     !Set the basis type (Lagrange/Simplex)
     CALL cmfe_Basis_TypeSet(BasisPressure,BASIS_TYPE,Err)
     !Set the basis xi number
     CALL cmfe_Basis_NumberOfXiSet(BasisPressure,NUMBER_OF_DIMENSIONS,Err)
     !Set the basis xi interpolation and number of Gauss points
     IF(NUMBER_OF_DIMENSIONS==2) THEN
        CALL cmfe_Basis_InterpolationXiSet(BasisPressure,[BASIS_XI_INTERPOLATION_PRESSURE,BASIS_XI_INTERPOLATION_PRESSURE],Err)
        CALL cmfe_Basis_QuadratureNumberOfGaussXiSet(BasisPressure,[BASIS_XI_GAUSS_PRESSURE,BASIS_XI_GAUSS_PRESSURE],Err)
     ELSE
        CALL cmfe_Basis_InterpolationXiSet(BasisPressure,[BASIS_XI_INTERPOLATION_PRESSURE,BASIS_XI_INTERPOLATION_PRESSURE, &
             & BASIS_XI_INTERPOLATION_PRESSURE],Err)
        CALL cmfe_Basis_QuadratureNumberOfGaussXiSet(BasisPressure,[BASIS_XI_GAUSS_PRESSURE,BASIS_XI_GAUSS_PRESSURE, &
             & BASIS_XI_GAUSS_PRESSURE],Err)
     ENDIF
     !Finish the creation of the basis
     CALL cmfe_Basis_CreateFinish(BasisPressure,Err)
  ENDIF

  !
  !================================================================================================================================
  !

  !MESH

  !Start the creation of a generated mesh in the region
  CALL cmfe_GeneratedMesh_Initialise(GeneratedMesh,Err)
  CALL cmfe_GeneratedMesh_CreateStart(GeneratedMeshUserNumber,Region,GeneratedMesh,Err)
  !Set up a regular x*y*z mesh
  CALL cmfe_GeneratedMesh_TypeSet(GeneratedMesh,CMFE_GENERATED_MESH_REGULAR_MESH_TYPE,Err)
  !Set the default basis
  CALL cmfe_GeneratedMesh_BasisSet(GeneratedMesh,BasisGeometry,Err)
  !Define the mesh on the region
  IF(NUMBER_OF_DIMENSIONS==2) THEN
     CALL cmfe_GeneratedMesh_ExtentSet(GeneratedMesh,[WIDTH,HEIGHT],Err)
     CALL cmfe_GeneratedMesh_NumberOfElementsSet(GeneratedMesh,[NUMBER_GLOBAL_X_ELEMENTS,NUMBER_GLOBAL_Y_ELEMENTS],Err)
  ELSE
     CALL cmfe_GeneratedMesh_ExtentSet(GeneratedMesh,[WIDTH,HEIGHT,LENGTH],Err)
     CALL cmfe_GeneratedMesh_NumberOfElementsSet(GeneratedMesh,[NUMBER_GLOBAL_X_ELEMENTS,NUMBER_GLOBAL_Y_ELEMENTS, &
          & NUMBER_GLOBAL_Z_ELEMENTS],Err)
  ENDIF
  !Finish the creation of a generated mesh in the region
  CALL cmfe_Mesh_Initialise(Mesh,Err)
  CALL cmfe_GeneratedMesh_CreateFinish(GeneratedMesh,MeshUserNumber,Mesh,Err)

  !
  !================================================================================================================================
  !

  !MESH DECOMPOSITION

  !Create a decomposition
  CALL cmfe_Decomposition_Initialise(Decomposition,Err)
  CALL cmfe_Decomposition_CreateStart(DecompositionUserNumber,Mesh,Decomposition,Err)
  !Set the decomposition to be a general decomposition with the specified number of domains
  CALL cmfe_Decomposition_TypeSet(Decomposition,CMFE_DECOMPOSITION_CALCULATED_TYPE,Err)
  CALL cmfe_Decomposition_NumberOfDomainsSet(Decomposition,NumberOfComputationalNodes,Err)
  !Finish the decomposition
  CALL cmfe_Decomposition_CreateFinish(Decomposition,Err)

  !
  !================================================================================================================================
  !

  !GEOMETRIC FIELD

  !Start to create a default (geometric) field on the region
  CALL cmfe_Field_Initialise(GeometricField,Err)
  CALL cmfe_Field_CreateStart(GeometricFieldUserNumber,Region,GeometricField,Err)
  !Set the field type
  CALL cmfe_Field_TypeSet(GeometricField,CMFE_FIELD_GEOMETRIC_TYPE,Err)
  !Set the decomposition to use
  CALL cmfe_Field_MeshDecompositionSet(GeometricField,Decomposition,Err)
  !Set the scaling to use
  CALL cmfe_Field_ScalingTypeSet(GeometricField,CMFE_FIELD_NO_SCALING,Err)
  !Set the mesh component to be used by the field components.
  DO COMPONENT_NUMBER=1,NUMBER_OF_DIMENSIONS
     CALL cmfe_Field_ComponentMeshComponentSet(GeometricField,CMFE_FIELD_U_VARIABLE_TYPE,COMPONENT_NUMBER, &
          & MESH_COMPONENT_NUMBER_SPACE,Err)
  ENDDO
  !Finish creating the field
  CALL cmfe_Field_CreateFinish(GeometricField,Err)
  !Update the geometric field parameters
  CALL cmfe_GeneratedMesh_GeometricParametersCalculate(GeneratedMesh,GeometricField,Err)

  !
  !================================================================================================================================
  !

  !EQUATIONS SETS

  !Create the equations set for dynamic Stokes
  CALL cmfe_EquationsSet_Initialise(EquationsSetStokes,Err)
  CALL cmfe_Field_Initialise(EquationsSetField,Err)
  !Set the equations set to be a dynamic Stokes problem
  CALL cmfe_EquationsSet_CreateStart(EquationsSetUserNumberStokes,Region,GeometricField,[CMFE_EQUATIONS_SET_FLUID_MECHANICS_CLASS, &
       & CMFE_EQUATIONS_SET_STOKES_EQUATION_TYPE,CMFE_EQUATIONS_SET_TRANSIENT_STOKES_SUBTYPE],EquationsSetFieldUserNumber, &
       & EquationsSetField,EquationsSetStokes,Err)
  !Finish creating the equations set
  CALL cmfe_EquationsSet_CreateFinish(EquationsSetStokes,Err)

  !
  !================================================================================================================================
  !

  !DEPENDENT FIELDS

  !Create the equations set dependent field variables for dynamic Stokes
  CALL cmfe_Field_Initialise(DependentFieldStokes,Err)
  CALL cmfe_EquationsSet_DependentCreateStart(EquationsSetStokes,DependentFieldUserNumberStokes,DependentFieldStokes,Err)
  CALL cmfe_Field_VariableLabelSet(DependentFieldStokes,CMFE_FIELD_U_VARIABLE_TYPE,"U",Err)
  !Set the mesh component to be used by the field components.
  DO COMPONENT_NUMBER=1,NUMBER_OF_DIMENSIONS
     CALL cmfe_Field_ComponentMeshComponentSet(DependentFieldStokes,CMFE_FIELD_U_VARIABLE_TYPE,COMPONENT_NUMBER, &
          & MESH_COMPONENT_NUMBER_VELOCITY,Err)
     CALL cmfe_Field_ComponentMeshComponentSet(DependentFieldStokes,CMFE_FIELD_DELUDELN_VARIABLE_TYPE,COMPONENT_NUMBER, &
          & MESH_COMPONENT_NUMBER_VELOCITY,Err)
  ENDDO
  COMPONENT_NUMBER=NUMBER_OF_DIMENSIONS+1
  CALL cmfe_Field_ComponentMeshComponentSet(DependentFieldStokes,CMFE_FIELD_U_VARIABLE_TYPE,COMPONENT_NUMBER, &
       & MESH_COMPONENT_NUMBER_PRESSURE,Err)
  CALL cmfe_Field_ComponentMeshComponentSet(DependentFieldStokes,CMFE_FIELD_DELUDELN_VARIABLE_TYPE,COMPONENT_NUMBER, &
       & MESH_COMPONENT_NUMBER_PRESSURE,Err)
  !Finish the equations set dependent field variables
  CALL cmfe_EquationsSet_DependentCreateFinish(EquationsSetStokes,Err)

  !Initialise dependent field
  DO COMPONENT_NUMBER=1,NUMBER_OF_DIMENSIONS
     CALL cmfe_Field_ComponentValuesInitialise(DependentFieldStokes,CMFE_FIELD_U_VARIABLE_TYPE,CMFE_FIELD_VALUES_SET_TYPE, &
          & COMPONENT_NUMBER,INITIAL_FIELD_STOKES(COMPONENT_NUMBER),Err)
  ENDDO


  !
  !================================================================================================================================
  !

  !MATERIALS FIELDS

  !Create the equations set materials field variables for dynamic Stokes
  CALL cmfe_Field_Initialise(MaterialsFieldStokes,Err)
  CALL cmfe_EquationsSet_MaterialsCreateStart(EquationsSetStokes,MaterialsFieldUserNumberStokes,MaterialsFieldStokes,Err)
  CALL cmfe_Field_VariableLabelSet(MaterialsFieldStokes,CMFE_FIELD_U_VARIABLE_TYPE,"Materials",Err)
  !Finish the equations set materials field variables
  CALL cmfe_EquationsSet_MaterialsCreateFinish(EquationsSetStokes,Err)
  CALL cmfe_Field_ComponentValuesInitialise(MaterialsFieldStokes,CMFE_FIELD_U_VARIABLE_TYPE,CMFE_FIELD_VALUES_SET_TYPE, &
       & MaterialsFieldUserNumberStokesMu,MU_PARAM_STOKES,Err)
  CALL cmfe_Field_ComponentValuesInitialise(MaterialsFieldStokes,CMFE_FIELD_U_VARIABLE_TYPE,CMFE_FIELD_VALUES_SET_TYPE, &
       & MaterialsFieldUserNumberStokesRho,RHO_PARAM_STOKES,Err)

  !
  !================================================================================================================================
  !

  !ANALYTIC FIELD

  !Create the equations set analytic field variables
  !CALL cmfe_Field_Initialise(AnalyticField,Err)
  !CALL cmfe_EquationsSet_AnalyticCreateStart(EquationsSetStokes,CMFE_EQUATIONS_SET_NAVIER_STOKES_EQUATION_TWO_DIM_4, &
  !     & AnalyticFieldUserNumber,AnalyticField,Err)
  !Finish the equations set analytic field variables
  !CALL cmfe_EquationsSet_AnalyticCreateFinish(EquationsSetStokes,Err)

  !
  !================================================================================================================================
  !

  !EQUATIONS

  !Create the equations set equations
  CALL cmfe_Equations_Initialise(EquationsStokes,Err)
  CALL cmfe_EquationsSet_EquationsCreateStart(EquationsSetStokes,EquationsStokes,Err)
  !Set the equations matrices sparsity type
  CALL cmfe_Equations_SparsityTypeSet(EquationsStokes,CMFE_EQUATIONS_SPARSE_MATRICES,Err)
  !Set the equations lumping type
  CALL cmfe_Equations_LumpingTypeSet(EquationsStokes,CMFE_EQUATIONS_UNLUMPED_MATRICES,Err)
  !Set the equations set output
  CALL cmfe_Equations_OutputTypeSet(EquationsStokes,EQUATIONS_STOKES_OUTPUT,Err)
  !Finish the equations set equations
  CALL cmfe_EquationsSet_EquationsCreateFinish(EquationsSetStokes,Err)

  !
  !================================================================================================================================
  !

  !PROBLEMS

  !Start the creation of a problem.
  CALL cmfe_Problem_Initialise(Problem,Err)
  CALL cmfe_ControlLoop_Initialise(ControlLoop,Err)
  CALL cmfe_Problem_CreateStart(ProblemUserNumber,context,[CMFE_PROBLEM_FLUID_MECHANICS_CLASS,CMFE_PROBLEM_STOKES_EQUATION_TYPE, &
       & CMFE_PROBLEM_TRANSIENT_STOKES_SUBTYPE],Problem,Err)
  !Finish the creation of a problem.
  CALL cmfe_Problem_CreateFinish(Problem,Err)

  !
  !================================================================================================================================
  !

  !CONTROL LOOP

  !Start the creation of the problem control loop
  CALL cmfe_Problem_ControlLoopCreateStart(Problem,Err)
  !Get the control loop
  CALL cmfe_Problem_ControlLoopGet(Problem,CMFE_CONTROL_LOOP_NODE,ControlLoop,Err)
  !Set the times
  CALL cmfe_ControlLoop_TimesSet(ControlLoop,DYNAMIC_SOLVER_STOKES_START_TIME,DYNAMIC_SOLVER_STOKES_STOP_TIME, &
       & DYNAMIC_SOLVER_STOKES_TIME_INCREMENT,Err)
  !Set the output timing
  CALL cmfe_ControlLoop_TimeOutputSet(ControlLoop,DYNAMIC_SOLVER_STOKES_OUTPUT_FREQUENCY,Err)
  !Finish creating the problem control loop
  CALL cmfe_Problem_ControlLoopCreateFinish(Problem,Err)

  !
  !================================================================================================================================
  !

  !SOLVERS

  !Start the creation of the problem solvers
  CALL cmfe_Solver_Initialise(DynamicSolverStokes,Err)
  CALL cmfe_Solver_Initialise(LinearSolverStokes,Err)
  CALL cmfe_Problem_SolversCreateStart(Problem,Err)
  !Get the dynamic dymamic solver
  CALL cmfe_Problem_SolverGet(Problem,CMFE_CONTROL_LOOP_NODE,SolverStokesUserNumber,DynamicSolverStokes,Err)
  !Set the output type
  CALL cmfe_Solver_OutputTypeSet(DynamicSolverStokes,DYNAMIC_SOLVER_STOKES_OUTPUT_TYPE,Err)
  !Set theta
  CALL cmfe_Solver_DynamicThetaSet(DynamicSolverStokes,DYNAMIC_SOLVER_STOKES_THETA,Err)
  !Get the dynamic linear solver
  CALL cmfe_Solver_DynamicLinearSolverGet(DynamicSolverStokes,LinearSolverStokes,Err)
  !Set the output type
  CALL cmfe_Solver_OutputTypeSet(LinearSolverStokes,LINEAR_SOLVER_STOKES_OUTPUT_TYPE,Err)
  !Set the solver settings
  IF(LINEAR_SOLVER_STOKES_DIRECT_FLAG) THEN
     CALL cmfe_Solver_LinearTypeSet(LinearSolverStokes,CMFE_SOLVER_LINEAR_DIRECT_SOLVE_TYPE,Err)
     CALL cmfe_Solver_LibraryTypeSet(LinearSolverStokes,CMFE_SOLVER_MUMPS_LIBRARY,Err)
  ELSE
     CALL cmfe_Solver_LinearTypeSet(LinearSolverStokes,CMFE_SOLVER_LINEAR_ITERATIVE_SOLVE_TYPE,Err)
     CALL cmfe_Solver_LinearIterativeMaximumIterationsSet(LinearSolverStokes,MAXIMUM_ITERATIONS,Err)
     CALL cmfe_Solver_LinearIterativeDivergenceToleranceSet(LinearSolverStokes,DIVERGENCE_TOLERANCE,Err)
     CALL cmfe_Solver_LinearIterativeRelativeToleranceSet(LinearSolverStokes,RELATIVE_TOLERANCE,Err)
     CALL cmfe_Solver_LinearIterativeAbsoluteToleranceSet(LinearSolverStokes,ABSOLUTE_TOLERANCE,Err)
     CALL cmfe_Solver_LinearIterativeGMRESRestartSet(LinearSolverStokes,RESTART_VALUE,Err)
  ENDIF
  !Finish the creation of the problem solver
  CALL cmfe_Problem_SolversCreateFinish(Problem,Err)

  !
  !================================================================================================================================
  !

  !SOLVER EQUATIONS

  !Start the creation of the problem solver equations
  CALL cmfe_Solver_Initialise(DynamicSolverStokes,Err)
  CALL cmfe_SolverEquations_Initialise(SolverEquationsStokes,Err)

  CALL cmfe_Problem_SolverEquationsCreateStart(Problem,Err)
  !Get the dynamic solver equations
  CALL cmfe_Problem_SolverGet(Problem,CMFE_CONTROL_LOOP_NODE,SolverStokesUserNumber,DynamicSolverStokes,Err)
  CALL cmfe_Solver_SolverEquationsGet(DynamicSolverStokes,SolverEquationsStokes,Err)
  !Set the solver equations sparsity
  CALL cmfe_SolverEquations_SparsityTypeSet(SolverEquationsStokes,CMFE_SOLVER_SPARSE_MATRICES,Err)
  !Add in the equations set
  CALL cmfe_SolverEquations_EquationsSetAdd(SolverEquationsStokes,EquationsSetStokes,EquationsSetIndex,Err)
  !Finish the creation of the problem solver equations
  CALL cmfe_Problem_SolverEquationsCreateFinish(Problem,Err)

  !
  !================================================================================================================================
  !

  !BOUNDARY CONDITIONS

  !Start the creation of the equations set boundary conditions for Stokes
  CALL cmfe_BoundaryConditions_Initialise(BoundaryConditionsStokes,Err)
  CALL cmfe_SolverEquations_BoundaryConditionsCreateStart(SolverEquationsStokes,BoundaryConditionsStokes,Err)
  !Set fixed wall nodes
  IF(FIXED_WALL_NODES_STOKES_FLAG) THEN
     DO NODE_COUNTER=1,NUMBER_OF_FIXED_WALL_NODES_STOKES
        NODE_NUMBER=FIXED_WALL_NODES_STOKES(NODE_COUNTER)
        CONDITION=CMFE_BOUNDARY_CONDITION_FIXED_WALL
        CALL cmfe_Decomposition_NodeDomainGet(Decomposition,NODE_NUMBER,1,BoundaryNodeDomain,Err)
        IF(BoundaryNodeDomain==ComputationalNodeNumber) THEN
           DO COMPONENT_NUMBER=1,NUMBER_OF_DIMENSIONS
              VALUE=0.0_CMISSRP
              CALL cmfe_BoundaryConditions_SetNode(BoundaryConditionsStokes,DependentFieldStokes,CMFE_FIELD_U_VARIABLE_TYPE,1, &
                   & CMFE_NO_GLOBAL_DERIV, &
                   & NODE_NUMBER,COMPONENT_NUMBER,CONDITION,VALUE,Err)
           ENDDO
        ENDIF
     ENDDO
  ENDIF
  !Set velocity boundary conditions
  IF(INLET_WALL_NODES_STOKES_FLAG) THEN
     DO NODE_COUNTER=1,NUMBER_OF_INLET_WALL_NODES_STOKES
        NODE_NUMBER=INLET_WALL_NODES_STOKES(NODE_COUNTER)
        CONDITION=CMFE_BOUNDARY_CONDITION_FIXED_INLET
        CALL cmfe_Decomposition_NodeDomainGet(Decomposition,NODE_NUMBER,1,BoundaryNodeDomain,Err)
        IF(BoundaryNodeDomain==ComputationalNodeNumber) THEN
           DO COMPONENT_NUMBER=1,NUMBER_OF_DIMENSIONS
              VALUE=BOUNDARY_CONDITIONS_STOKES(COMPONENT_NUMBER)
              CALL cmfe_BoundaryConditions_SetNode(BoundaryConditionsStokes,DependentFieldStokes,CMFE_FIELD_U_VARIABLE_TYPE,1, &
                   & CMFE_NO_GLOBAL_DERIV, &
                   & NODE_NUMBER,COMPONENT_NUMBER,CONDITION,VALUE,Err)
           ENDDO
        ENDIF
     ENDDO
  ENDIF
  !CALL cmfe_SolverEquations_BoundaryConditionsAnalytic(SolverEquationsStokes,Err)
  !Finish the creation of the equations set boundary conditions
  CALL cmfe_SolverEquations_BoundaryConditionsCreateFinish(SolverEquationsStokes,Err)

  !
  !================================================================================================================================
  !

  !Output Analytic analysis
  !Call cmfe_AnalyticAnalysis_Output(DependentFieldStokes,"StokesAnalytic",Err)

  !
  !================================================================================================================================
  !

  !Solve the problem
  WRITE(*,'(A)') "Solving problem..."
  CALL cmfe_Problem_Solve(Problem,Err)
  WRITE(*,'(A)') "Problem solved!"

  !
  !================================================================================================================================
  !

  !OUTPUT

  EXPORT_FIELD_IO=.FALSE.
  IF(EXPORT_FIELD_IO) THEN
     WRITE(*,'(A)') "Exporting fields..."
     CALL cmfe_Fields_Initialise(Fields,Err)
     CALL cmfe_Fields_Create(Region,Fields,Err)
     CALL cmfe_Fields_NodesExport(Fields,"stokes_dynamic","FORTRAN",Err)
     CALL cmfe_Fields_ElementsExport(Fields,"stokes_dynamic","FORTRAN",Err)
     CALL cmfe_Fields_Finalise(Fields,Err)
     WRITE(*,'(A)') "Field exported!"
  ENDIF

  !Finialise CMISS
  CALL cmfe_Finalise(context,Err)
  WRITE(*,'(A)') "Program successfully completed."
  STOP

END PROGRAM stokes_dynamic
