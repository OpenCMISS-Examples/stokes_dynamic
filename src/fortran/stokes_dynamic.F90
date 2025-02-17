PROGRAM StokesDynamic

  !
  !================================================================================================================================
  !

  USE OpenCMISS

  !Test program parameters
  REAL(OC_RP), PARAMETER :: HEIGHT=3.0_OC_RP
  REAL(OC_RP), PARAMETER :: WIDTH=1.0_OC_RP
  REAL(OC_RP), PARAMETER :: LENGTH=1.0_OC_RP

  INTEGER(OC_Intg), PARAMETER :: ContextUserNumber=1
  INTEGER(OC_Intg), PARAMETER :: CoordinateSystemUserNumber=2
  INTEGER(OC_Intg), PARAMETER :: RegionUserNumber=3
  INTEGER(OC_Intg), PARAMETER :: MeshUserNumber=4
  INTEGER(OC_Intg), PARAMETER :: DecompositionUserNumber=5
  INTEGER(OC_Intg), PARAMETER :: DecomposerUserNumber=6
  INTEGER(OC_Intg), PARAMETER :: GeometricFieldUserNumber=7
  INTEGER(OC_Intg), PARAMETER :: EquationsSetFieldUserNumber=8
  INTEGER(OC_Intg), PARAMETER :: DependentFieldUserNumberStokes=9
  INTEGER(OC_Intg), PARAMETER :: MaterialsFieldUserNumberStokes=10
  INTEGER(OC_Intg), PARAMETER :: EquationsSetUserNumberStokes=11
  INTEGER(OC_Intg), PARAMETER :: ProblemUserNumber=12
  INTEGER(OC_Intg), PARAMETER :: GeneratedMeshUserNumber=13
  INTEGER(OC_Intg), PARAMETER :: AnalyticFieldUserNumber=14

  INTEGER(OC_Intg), PARAMETER :: DomainUserNumber=2
  INTEGER(OC_Intg), PARAMETER :: SolverStokesUserNumber=1
  INTEGER(OC_Intg), PARAMETER :: MaterialsFieldUserNumberStokesMu=1
  INTEGER(OC_Intg), PARAMETER :: MaterialsFieldUserNumberStokesRho=2

  !Program types
  !Program variables

  INTEGER(OC_Intg) :: NUMBER_OF_DIMENSIONS
  INTEGER(OC_Intg) :: NUMBER_GLOBAL_X_ELEMENTS,NUMBER_GLOBAL_Y_ELEMENTS,NUMBER_GLOBAL_Z_ELEMENTS
  INTEGER(OC_Intg) :: BASIS_TYPE
  INTEGER(OC_Intg) :: BASIS_NUMBER_SPACE
  INTEGER(OC_Intg) :: BASIS_NUMBER_VELOCITY
  INTEGER(OC_Intg) :: BASIS_NUMBER_PRESSURE
  INTEGER(OC_Intg) :: BASIS_XI_GAUSS_SPACE
  INTEGER(OC_Intg) :: BASIS_XI_GAUSS_VELOCITY
  INTEGER(OC_Intg) :: BASIS_XI_GAUSS_PRESSURE
  INTEGER(OC_Intg) :: BASIS_XI_INTERPOLATION_SPACE
  INTEGER(OC_Intg) :: BASIS_XI_INTERPOLATION_VELOCITY
  INTEGER(OC_Intg) :: BASIS_XI_INTERPOLATION_PRESSURE
  INTEGER(OC_Intg) :: MESH_NUMBER_OF_COMPONENTS
  INTEGER(OC_Intg) :: MESH_COMPONENT_NUMBER_SPACE
  INTEGER(OC_Intg) :: MESH_COMPONENT_NUMBER_VELOCITY
  INTEGER(OC_Intg) :: MESH_COMPONENT_NUMBER_PRESSURE
  INTEGER(OC_Intg) :: NUMBER_OF_NODES_SPACE
  INTEGER(OC_Intg) :: NUMBER_OF_NODES_VELOCITY
  INTEGER(OC_Intg) :: NUMBER_OF_NODES_PRESSURE
  INTEGER(OC_Intg) :: NUMBER_OF_ELEMENT_NODES_SPACE
  INTEGER(OC_Intg) :: NUMBER_OF_ELEMENT_NODES_VELOCITY
  INTEGER(OC_Intg) :: NUMBER_OF_ELEMENT_NODES_PRESSURE
  INTEGER(OC_Intg) :: TOTAL_NUMBER_OF_NODES
  INTEGER(OC_Intg) :: TOTAL_NUMBER_OF_ELEMENTS
  INTEGER(OC_Intg) :: MAXIMUM_ITERATIONS
  INTEGER(OC_Intg) :: RESTART_VALUE
  !   INTEGER(OC_Intg) :: MPI_IERROR
  INTEGER(OC_Intg) :: NUMBER_OF_FIXED_WALL_NODES_STOKES
  INTEGER(OC_Intg) :: NUMBER_OF_INLET_WALL_NODES_STOKES

  INTEGER(OC_Intg) :: EQUATIONS_STOKES_OUTPUT
  INTEGER(OC_Intg) :: COMPONENT_NUMBER
  INTEGER(OC_Intg) :: NODE_NUMBER
  INTEGER(OC_Intg) :: ELEMENT_NUMBER
  INTEGER(OC_Intg) :: NODE_COUNTER
  INTEGER(OC_Intg) :: CONDITION

  INTEGER(OC_Intg) :: DYNAMIC_SOLVER_STOKES_OUTPUT_FREQUENCY
  INTEGER(OC_Intg) :: DYNAMIC_SOLVER_STOKES_OUTPUT_TYPE
  INTEGER(OC_Intg) :: LINEAR_SOLVER_STOKES_OUTPUT_TYPE

  INTEGER, ALLOCATABLE, DIMENSION(:):: FIXED_WALL_NODES_STOKES
  INTEGER, ALLOCATABLE, DIMENSION(:):: INLET_WALL_NODES_STOKES

  REAL(OC_RP) :: INITIAL_FIELD_STOKES(3)
  REAL(OC_RP) :: BOUNDARY_CONDITIONS_STOKES(3)
  REAL(OC_RP) :: DIVERGENCE_TOLERANCE
  REAL(OC_RP) :: RELATIVE_TOLERANCE
  REAL(OC_RP) :: ABSOLUTE_TOLERANCE
  REAL(OC_RP) :: LINESEARCH_ALPHA
  REAL(OC_RP) :: VALUE
  REAL(OC_RP) :: MU_PARAM_STOKES
  REAL(OC_RP) :: RHO_PARAM_STOKES

  REAL(OC_RP) :: DYNAMIC_SOLVER_STOKES_START_TIME
  REAL(OC_RP) :: DYNAMIC_SOLVER_STOKES_STOP_TIME
  REAL(OC_RP) :: DYNAMIC_SOLVER_STOKES_THETA
  REAL(OC_RP) :: DYNAMIC_SOLVER_STOKES_TIME_INCREMENT

  LOGICAL :: EXPORT_FIELD_IO
  LOGICAL :: LINEAR_SOLVER_STOKES_DIRECT_FLAG
  LOGICAL :: FIXED_WALL_NODES_STOKES_FLAG
  LOGICAL :: INLET_WALL_NODES_STOKES_FLAG

  !CMISS variables

  TYPE(OC_RegionType) :: Region
  TYPE(OC_RegionType) :: WorldRegion
  TYPE(OC_ComputationEnvironmentType) :: computationEnvironment
  TYPE(OC_ContextType) :: context
  TYPE(OC_CoordinateSystemType) :: CoordinateSystem
  TYPE(OC_BasisType) :: BasisGeometry
  TYPE(OC_BasisType) :: BasisVelocity
  TYPE(OC_BasisType) :: BasisPressure
  TYPE(OC_NodesType) :: Nodes
  TYPE(OC_MeshElementsType) :: MeshElementsSpace
  TYPE(OC_MeshElementsType) :: MeshElementsVelocity
  TYPE(OC_MeshElementsType) :: MeshElementsPressure
  TYPE(OC_MeshType) :: Mesh
  TYPE(OC_GeneratedMeshType) :: GeneratedMesh
  TYPE(OC_DecompositionType) :: Decomposition
  TYPE(OC_DecomposerType) :: Decomposer
  TYPE(OC_FieldsType) :: Fields
  TYPE(OC_FieldType) :: GeometricField,AnalyticField
  TYPE(OC_FieldType) :: EquationsSetField
  TYPE(OC_FieldType) :: DependentFieldStokes
  TYPE(OC_FieldType) :: MaterialsFieldStokes
  TYPE(OC_BoundaryConditionsType) :: BoundaryConditionsStokes
  TYPE(OC_EquationsSetType) :: EquationsSetStokes
  TYPE(OC_EquationsType) :: EquationsStokes
  TYPE(OC_ProblemType) :: Problem
  TYPE(OC_ControlLoopType) :: ControlLoop
  TYPE(OC_SolverType) :: DynamicSolverStokes
  TYPE(OC_SolverType) :: LinearSolverStokes
  TYPE(OC_SolverEquationsType) :: SolverEquationsStokes
  TYPE(OC_WorkGroupType) :: worldWorkGroup

  !Generic CMISS variables

  INTEGER(OC_Intg) :: NumberOfComputationalNodes,ComputationalNodeNumber,BoundaryNodeDomain
  INTEGER(OC_Intg) :: DecompositionIndex,EquationsSetIndex,Err

  !
  !================================================================================================================================
  !

  !INITIALISE OPENCMISS

  CALL OC_Initialise(err)
  CALL OC_ErrorHandlingModeSet(OC_ERRORS_TRAP_ERROR,err)
  !Create a context
  CALL OC_Context_Initialise(context,err)
  CALL OC_Context_Create(ContextUserNumber,context,err)
  CALL OC_Region_Initialise(worldRegion,err)
  CALL OC_Context_WorldRegionGet(context,worldRegion,err)

  !
  !================================================================================================================================
  !

  !CHECK COMPUTATIONAL NODE

  !Get the computational nodes information
  CALL OC_ComputationEnvironment_Initialise(computationEnvironment,err)
  CALL OC_Context_ComputationEnvironmentGet(context,computationEnvironment,err)
  
  CALL OC_WorkGroup_Initialise(worldWorkGroup,err)
  CALL OC_ComputationEnvironment_WorldWorkGroupGet(computationEnvironment,worldWorkGroup,err)
  CALL OC_WorkGroup_NumberOfGroupNodesGet(worldWorkGroup,numberOfComputationalNodes,err)
  CALL OC_WorkGroup_GroupNodeNumberGet(worldWorkGroup,computationalNodeNumber,err)

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
  BASIS_TYPE=OC_BASIS_LINEAR_LAGRANGE_INTERPOLATION
  BASIS_XI_INTERPOLATION_SPACE=OC_BASIS_LINEAR_LAGRANGE_INTERPOLATION
  BASIS_XI_INTERPOLATION_VELOCITY=OC_BASIS_LINEAR_LAGRANGE_INTERPOLATION
  BASIS_XI_INTERPOLATION_PRESSURE=OC_BASIS_LINEAR_LAGRANGE_INTERPOLATION

  !Set initial values
  INITIAL_FIELD_STOKES(1)=0.0_OC_RP
  INITIAL_FIELD_STOKES(2)=0.0_OC_RP
  INITIAL_FIELD_STOKES(3)=0.0_OC_RP
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
     BOUNDARY_CONDITIONS_STOKES(1)=0.0_OC_RP
     BOUNDARY_CONDITIONS_STOKES(2)=1.0_OC_RP
     BOUNDARY_CONDITIONS_STOKES(3)=0.0_OC_RP
  ENDIF
  !Set material parameters
  MU_PARAM_STOKES=1.0_OC_RP
  RHO_PARAM_STOKES=1.0_OC_RP
  !Set interpolation parameters
  BASIS_XI_GAUSS_SPACE=3
  BASIS_XI_GAUSS_VELOCITY=3
  BASIS_XI_GAUSS_PRESSURE=3
  !Set output parameter
  !(NoOutput/ProgressOutput/TimingOutput/SolverOutput/SolverMatrixOutput)
  DYNAMIC_SOLVER_STOKES_OUTPUT_TYPE=OC_SOLVER_NO_OUTPUT
  LINEAR_SOLVER_STOKES_OUTPUT_TYPE=OC_SOLVER_NO_OUTPUT
  !(NoOutput/TimingOutput/MatrixOutput/ElementOutput)
  EQUATIONS_STOKES_OUTPUT=OC_EQUATIONS_NO_OUTPUT
  !Set time parameter
  DYNAMIC_SOLVER_STOKES_START_TIME=0.0_OC_RP
  DYNAMIC_SOLVER_STOKES_STOP_TIME=5.0_OC_RP
  DYNAMIC_SOLVER_STOKES_TIME_INCREMENT=1.0_OC_RP
  DYNAMIC_SOLVER_STOKES_THETA=1.0_OC_RP
  !Set result output parameter
  DYNAMIC_SOLVER_STOKES_OUTPUT_FREQUENCY=1
  !Set solver parameters
  LINEAR_SOLVER_STOKES_DIRECT_FLAG=.FALSE.
  RELATIVE_TOLERANCE=1.0E-10_OC_RP !default: 1.0E-05_OC_RP
  ABSOLUTE_TOLERANCE=1.0E-10_OC_RP !default: 1.0E-10_OC_RP
  DIVERGENCE_TOLERANCE=1.0E20 !default: 1.0E5
  MAXIMUM_ITERATIONS=100000 !default: 100000
  RESTART_VALUE=3000 !default: 30
  LINESEARCH_ALPHA=1.0

  !
  !================================================================================================================================
  !

  !COORDINATE SYSTEM

  !Start the creation of a new RC coordinate system
  CALL OC_CoordinateSystem_Initialise(CoordinateSystem,Err)
  CALL OC_CoordinateSystem_CreateStart(CoordinateSystemUserNumber,context,CoordinateSystem,Err)
  !Set the coordinate system dimension
  CALL OC_CoordinateSystem_DimensionSet(CoordinateSystem,NUMBER_OF_DIMENSIONS,Err)
  !Finish the creation of the coordinate system
  CALL OC_CoordinateSystem_CreateFinish(CoordinateSystem,Err)

  !
  !================================================================================================================================
  !

  !REGION

  !Start the creation of a new region
  CALL OC_Region_Initialise(Region,Err)
  CALL OC_Region_CreateStart(RegionUserNumber,WorldRegion,Region,Err)
  CALL OC_Region_LabelSet(Region,"StokesRegion",Err)
  !Set the regions coordinate system as defined above
  CALL OC_Region_CoordinateSystemSet(Region,CoordinateSystem,Err)
  !Finish the creation of the region
  CALL OC_Region_CreateFinish(Region,Err)

  !
  !================================================================================================================================
  !

  !BASES

  !Start the creation of new bases
  MESH_NUMBER_OF_COMPONENTS=1
  CALL OC_Basis_Initialise(BasisGeometry,Err)
  CALL OC_Basis_CreateStart(BASIS_NUMBER_SPACE,context,BasisGeometry,Err)
  !Set the basis type (Lagrange/Simplex)
  CALL OC_Basis_TypeSet(BasisGeometry,BASIS_TYPE,Err)
  !Set the basis xi number
  CALL OC_Basis_NumberOfXiSet(BasisGeometry,NUMBER_OF_DIMENSIONS,Err)
  !Set the basis xi interpolation and number of Gauss points
  IF(NUMBER_OF_DIMENSIONS==2) THEN
     CALL OC_Basis_InterpolationXiSet(BasisGeometry,[BASIS_XI_INTERPOLATION_SPACE,BASIS_XI_INTERPOLATION_SPACE],Err)
     CALL OC_Basis_QuadratureNumberOfGaussXiSet(BasisGeometry,[BASIS_XI_GAUSS_SPACE,BASIS_XI_GAUSS_SPACE],Err)
  ELSE
     CALL OC_Basis_InterpolationXiSet(BasisGeometry,[BASIS_XI_INTERPOLATION_SPACE,BASIS_XI_INTERPOLATION_SPACE, &
          & BASIS_XI_INTERPOLATION_SPACE],Err)
     CALL OC_Basis_QuadratureNumberOfGaussXiSet(BasisGeometry,[BASIS_XI_GAUSS_SPACE,BASIS_XI_GAUSS_SPACE,BASIS_XI_GAUSS_SPACE], &
          & Err)
  ENDIF
  !Finish the creation of the basis
  CALL OC_Basis_CreateFinish(BasisGeometry,Err)

  !Start the creation of another basis
  IF(BASIS_XI_INTERPOLATION_VELOCITY==BASIS_XI_INTERPOLATION_SPACE) THEN
     BasisVelocity=BasisGeometry
  ELSE
     MESH_NUMBER_OF_COMPONENTS=MESH_NUMBER_OF_COMPONENTS+1
     !Initialise a new velocity basis
     CALL OC_Basis_Initialise(BasisVelocity,Err)
     !Start the creation of a basis
     CALL OC_Basis_CreateStart(BASIS_NUMBER_VELOCITY,context,BasisVelocity,Err)
     !Set the basis type (Lagrange/Simplex)
     CALL OC_Basis_TypeSet(BasisVelocity,BASIS_TYPE,Err)
     !Set the basis xi number
     CALL OC_Basis_NumberOfXiSet(BasisVelocity,NUMBER_OF_DIMENSIONS,Err)
     !Set the basis xi interpolation and number of Gauss points
     IF(NUMBER_OF_DIMENSIONS==2) THEN
        CALL OC_Basis_InterpolationXiSet(BasisVelocity,[BASIS_XI_INTERPOLATION_VELOCITY,BASIS_XI_INTERPOLATION_VELOCITY],Err)
        CALL OC_Basis_QuadratureNumberOfGaussXiSet(BasisVelocity,[BASIS_XI_GAUSS_VELOCITY,BASIS_XI_GAUSS_VELOCITY],Err)
     ELSE
        CALL OC_Basis_InterpolationXiSet(BasisVelocity,[BASIS_XI_INTERPOLATION_VELOCITY,BASIS_XI_INTERPOLATION_VELOCITY, &
             & BASIS_XI_INTERPOLATION_VELOCITY],Err)
        CALL OC_Basis_QuadratureNumberOfGaussXiSet(BasisVelocity,[BASIS_XI_GAUSS_VELOCITY,BASIS_XI_GAUSS_VELOCITY, &
             & BASIS_XI_GAUSS_VELOCITY],Err)
     ENDIF
     !Finish the creation of the basis
     CALL OC_Basis_CreateFinish(BasisVelocity,Err)
  ENDIF

  !Start the creation of another basis
  IF(BASIS_XI_INTERPOLATION_PRESSURE==BASIS_XI_INTERPOLATION_SPACE) THEN
     BasisPressure=BasisGeometry
  ELSE IF(BASIS_XI_INTERPOLATION_PRESSURE==BASIS_XI_INTERPOLATION_VELOCITY) THEN
     BasisPressure=BasisVelocity
  ELSE
     MESH_NUMBER_OF_COMPONENTS=MESH_NUMBER_OF_COMPONENTS+1
     !Initialise a new pressure basis
     CALL OC_Basis_Initialise(BasisPressure,Err)
     !Start the creation of a basis
     CALL OC_Basis_CreateStart(BASIS_NUMBER_PRESSURE,context,BasisPressure,Err)
     !Set the basis type (Lagrange/Simplex)
     CALL OC_Basis_TypeSet(BasisPressure,BASIS_TYPE,Err)
     !Set the basis xi number
     CALL OC_Basis_NumberOfXiSet(BasisPressure,NUMBER_OF_DIMENSIONS,Err)
     !Set the basis xi interpolation and number of Gauss points
     IF(NUMBER_OF_DIMENSIONS==2) THEN
        CALL OC_Basis_InterpolationXiSet(BasisPressure,[BASIS_XI_INTERPOLATION_PRESSURE,BASIS_XI_INTERPOLATION_PRESSURE],Err)
        CALL OC_Basis_QuadratureNumberOfGaussXiSet(BasisPressure,[BASIS_XI_GAUSS_PRESSURE,BASIS_XI_GAUSS_PRESSURE],Err)
     ELSE
        CALL OC_Basis_InterpolationXiSet(BasisPressure,[BASIS_XI_INTERPOLATION_PRESSURE,BASIS_XI_INTERPOLATION_PRESSURE, &
             & BASIS_XI_INTERPOLATION_PRESSURE],Err)
        CALL OC_Basis_QuadratureNumberOfGaussXiSet(BasisPressure,[BASIS_XI_GAUSS_PRESSURE,BASIS_XI_GAUSS_PRESSURE, &
             & BASIS_XI_GAUSS_PRESSURE],Err)
     ENDIF
     !Finish the creation of the basis
     CALL OC_Basis_CreateFinish(BasisPressure,Err)
  ENDIF

  !
  !================================================================================================================================
  !

  !MESH

  !Start the creation of a generated mesh in the region
  CALL OC_GeneratedMesh_Initialise(GeneratedMesh,Err)
  CALL OC_GeneratedMesh_CreateStart(GeneratedMeshUserNumber,Region,GeneratedMesh,Err)
  !Set up a regular x*y*z mesh
  CALL OC_GeneratedMesh_TypeSet(GeneratedMesh,OC_GENERATED_MESH_REGULAR_MESH_TYPE,Err)
  !Set the default basis
  CALL OC_GeneratedMesh_BasisSet(GeneratedMesh,BasisGeometry,Err)
  !Define the mesh on the region
  IF(NUMBER_OF_DIMENSIONS==2) THEN
     CALL OC_GeneratedMesh_ExtentSet(GeneratedMesh,[WIDTH,HEIGHT],Err)
     CALL OC_GeneratedMesh_NumberOfElementsSet(GeneratedMesh,[NUMBER_GLOBAL_X_ELEMENTS,NUMBER_GLOBAL_Y_ELEMENTS],Err)
  ELSE
     CALL OC_GeneratedMesh_ExtentSet(GeneratedMesh,[WIDTH,HEIGHT,LENGTH],Err)
     CALL OC_GeneratedMesh_NumberOfElementsSet(GeneratedMesh,[NUMBER_GLOBAL_X_ELEMENTS,NUMBER_GLOBAL_Y_ELEMENTS, &
          & NUMBER_GLOBAL_Z_ELEMENTS],Err)
  ENDIF
  !Finish the creation of a generated mesh in the region
  CALL OC_Mesh_Initialise(Mesh,Err)
  CALL OC_GeneratedMesh_CreateFinish(GeneratedMesh,MeshUserNumber,Mesh,Err)

  !
  !================================================================================================================================
  !

  !DECOMPOSITION

  !Create a decomposition
  CALL OC_Decomposition_Initialise(Decomposition,Err)
  CALL OC_Decomposition_CreateStart(DecompositionUserNumber,Mesh,Decomposition,Err)
  !Set the decomposition to be a general decomposition 
  CALL OC_Decomposition_TypeSet(Decomposition,OC_DECOMPOSITION_CALCULATED_TYPE,Err)
  !Finish the decomposition
  CALL OC_Decomposition_CreateFinish(Decomposition,Err)

  !Decompose
  CALL OC_Decomposer_Initialise(decomposer,err)
  CALL OC_Decomposer_CreateStart(decomposerUserNumber,region,worldWorkGroup,decomposer,err)
  !Add in the decomposition
  CALL OC_Decomposer_DecompositionAdd(decomposer,decomposition,decompositionIndex,err)
  !Finish the decomposer
  CALL OC_Decomposer_CreateFinish(decomposer,err)
  
  !
  !================================================================================================================================
  !

  !GEOMETRIC FIELD

  !Start to create a default (geometric) field on the region
  CALL OC_Field_Initialise(GeometricField,Err)
  CALL OC_Field_CreateStart(GeometricFieldUserNumber,Region,GeometricField,Err)
  !Set the field type
  CALL OC_Field_TypeSet(GeometricField,OC_FIELD_GEOMETRIC_TYPE,Err)
  !Set the decomposition to use
  CALL OC_Field_DecompositionSet(GeometricField,Decomposition,Err)
  !Set the scaling to use
  CALL OC_Field_ScalingTypeSet(GeometricField,OC_FIELD_NO_SCALING,Err)
  !Set the mesh component to be used by the field components.
  DO COMPONENT_NUMBER=1,NUMBER_OF_DIMENSIONS
     CALL OC_Field_ComponentMeshComponentSet(GeometricField,OC_FIELD_U_VARIABLE_TYPE,COMPONENT_NUMBER, &
          & MESH_COMPONENT_NUMBER_SPACE,Err)
  ENDDO
  !Finish creating the field
  CALL OC_Field_CreateFinish(GeometricField,Err)
  !Update the geometric field parameters
  CALL OC_GeneratedMesh_GeometricParametersCalculate(GeneratedMesh,GeometricField,Err)

  !
  !================================================================================================================================
  !

  !EQUATIONS SETS

  !Create the equations set for dynamic Stokes
  CALL OC_EquationsSet_Initialise(EquationsSetStokes,Err)
  CALL OC_Field_Initialise(EquationsSetField,Err)
  !Set the equations set to be a dynamic Stokes problem
  CALL OC_EquationsSet_CreateStart(EquationsSetUserNumberStokes,Region,GeometricField,[OC_EQUATIONS_SET_FLUID_MECHANICS_CLASS, &
       & OC_EQUATIONS_SET_STOKES_EQUATION_TYPE,OC_EQUATIONS_SET_TRANSIENT_STOKES_SUBTYPE],EquationsSetFieldUserNumber, &
       & EquationsSetField,EquationsSetStokes,Err)
  !Finish creating the equations set
  CALL OC_EquationsSet_CreateFinish(EquationsSetStokes,Err)

  !
  !================================================================================================================================
  !

  !DEPENDENT FIELDS

  !Create the equations set dependent field variables for dynamic Stokes
  CALL OC_Field_Initialise(DependentFieldStokes,Err)
  CALL OC_EquationsSet_DependentCreateStart(EquationsSetStokes,DependentFieldUserNumberStokes,DependentFieldStokes,Err)
  CALL OC_Field_VariableLabelSet(DependentFieldStokes,OC_FIELD_U_VARIABLE_TYPE,"U",Err)
  !Set the mesh component to be used by the field components.
  DO COMPONENT_NUMBER=1,NUMBER_OF_DIMENSIONS
     CALL OC_Field_ComponentMeshComponentSet(DependentFieldStokes,OC_FIELD_U_VARIABLE_TYPE,COMPONENT_NUMBER, &
          & MESH_COMPONENT_NUMBER_VELOCITY,Err)
     CALL OC_Field_ComponentMeshComponentSet(DependentFieldStokes,OC_FIELD_DELUDELN_VARIABLE_TYPE,COMPONENT_NUMBER, &
          & MESH_COMPONENT_NUMBER_VELOCITY,Err)
  ENDDO
  COMPONENT_NUMBER=NUMBER_OF_DIMENSIONS+1
  CALL OC_Field_ComponentMeshComponentSet(DependentFieldStokes,OC_FIELD_U_VARIABLE_TYPE,COMPONENT_NUMBER, &
       & MESH_COMPONENT_NUMBER_PRESSURE,Err)
  CALL OC_Field_ComponentMeshComponentSet(DependentFieldStokes,OC_FIELD_DELUDELN_VARIABLE_TYPE,COMPONENT_NUMBER, &
       & MESH_COMPONENT_NUMBER_PRESSURE,Err)
  !Finish the equations set dependent field variables
  CALL OC_EquationsSet_DependentCreateFinish(EquationsSetStokes,Err)

  !Initialise dependent field
  DO COMPONENT_NUMBER=1,NUMBER_OF_DIMENSIONS
     CALL OC_Field_ComponentValuesInitialise(DependentFieldStokes,OC_FIELD_U_VARIABLE_TYPE,OC_FIELD_VALUES_SET_TYPE, &
          & COMPONENT_NUMBER,INITIAL_FIELD_STOKES(COMPONENT_NUMBER),Err)
  ENDDO


  !
  !================================================================================================================================
  !

  !MATERIALS FIELDS

  !Create the equations set materials field variables for dynamic Stokes
  CALL OC_Field_Initialise(MaterialsFieldStokes,Err)
  CALL OC_EquationsSet_MaterialsCreateStart(EquationsSetStokes,MaterialsFieldUserNumberStokes,MaterialsFieldStokes,Err)
  CALL OC_Field_VariableLabelSet(MaterialsFieldStokes,OC_FIELD_U_VARIABLE_TYPE,"Materials",Err)
  !Finish the equations set materials field variables
  CALL OC_EquationsSet_MaterialsCreateFinish(EquationsSetStokes,Err)
  CALL OC_Field_ComponentValuesInitialise(MaterialsFieldStokes,OC_FIELD_U_VARIABLE_TYPE,OC_FIELD_VALUES_SET_TYPE, &
       & MaterialsFieldUserNumberStokesMu,MU_PARAM_STOKES,Err)
  CALL OC_Field_ComponentValuesInitialise(MaterialsFieldStokes,OC_FIELD_U_VARIABLE_TYPE,OC_FIELD_VALUES_SET_TYPE, &
       & MaterialsFieldUserNumberStokesRho,RHO_PARAM_STOKES,Err)

  !
  !================================================================================================================================
  !

  !ANALYTIC FIELD

  !Create the equations set analytic field variables
  !CALL OC_Field_Initialise(AnalyticField,Err)
  !CALL OC_EquationsSet_AnalyticCreateStart(EquationsSetStokes,OC_EQUATIONS_SET_NAVIER_STOKES_EQUATION_TWO_DIM_4, &
  !     & AnalyticFieldUserNumber,AnalyticField,Err)
  !Finish the equations set analytic field variables
  !CALL OC_EquationsSet_AnalyticCreateFinish(EquationsSetStokes,Err)

  !
  !================================================================================================================================
  !

  !EQUATIONS

  !Create the equations set equations
  CALL OC_Equations_Initialise(EquationsStokes,Err)
  CALL OC_EquationsSet_EquationsCreateStart(EquationsSetStokes,EquationsStokes,Err)
  !Set the equations matrices sparsity type
  CALL OC_Equations_SparsityTypeSet(EquationsStokes,OC_EQUATIONS_SPARSE_MATRICES,Err)
  !Set the equations lumping type
  CALL OC_Equations_LumpingTypeSet(EquationsStokes,OC_EQUATIONS_UNLUMPED_MATRICES,Err)
  !Set the equations set output
  CALL OC_Equations_OutputTypeSet(EquationsStokes,EQUATIONS_STOKES_OUTPUT,Err)
  !Finish the equations set equations
  CALL OC_EquationsSet_EquationsCreateFinish(EquationsSetStokes,Err)

  !
  !================================================================================================================================
  !

  !PROBLEMS

  !Start the creation of a problem.
  CALL OC_Problem_Initialise(Problem,Err)
  CALL OC_ControlLoop_Initialise(ControlLoop,Err)
  CALL OC_Problem_CreateStart(ProblemUserNumber,context,[OC_PROBLEM_FLUID_MECHANICS_CLASS,OC_PROBLEM_STOKES_EQUATION_TYPE, &
       & OC_PROBLEM_TRANSIENT_STOKES_SUBTYPE],Problem,Err)
  !Finish the creation of a problem.
  CALL OC_Problem_CreateFinish(Problem,Err)

  !
  !================================================================================================================================
  !

  !CONTROL LOOP

  !Start the creation of the problem control loop
  CALL OC_Problem_ControlLoopCreateStart(Problem,Err)
  !Get the control loop
  CALL OC_Problem_ControlLoopGet(Problem,OC_CONTROL_LOOP_NODE,ControlLoop,Err)
  !Set the times
  CALL OC_ControlLoop_TimesSet(ControlLoop,DYNAMIC_SOLVER_STOKES_START_TIME,DYNAMIC_SOLVER_STOKES_STOP_TIME, &
       & DYNAMIC_SOLVER_STOKES_TIME_INCREMENT,Err)
  !Set the output timing
  CALL OC_ControlLoop_TimeOutputSet(ControlLoop,DYNAMIC_SOLVER_STOKES_OUTPUT_FREQUENCY,Err)
  !Finish creating the problem control loop
  CALL OC_Problem_ControlLoopCreateFinish(Problem,Err)

  !
  !================================================================================================================================
  !

  !SOLVERS

  !Start the creation of the problem solvers
  CALL OC_Solver_Initialise(DynamicSolverStokes,Err)
  CALL OC_Solver_Initialise(LinearSolverStokes,Err)
  CALL OC_Problem_SolversCreateStart(Problem,Err)
  !Get the dynamic dymamic solver
  CALL OC_Problem_SolverGet(Problem,OC_CONTROL_LOOP_NODE,SolverStokesUserNumber,DynamicSolverStokes,Err)
  !Set the output type
  CALL OC_Solver_OutputTypeSet(DynamicSolverStokes,DYNAMIC_SOLVER_STOKES_OUTPUT_TYPE,Err)
  !Set theta
  CALL OC_Solver_DynamicThetaSet(DynamicSolverStokes,DYNAMIC_SOLVER_STOKES_THETA,Err)
  !Get the dynamic linear solver
  CALL OC_Solver_DynamicLinearSolverGet(DynamicSolverStokes,LinearSolverStokes,Err)
  !Set the output type
  CALL OC_Solver_OutputTypeSet(LinearSolverStokes,LINEAR_SOLVER_STOKES_OUTPUT_TYPE,Err)
  !Set the solver settings
  IF(LINEAR_SOLVER_STOKES_DIRECT_FLAG) THEN
     CALL OC_Solver_LinearTypeSet(LinearSolverStokes,OC_SOLVER_LINEAR_DIRECT_SOLVE_TYPE,Err)
     CALL OC_Solver_LibraryTypeSet(LinearSolverStokes,OC_SOLVER_MUMPS_LIBRARY,Err)
  ELSE
     CALL OC_Solver_LinearTypeSet(LinearSolverStokes,OC_SOLVER_LINEAR_ITERATIVE_SOLVE_TYPE,Err)
     CALL OC_Solver_LinearIterativeMaximumIterationsSet(LinearSolverStokes,MAXIMUM_ITERATIONS,Err)
     CALL OC_Solver_LinearIterativeDivergenceToleranceSet(LinearSolverStokes,DIVERGENCE_TOLERANCE,Err)
     CALL OC_Solver_LinearIterativeRelativeToleranceSet(LinearSolverStokes,RELATIVE_TOLERANCE,Err)
     CALL OC_Solver_LinearIterativeAbsoluteToleranceSet(LinearSolverStokes,ABSOLUTE_TOLERANCE,Err)
     CALL OC_Solver_LinearIterativeGMRESRestartSet(LinearSolverStokes,RESTART_VALUE,Err)
  ENDIF
  !Finish the creation of the problem solver
  CALL OC_Problem_SolversCreateFinish(Problem,Err)

  !
  !================================================================================================================================
  !

  !SOLVER EQUATIONS

  !Start the creation of the problem solver equations
  CALL OC_Solver_Initialise(DynamicSolverStokes,Err)
  CALL OC_SolverEquations_Initialise(SolverEquationsStokes,Err)

  CALL OC_Problem_SolverEquationsCreateStart(Problem,Err)
  !Get the dynamic solver equations
  CALL OC_Problem_SolverGet(Problem,OC_CONTROL_LOOP_NODE,SolverStokesUserNumber,DynamicSolverStokes,Err)
  CALL OC_Solver_SolverEquationsGet(DynamicSolverStokes,SolverEquationsStokes,Err)
  !Set the solver equations sparsity
  CALL OC_SolverEquations_SparsityTypeSet(SolverEquationsStokes,OC_SOLVER_SPARSE_MATRICES,Err)
  !Add in the equations set
  CALL OC_SolverEquations_EquationsSetAdd(SolverEquationsStokes,EquationsSetStokes,EquationsSetIndex,Err)
  !Finish the creation of the problem solver equations
  CALL OC_Problem_SolverEquationsCreateFinish(Problem,Err)

  !
  !================================================================================================================================
  !

  !BOUNDARY CONDITIONS

  !Start the creation of the equations set boundary conditions for Stokes
  CALL OC_BoundaryConditions_Initialise(BoundaryConditionsStokes,Err)
  CALL OC_SolverEquations_BoundaryConditionsCreateStart(SolverEquationsStokes,BoundaryConditionsStokes,Err)
  !Set fixed wall nodes
  IF(FIXED_WALL_NODES_STOKES_FLAG) THEN
     DO NODE_COUNTER=1,NUMBER_OF_FIXED_WALL_NODES_STOKES
        NODE_NUMBER=FIXED_WALL_NODES_STOKES(NODE_COUNTER)
        CONDITION=OC_BOUNDARY_CONDITION_FIXED_WALL
        CALL OC_Decomposition_NodeDomainGet(Decomposition,1,NODE_NUMBER,BoundaryNodeDomain,Err)
        IF(BoundaryNodeDomain==ComputationalNodeNumber) THEN
           DO COMPONENT_NUMBER=1,NUMBER_OF_DIMENSIONS
              VALUE=0.0_OC_RP
              CALL OC_BoundaryConditions_SetNode(BoundaryConditionsStokes,DependentFieldStokes,OC_FIELD_U_VARIABLE_TYPE,1, &
                   & OC_NO_GLOBAL_DERIV, &
                   & NODE_NUMBER,COMPONENT_NUMBER,CONDITION,VALUE,Err)
           ENDDO
        ENDIF
     ENDDO
  ENDIF
  !Set velocity boundary conditions
  IF(INLET_WALL_NODES_STOKES_FLAG) THEN
     DO NODE_COUNTER=1,NUMBER_OF_INLET_WALL_NODES_STOKES
        NODE_NUMBER=INLET_WALL_NODES_STOKES(NODE_COUNTER)
        CONDITION=OC_BOUNDARY_CONDITION_FIXED_INLET
        CALL OC_Decomposition_NodeDomainGet(Decomposition,1,NODE_NUMBER,BoundaryNodeDomain,Err)
        IF(BoundaryNodeDomain==ComputationalNodeNumber) THEN
           DO COMPONENT_NUMBER=1,NUMBER_OF_DIMENSIONS
              VALUE=BOUNDARY_CONDITIONS_STOKES(COMPONENT_NUMBER)
              CALL OC_BoundaryConditions_SetNode(BoundaryConditionsStokes,DependentFieldStokes,OC_FIELD_U_VARIABLE_TYPE,1, &
                   & OC_NO_GLOBAL_DERIV, &
                   & NODE_NUMBER,COMPONENT_NUMBER,CONDITION,VALUE,Err)
           ENDDO
        ENDIF
     ENDDO
  ENDIF
  !CALL OC_SolverEquations_BoundaryConditionsAnalytic(SolverEquationsStokes,Err)
  !Finish the creation of the equations set boundary conditions
  CALL OC_SolverEquations_BoundaryConditionsCreateFinish(SolverEquationsStokes,Err)

  !
  !================================================================================================================================
  !

  !Output Analytic analysis
  !Call OC_AnalyticAnalysis_Output(DependentFieldStokes,"StokesAnalytic",Err)

  !
  !================================================================================================================================
  !

  !Solve the problem
  WRITE(*,'(A)') "Solving problem..."
  CALL OC_Problem_Solve(Problem,Err)
  WRITE(*,'(A)') "Problem solved!"

  !
  !================================================================================================================================
  !

  !OUTPUT

  EXPORT_FIELD_IO=.FALSE.
  IF(EXPORT_FIELD_IO) THEN
     WRITE(*,'(A)') "Exporting fields..."
     CALL OC_Fields_Initialise(Fields,Err)
     CALL OC_Fields_Create(Region,Fields,Err)
     CALL OC_Fields_NodesExport(Fields,"stokes_dynamic","FORTRAN",Err)
     CALL OC_Fields_ElementsExport(Fields,"stokes_dynamic","FORTRAN",Err)
     CALL OC_Fields_Finalise(Fields,Err)
     WRITE(*,'(A)') "Field exported!"
  ENDIF

  !Destroy the context
  CALL OC_Context_Destroy(context,err)
  !Finialise OpenCMISS
  CALL OC_Finalise(err)
  
  WRITE(*,'(A)') "Program successfully completed."
  STOP

END PROGRAM StokesDynamic
