Ϫ
��
�
ArgMax

input"T
	dimension"Tidx
output"output_type"!
Ttype:
2	
"
Tidxtype0:
2	"!
output_typetype0	:
2	
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( �
�
BiasAdd

value"T	
bias"T
output"T""
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
A
BroadcastArgs
s0"T
s1"T
r0"T"
Ttype0:
2	
Z
BroadcastTo

input"T
shape"Tidx
output"T"	
Ttype"
Tidxtype0:
2	
N
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype"
Truncatebool( 
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype
$
DisableCopyOnRead
resource�
.
Identity

input"T
output"T"	
Ttype
u
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:
2	
>
Maximum
x"T
y"T
z"T"
Ttype:
2	
�
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool("
allow_missing_filesbool( �
>
Minimum
x"T
y"T
z"T"
Ttype:
2	
?
Mul
x"T
y"T
z"T"
Ttype:
2	�

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
�
PartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype�
E
Relu
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
?
Select
	condition

t"T
e"T
output"T"	
Ttype
d
Shape

input"T&
output"out_type��out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
9
Softmax
logits"T
softmax"T"
Ttype:
2
�
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ��
@
StaticRegexFullMatch	
input

output
"
patternstring
�
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
L

StringJoin
inputs*N

output"

Nint("
	separatorstring 
�
Sum

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( ""
Ttype:
2	"
Tidxtype0:
2	
�
VarHandleOp
resource"
	containerstring "
shared_namestring "

debug_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 �"serve*2.15.12v2.15.0-11-g63f5a65c7cd8̔
�
ConstConst*
_output_shapes
:Q*
dtype0*�
value�B�Q"�  �� ��� ��� ��� ��� `�� @��  ��  �� ��� ��� ��� ��� `�� @��  ��  �� ��� ��� ��� ��� `�� @��  ��  �� �� ��� ��� ��� `�� @��  ��  �� ��� ��� ��� ��� `�� @��  ��  z� �s� �m� @g�  a� �Z� �T� @N�  H� �A� �;� @5�  /� �(� �"� @�  � �� �	� @�  �� ���  �� ���  �� ���  �� ���  �� ���  z�  a�  H�  /�  �  ��  ��  ��  H�  ��    
�
4CategoricalQNetwork/CategoricalQNetwork/dense_2/biasVarHandleOp*
_output_shapes
: *E

debug_name75CategoricalQNetwork/CategoricalQNetwork/dense_2/bias/*
dtype0*
shape:�*E
shared_name64CategoricalQNetwork/CategoricalQNetwork/dense_2/bias
�
HCategoricalQNetwork/CategoricalQNetwork/dense_2/bias/Read/ReadVariableOpReadVariableOp4CategoricalQNetwork/CategoricalQNetwork/dense_2/bias*
_output_shapes	
:�*
dtype0
�
6CategoricalQNetwork/CategoricalQNetwork/dense_2/kernelVarHandleOp*
_output_shapes
: *G

debug_name97CategoricalQNetwork/CategoricalQNetwork/dense_2/kernel/*
dtype0*
shape:
��*G
shared_name86CategoricalQNetwork/CategoricalQNetwork/dense_2/kernel
�
JCategoricalQNetwork/CategoricalQNetwork/dense_2/kernel/Read/ReadVariableOpReadVariableOp6CategoricalQNetwork/CategoricalQNetwork/dense_2/kernel* 
_output_shapes
:
��*
dtype0
�
DCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense_1/biasVarHandleOp*
_output_shapes
: *U

debug_nameGECategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense_1/bias/*
dtype0*
shape:�*U
shared_nameFDCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense_1/bias
�
XCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense_1/bias/Read/ReadVariableOpReadVariableOpDCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense_1/bias*
_output_shapes	
:�*
dtype0
�
FCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense_1/kernelVarHandleOp*
_output_shapes
: *W

debug_nameIGCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense_1/kernel/*
dtype0*
shape:
��*W
shared_nameHFCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense_1/kernel
�
ZCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense_1/kernel/Read/ReadVariableOpReadVariableOpFCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense_1/kernel* 
_output_shapes
:
��*
dtype0
�
BCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense/biasVarHandleOp*
_output_shapes
: *S

debug_nameECCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense/bias/*
dtype0*
shape:�*S
shared_nameDBCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense/bias
�
VCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense/bias/Read/ReadVariableOpReadVariableOpBCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense/bias*
_output_shapes	
:�*
dtype0
�
DCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense/kernelVarHandleOp*
_output_shapes
: *U

debug_nameGECategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense/kernel/*
dtype0*
shape:	�*U
shared_nameFDCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense/kernel
�
XCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense/kernel/Read/ReadVariableOpReadVariableOpDCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense/kernel*
_output_shapes
:	�*
dtype0

VariableVarHandleOp*
_output_shapes
: *

debug_name	Variable/*
dtype0	*
shape: *
shared_name
Variable
]
Variable/Read/ReadVariableOpReadVariableOpVariable*
_output_shapes
: *
dtype0	
l
action_0_discountPlaceholder*#
_output_shapes
:���������*
dtype0*
shape:���������
w
action_0_observationPlaceholder*'
_output_shapes
:���������*
dtype0*
shape:���������
j
action_0_rewardPlaceholder*#
_output_shapes
:���������*
dtype0*
shape:���������
m
action_0_step_typePlaceholder*#
_output_shapes
:���������*
dtype0*
shape:���������
�
StatefulPartitionedCallStatefulPartitionedCallaction_0_discountaction_0_observationaction_0_rewardaction_0_step_typeDCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense/kernelBCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense/biasFCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense_1/kernelDCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense_1/bias6CategoricalQNetwork/CategoricalQNetwork/dense_2/kernel4CategoricalQNetwork/CategoricalQNetwork/dense_2/biasConst*
Tin
2*
Tout
2	*
_collective_manager_ids
 *#
_output_shapes
:���������*(
_read_only_resource_inputs

	*-
config_proto

CPU

GPU 2J 8� *F
fAR?
=__inference_signature_wrapper_function_with_signature_2415700
]
get_initial_state_batch_sizePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�
PartitionedCallPartitionedCallget_initial_state_batch_size*
Tin
2*

Tout
 *
_collective_manager_ids
 * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *F
fAR?
=__inference_signature_wrapper_function_with_signature_2415710
�
PartitionedCall_1PartitionedCall*	
Tin
 *

Tout
 *
_collective_manager_ids
 * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *F
fAR?
=__inference_signature_wrapper_function_with_signature_2415728
�
StatefulPartitionedCall_1StatefulPartitionedCallVariable*
Tin
2*
Tout
2	*
_collective_manager_ids
 *
_output_shapes
: *#
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *F
fAR?
=__inference_signature_wrapper_function_with_signature_2415723

NoOpNoOp
�)
Const_1Const"/device:CPU:0*
_output_shapes
: *
dtype0*�(
value�(B�( B�(
�

train_step
metadata
model_variables
_all_assets

action
distribution
get_initial_state
get_metadata
	get_train_step


signatures*
GA
VARIABLE_VALUEVariable%train_step/.ATTRIBUTES/VARIABLE_VALUE*
* 
.
0
1
2
3
4
5*

_wrapped_policy*

trace_0
trace_1* 

trace_0* 

trace_0* 
* 
* 
K

action
get_initial_state
get_train_step
get_metadata* 
��
VARIABLE_VALUEDCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense/kernel,model_variables/0/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUEBCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense/bias,model_variables/1/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUEFCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense_1/kernel,model_variables/2/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUEDCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense_1/bias,model_variables/3/.ATTRIBUTES/VARIABLE_VALUE*
|v
VARIABLE_VALUE6CategoricalQNetwork/CategoricalQNetwork/dense_2/kernel,model_variables/4/.ATTRIBUTES/VARIABLE_VALUE*
zt
VARIABLE_VALUE4CategoricalQNetwork/CategoricalQNetwork/dense_2/bias,model_variables/5/.ATTRIBUTES/VARIABLE_VALUE*


_q_network*

	capture_6* 

	capture_6* 

	capture_6* 
* 

	capture_6* 
* 
* 
* 
�
	variables
trainable_variables
regularization_losses
	keras_api
 __call__
*!&call_and_return_all_conditional_losses
"
_q_network*
* 
.
0
1
2
3
4
5*
.
0
1
2
3
4
5*
* 
�
#non_trainable_variables

$layers
%metrics
&layer_regularization_losses
'layer_metrics
	variables
trainable_variables
regularization_losses
 __call__
*!&call_and_return_all_conditional_losses
&!"call_and_return_conditional_losses*
* 
* 
�
(	variables
)trainable_variables
*regularization_losses
+	keras_api
,__call__
*-&call_and_return_all_conditional_losses
._encoder
/_q_value_layer*
* 

"0*
* 
* 
* 
.
0
1
2
3
4
5*
.
0
1
2
3
4
5*
* 
�
0non_trainable_variables

1layers
2metrics
3layer_regularization_losses
4layer_metrics
(	variables
)trainable_variables
*regularization_losses
,__call__
*-&call_and_return_all_conditional_losses
&-"call_and_return_conditional_losses*
* 
* 
�
5	variables
6trainable_variables
7regularization_losses
8	keras_api
9__call__
*:&call_and_return_all_conditional_losses
;_postprocessing_layers*
�
<	variables
=trainable_variables
>regularization_losses
?	keras_api
@__call__
*A&call_and_return_all_conditional_losses

kernel
bias*
* 

.0
/1*
* 
* 
* 
 
0
1
2
3*
 
0
1
2
3*
* 
�
Bnon_trainable_variables

Clayers
Dmetrics
Elayer_regularization_losses
Flayer_metrics
5	variables
6trainable_variables
7regularization_losses
9__call__
*:&call_and_return_all_conditional_losses
&:"call_and_return_conditional_losses*
* 
* 

G0
H1
I2*

0
1*

0
1*
* 
�
Jnon_trainable_variables

Klayers
Lmetrics
Mlayer_regularization_losses
Nlayer_metrics
<	variables
=trainable_variables
>regularization_losses
@__call__
*A&call_and_return_all_conditional_losses
&A"call_and_return_conditional_losses*
* 
* 
* 

G0
H1
I2*
* 
* 
* 
�
O	variables
Ptrainable_variables
Qregularization_losses
R	keras_api
S__call__
*T&call_and_return_all_conditional_losses* 
�
U	variables
Vtrainable_variables
Wregularization_losses
X	keras_api
Y__call__
*Z&call_and_return_all_conditional_losses

kernel
bias*
�
[	variables
\trainable_variables
]regularization_losses
^	keras_api
___call__
*`&call_and_return_all_conditional_losses

kernel
bias*
* 
* 
* 
* 
* 
* 
* 
* 
�
anon_trainable_variables

blayers
cmetrics
dlayer_regularization_losses
elayer_metrics
O	variables
Ptrainable_variables
Qregularization_losses
S__call__
*T&call_and_return_all_conditional_losses
&T"call_and_return_conditional_losses* 
* 
* 

0
1*

0
1*
* 
�
fnon_trainable_variables

glayers
hmetrics
ilayer_regularization_losses
jlayer_metrics
U	variables
Vtrainable_variables
Wregularization_losses
Y__call__
*Z&call_and_return_all_conditional_losses
&Z"call_and_return_conditional_losses*
* 
* 

0
1*

0
1*
* 
�
knon_trainable_variables

llayers
mmetrics
nlayer_regularization_losses
olayer_metrics
[	variables
\trainable_variables
]regularization_losses
___call__
*`&call_and_return_all_conditional_losses
&`"call_and_return_conditional_losses*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameVariableDCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense/kernelBCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense/biasFCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense_1/kernelDCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense_1/bias6CategoricalQNetwork/CategoricalQNetwork/dense_2/kernel4CategoricalQNetwork/CategoricalQNetwork/dense_2/biasConst_1*
Tin
2	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *)
f$R"
 __inference__traced_save_2415985
�
StatefulPartitionedCall_3StatefulPartitionedCallsaver_filenameVariableDCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense/kernelBCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense/biasFCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense_1/kernelDCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense_1/bias6CategoricalQNetwork/CategoricalQNetwork/dense_2/kernel4CategoricalQNetwork/CategoricalQNetwork/dense_2/bias*
Tin

2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *,
f'R%
#__inference__traced_restore_2416015��
�f
�	
)__inference_polymorphic_action_fn_2415660
	time_step
time_step_1
time_step_2
time_step_3o
\categoricalqnetwork_categoricalqnetwork_encodingnetwork_dense_matmul_readvariableop_resource:	�l
]categoricalqnetwork_categoricalqnetwork_encodingnetwork_dense_biasadd_readvariableop_resource:	�r
^categoricalqnetwork_categoricalqnetwork_encodingnetwork_dense_1_matmul_readvariableop_resource:
��n
_categoricalqnetwork_categoricalqnetwork_encodingnetwork_dense_1_biasadd_readvariableop_resource:	�b
Ncategoricalqnetwork_categoricalqnetwork_dense_2_matmul_readvariableop_resource:
��^
Ocategoricalqnetwork_categoricalqnetwork_dense_2_biasadd_readvariableop_resource:	�	
mul_x
identity	��TCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOp�SCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense/MatMul/ReadVariableOp�VCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense_1/BiasAdd/ReadVariableOp�UCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense_1/MatMul/ReadVariableOp�FCategoricalQNetwork/CategoricalQNetwork/dense_2/BiasAdd/ReadVariableOp�ECategoricalQNetwork/CategoricalQNetwork/dense_2/MatMul/ReadVariableOp�
ECategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"����   �
GCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/flatten/ReshapeReshapetime_step_3NCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/flatten/Const:output:0*
T0*'
_output_shapes
:����������
BCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense/CastCastPCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/flatten/Reshape:output:0*

DstT0*

SrcT0*'
_output_shapes
:����������
SCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense/MatMul/ReadVariableOpReadVariableOp\categoricalqnetwork_categoricalqnetwork_encodingnetwork_dense_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype0�
DCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense/MatMulMatMulFCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense/Cast:y:0[CategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
TCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOpReadVariableOp]categoricalqnetwork_categoricalqnetwork_encodingnetwork_dense_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
ECategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense/BiasAddBiasAddNCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense/MatMul:product:0\CategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
BCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense/ReluReluNCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense/BiasAdd:output:0*
T0*(
_output_shapes
:�����������
UCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense_1/MatMul/ReadVariableOpReadVariableOp^categoricalqnetwork_categoricalqnetwork_encodingnetwork_dense_1_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
FCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense_1/MatMulMatMulPCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense/Relu:activations:0]CategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
VCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense_1/BiasAdd/ReadVariableOpReadVariableOp_categoricalqnetwork_categoricalqnetwork_encodingnetwork_dense_1_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
GCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense_1/BiasAddBiasAddPCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense_1/MatMul:product:0^CategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
DCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense_1/ReluReluPCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense_1/BiasAdd:output:0*
T0*(
_output_shapes
:�����������
ECategoricalQNetwork/CategoricalQNetwork/dense_2/MatMul/ReadVariableOpReadVariableOpNcategoricalqnetwork_categoricalqnetwork_dense_2_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
6CategoricalQNetwork/CategoricalQNetwork/dense_2/MatMulMatMulRCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense_1/Relu:activations:0MCategoricalQNetwork/CategoricalQNetwork/dense_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
FCategoricalQNetwork/CategoricalQNetwork/dense_2/BiasAdd/ReadVariableOpReadVariableOpOcategoricalqnetwork_categoricalqnetwork_dense_2_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
7CategoricalQNetwork/CategoricalQNetwork/dense_2/BiasAddBiasAdd@CategoricalQNetwork/CategoricalQNetwork/dense_2/MatMul:product:0NCategoricalQNetwork/CategoricalQNetwork/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������v
!CategoricalQNetwork/Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"����   Q   �
CategoricalQNetwork/ReshapeReshape@CategoricalQNetwork/CategoricalQNetwork/dense_2/BiasAdd:output:0*CategoricalQNetwork/Reshape/shape:output:0*
T0*+
_output_shapes
:���������Qn
SoftmaxSoftmax$CategoricalQNetwork/Reshape:output:0*
T0*+
_output_shapes
:���������QZ
mulMulmul_xSoftmax:softmax:0*
T0*+
_output_shapes
:���������Q`
Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
���������e
SumSummul:z:0Sum/reduction_indices:output:0*
T0*'
_output_shapes
:���������l
!Categorical/mode/ArgMax/dimensionConst*
_output_shapes
: *
dtype0*
valueB :
����������
Categorical/mode/ArgMaxArgMaxSum:output:0*Categorical/mode/ArgMax/dimension:output:0*
T0*#
_output_shapes
:���������T
Deterministic/atolConst*
_output_shapes
: *
dtype0	*
value	B	 R T
Deterministic/rtolConst*
_output_shapes
: *
dtype0	*
value	B	 R d
!Deterministic/sample/sample_shapeConst*
_output_shapes
: *
dtype0*
valueB x
Deterministic/sample/ShapeShape Categorical/mode/ArgMax:output:0*
T0	*
_output_shapes
::��\
Deterministic/sample/ConstConst*
_output_shapes
: *
dtype0*
value	B : r
(Deterministic/sample/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: t
*Deterministic/sample/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:t
*Deterministic/sample/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
"Deterministic/sample/strided_sliceStridedSlice#Deterministic/sample/Shape:output:01Deterministic/sample/strided_slice/stack:output:03Deterministic/sample/strided_slice/stack_1:output:03Deterministic/sample/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_maskh
%Deterministic/sample/BroadcastArgs/s0Const*
_output_shapes
: *
dtype0*
valueB j
'Deterministic/sample/BroadcastArgs/s0_1Const*
_output_shapes
: *
dtype0*
valueB �
"Deterministic/sample/BroadcastArgsBroadcastArgs0Deterministic/sample/BroadcastArgs/s0_1:output:0+Deterministic/sample/strided_slice:output:0*
_output_shapes
:n
$Deterministic/sample/concat/values_0Const*
_output_shapes
:*
dtype0*
valueB:g
$Deterministic/sample/concat/values_2Const*
_output_shapes
: *
dtype0*
valueB b
 Deterministic/sample/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Deterministic/sample/concatConcatV2-Deterministic/sample/concat/values_0:output:0'Deterministic/sample/BroadcastArgs:r0:0-Deterministic/sample/concat/values_2:output:0)Deterministic/sample/concat/axis:output:0*
N*
T0*
_output_shapes
:�
 Deterministic/sample/BroadcastToBroadcastTo Categorical/mode/ArgMax:output:0$Deterministic/sample/concat:output:0*
T0	*'
_output_shapes
:���������e
"Deterministic/sample/Shape_1/ConstConst*
_output_shapes
: *
dtype0*
valueB f
Deterministic/sample/Shape_1Const*
_output_shapes
:*
dtype0*
valueB: �
Deterministic/sample/Shape_2Shape)Deterministic/sample/BroadcastTo:output:0*
T0	*
_output_shapes
::��t
*Deterministic/sample/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:v
,Deterministic/sample/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: v
,Deterministic/sample/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
$Deterministic/sample/strided_slice_1StridedSlice%Deterministic/sample/Shape_2:output:03Deterministic/sample/strided_slice_1/stack:output:05Deterministic/sample/strided_slice_1/stack_1:output:05Deterministic/sample/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_maskd
"Deterministic/sample/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Deterministic/sample/concat_1ConcatV2*Deterministic/sample/sample_shape:output:0-Deterministic/sample/strided_slice_1:output:0+Deterministic/sample/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
Deterministic/sample/ReshapeReshape)Deterministic/sample/BroadcastTo:output:0&Deterministic/sample/concat_1:output:0*
T0	*#
_output_shapes
:���������Y
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0	*
value	B	 R�
clip_by_value/MinimumMinimum%Deterministic/sample/Reshape:output:0 clip_by_value/Minimum/y:output:0*
T0	*#
_output_shapes
:���������Q
clip_by_value/yConst*
_output_shapes
: *
dtype0	*
value	B	 R {
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0	*#
_output_shapes
:���������\
IdentityIdentityclip_by_value:z:0^NoOp*
T0	*#
_output_shapes
:����������
NoOpNoOpU^CategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOpT^CategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense/MatMul/ReadVariableOpW^CategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense_1/BiasAdd/ReadVariableOpV^CategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense_1/MatMul/ReadVariableOpG^CategoricalQNetwork/CategoricalQNetwork/dense_2/BiasAdd/ReadVariableOpF^CategoricalQNetwork/CategoricalQNetwork/dense_2/MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*e
_input_shapesT
R:���������:���������:���������:���������: : : : : : :Q2�
TCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOpTCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOp2�
SCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense/MatMul/ReadVariableOpSCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense/MatMul/ReadVariableOp2�
VCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense_1/BiasAdd/ReadVariableOpVCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense_1/BiasAdd/ReadVariableOp2�
UCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense_1/MatMul/ReadVariableOpUCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense_1/MatMul/ReadVariableOp2�
FCategoricalQNetwork/CategoricalQNetwork/dense_2/BiasAdd/ReadVariableOpFCategoricalQNetwork/CategoricalQNetwork/dense_2/BiasAdd/ReadVariableOp2�
ECategoricalQNetwork/CategoricalQNetwork/dense_2/MatMul/ReadVariableOpECategoricalQNetwork/CategoricalQNetwork/dense_2/MatMul/ReadVariableOp: 


_output_shapes
:Q:(	$
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:RN
'
_output_shapes
:���������
#
_user_specified_name	time_step:NJ
#
_output_shapes
:���������
#
_user_specified_name	time_step:NJ
#
_output_shapes
:���������
#
_user_specified_name	time_step:N J
#
_output_shapes
:���������
#
_user_specified_name	time_step
�
?
=__inference_signature_wrapper_function_with_signature_2415728�
PartitionedCallPartitionedCall*	
Tin
 *

Tout
 *
_collective_manager_ids
 *
_output_shapes
 * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *4
f/R-
+__inference_function_with_signature_2415725*(
_construction_contextkEagerRuntime*
_input_shapes 
�I
�

/__inference_polymorphic_distribution_fn_2415911
	step_type

reward
discount
observationo
\categoricalqnetwork_categoricalqnetwork_encodingnetwork_dense_matmul_readvariableop_resource:	�l
]categoricalqnetwork_categoricalqnetwork_encodingnetwork_dense_biasadd_readvariableop_resource:	�r
^categoricalqnetwork_categoricalqnetwork_encodingnetwork_dense_1_matmul_readvariableop_resource:
��n
_categoricalqnetwork_categoricalqnetwork_encodingnetwork_dense_1_biasadd_readvariableop_resource:	�b
Ncategoricalqnetwork_categoricalqnetwork_dense_2_matmul_readvariableop_resource:
��^
Ocategoricalqnetwork_categoricalqnetwork_dense_2_biasadd_readvariableop_resource:	�	
mul_x
identity	

identity_1	

identity_2	��TCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOp�SCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense/MatMul/ReadVariableOp�VCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense_1/BiasAdd/ReadVariableOp�UCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense_1/MatMul/ReadVariableOp�FCategoricalQNetwork/CategoricalQNetwork/dense_2/BiasAdd/ReadVariableOp�ECategoricalQNetwork/CategoricalQNetwork/dense_2/MatMul/ReadVariableOp�
ECategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"����   �
GCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/flatten/ReshapeReshapeobservationNCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/flatten/Const:output:0*
T0*'
_output_shapes
:����������
BCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense/CastCastPCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/flatten/Reshape:output:0*

DstT0*

SrcT0*'
_output_shapes
:����������
SCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense/MatMul/ReadVariableOpReadVariableOp\categoricalqnetwork_categoricalqnetwork_encodingnetwork_dense_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype0�
DCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense/MatMulMatMulFCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense/Cast:y:0[CategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
TCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOpReadVariableOp]categoricalqnetwork_categoricalqnetwork_encodingnetwork_dense_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
ECategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense/BiasAddBiasAddNCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense/MatMul:product:0\CategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
BCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense/ReluReluNCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense/BiasAdd:output:0*
T0*(
_output_shapes
:�����������
UCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense_1/MatMul/ReadVariableOpReadVariableOp^categoricalqnetwork_categoricalqnetwork_encodingnetwork_dense_1_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
FCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense_1/MatMulMatMulPCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense/Relu:activations:0]CategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
VCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense_1/BiasAdd/ReadVariableOpReadVariableOp_categoricalqnetwork_categoricalqnetwork_encodingnetwork_dense_1_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
GCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense_1/BiasAddBiasAddPCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense_1/MatMul:product:0^CategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
DCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense_1/ReluReluPCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense_1/BiasAdd:output:0*
T0*(
_output_shapes
:�����������
ECategoricalQNetwork/CategoricalQNetwork/dense_2/MatMul/ReadVariableOpReadVariableOpNcategoricalqnetwork_categoricalqnetwork_dense_2_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
6CategoricalQNetwork/CategoricalQNetwork/dense_2/MatMulMatMulRCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense_1/Relu:activations:0MCategoricalQNetwork/CategoricalQNetwork/dense_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
FCategoricalQNetwork/CategoricalQNetwork/dense_2/BiasAdd/ReadVariableOpReadVariableOpOcategoricalqnetwork_categoricalqnetwork_dense_2_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
7CategoricalQNetwork/CategoricalQNetwork/dense_2/BiasAddBiasAdd@CategoricalQNetwork/CategoricalQNetwork/dense_2/MatMul:product:0NCategoricalQNetwork/CategoricalQNetwork/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������v
!CategoricalQNetwork/Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"����   Q   �
CategoricalQNetwork/ReshapeReshape@CategoricalQNetwork/CategoricalQNetwork/dense_2/BiasAdd:output:0*CategoricalQNetwork/Reshape/shape:output:0*
T0*+
_output_shapes
:���������Qn
SoftmaxSoftmax$CategoricalQNetwork/Reshape:output:0*
T0*+
_output_shapes
:���������QZ
mulMulmul_xSoftmax:softmax:0*
T0*+
_output_shapes
:���������Q`
Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
���������e
SumSummul:z:0Sum/reduction_indices:output:0*
T0*'
_output_shapes
:���������l
!Categorical/mode/ArgMax/dimensionConst*
_output_shapes
: *
dtype0*
valueB :
����������
Categorical/mode/ArgMaxArgMaxSum:output:0*Categorical/mode/ArgMax/dimension:output:0*
T0*#
_output_shapes
:���������T
Deterministic/atolConst*
_output_shapes
: *
dtype0	*
value	B	 R T
Deterministic/rtolConst*
_output_shapes
: *
dtype0	*
value	B	 R Y
IdentityIdentityDeterministic/atol:output:0^NoOp*
T0	*
_output_shapes
: m

Identity_1Identity Categorical/mode/ArgMax:output:0^NoOp*
T0	*#
_output_shapes
:���������[

Identity_2IdentityDeterministic/rtol:output:0^NoOp*
T0	*
_output_shapes
: �
NoOpNoOpU^CategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOpT^CategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense/MatMul/ReadVariableOpW^CategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense_1/BiasAdd/ReadVariableOpV^CategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense_1/MatMul/ReadVariableOpG^CategoricalQNetwork/CategoricalQNetwork/dense_2/BiasAdd/ReadVariableOpF^CategoricalQNetwork/CategoricalQNetwork/dense_2/MatMul/ReadVariableOp*
_output_shapes
 "!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*e
_input_shapesT
R:���������:���������:���������:���������: : : : : : :Q2�
TCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOpTCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOp2�
SCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense/MatMul/ReadVariableOpSCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense/MatMul/ReadVariableOp2�
VCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense_1/BiasAdd/ReadVariableOpVCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense_1/BiasAdd/ReadVariableOp2�
UCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense_1/MatMul/ReadVariableOpUCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense_1/MatMul/ReadVariableOp2�
FCategoricalQNetwork/CategoricalQNetwork/dense_2/BiasAdd/ReadVariableOpFCategoricalQNetwork/CategoricalQNetwork/dense_2/BiasAdd/ReadVariableOp2�
ECategoricalQNetwork/CategoricalQNetwork/dense_2/MatMul/ReadVariableOpECategoricalQNetwork/CategoricalQNetwork/dense_2/MatMul/ReadVariableOp: 


_output_shapes
:Q:(	$
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:TP
'
_output_shapes
:���������
%
_user_specified_nameobservation:MI
#
_output_shapes
:���������
"
_user_specified_name
discount:KG
#
_output_shapes
:���������
 
_user_specified_namereward:N J
#
_output_shapes
:���������
#
_user_specified_name	step_type
�+
�
#__inference__traced_restore_2416015
file_prefix#
assignvariableop_variable:	 j
Wassignvariableop_1_categoricalqnetwork_categoricalqnetwork_encodingnetwork_dense_kernel:	�d
Uassignvariableop_2_categoricalqnetwork_categoricalqnetwork_encodingnetwork_dense_bias:	�m
Yassignvariableop_3_categoricalqnetwork_categoricalqnetwork_encodingnetwork_dense_1_kernel:
��f
Wassignvariableop_4_categoricalqnetwork_categoricalqnetwork_encodingnetwork_dense_1_bias:	�]
Iassignvariableop_5_categoricalqnetwork_categoricalqnetwork_dense_2_kernel:
��V
Gassignvariableop_6_categoricalqnetwork_categoricalqnetwork_dense_2_bias:	�

identity_8��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_2�AssignVariableOp_3�AssignVariableOp_4�AssignVariableOp_5�AssignVariableOp_6�
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*�
value�B�B%train_step/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/0/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/1/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/2/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/3/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/4/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/5/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*#
valueBB B B B B B B B �
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*4
_output_shapes"
 ::::::::*
dtypes

2	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0	*
_output_shapes
:�
AssignVariableOpAssignVariableOpassignvariableop_variableIdentity:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0	]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_1AssignVariableOpWassignvariableop_1_categoricalqnetwork_categoricalqnetwork_encodingnetwork_dense_kernelIdentity_1:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_2AssignVariableOpUassignvariableop_2_categoricalqnetwork_categoricalqnetwork_encodingnetwork_dense_biasIdentity_2:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_3AssignVariableOpYassignvariableop_3_categoricalqnetwork_categoricalqnetwork_encodingnetwork_dense_1_kernelIdentity_3:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_4AssignVariableOpWassignvariableop_4_categoricalqnetwork_categoricalqnetwork_encodingnetwork_dense_1_biasIdentity_4:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_5AssignVariableOpIassignvariableop_5_categoricalqnetwork_categoricalqnetwork_dense_2_kernelIdentity_5:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_6AssignVariableOpGassignvariableop_6_categoricalqnetwork_categoricalqnetwork_dense_2_biasIdentity_6:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0Y
NoOpNoOp"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 �

Identity_7Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^NoOp"/device:CPU:0*
T0*
_output_shapes
: U

Identity_8IdentityIdentity_7:output:0^NoOp_1*
T0*
_output_shapes
: �
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6*
_output_shapes
 "!

identity_8Identity_8:output:0*(
_construction_contextkEagerRuntime*#
_input_shapes
: : : : : : : : 2(
AssignVariableOp_1AssignVariableOp_12(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62$
AssignVariableOpAssignVariableOp:TP
N
_user_specified_name64CategoricalQNetwork/CategoricalQNetwork/dense_2/bias:VR
P
_user_specified_name86CategoricalQNetwork/CategoricalQNetwork/dense_2/kernel:d`
^
_user_specified_nameFDCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense_1/bias:fb
`
_user_specified_nameHFCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense_1/kernel:b^
\
_user_specified_nameDBCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense/bias:d`
^
_user_specified_nameFDCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense/kernel:($
"
_user_specified_name
Variable:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
�
O
=__inference_signature_wrapper_function_with_signature_2415710

batch_size�
PartitionedCallPartitionedCall
batch_size*
Tin
2*

Tout
 *
_collective_manager_ids
 *
_output_shapes
 * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *4
f/R-
+__inference_function_with_signature_2415706*(
_construction_contextkEagerRuntime*
_input_shapes
: :B >

_output_shapes
: 
$
_user_specified_name
batch_size
�f
�	
)__inference_polymorphic_action_fn_2415798
	step_type

reward
discount
observationo
\categoricalqnetwork_categoricalqnetwork_encodingnetwork_dense_matmul_readvariableop_resource:	�l
]categoricalqnetwork_categoricalqnetwork_encodingnetwork_dense_biasadd_readvariableop_resource:	�r
^categoricalqnetwork_categoricalqnetwork_encodingnetwork_dense_1_matmul_readvariableop_resource:
��n
_categoricalqnetwork_categoricalqnetwork_encodingnetwork_dense_1_biasadd_readvariableop_resource:	�b
Ncategoricalqnetwork_categoricalqnetwork_dense_2_matmul_readvariableop_resource:
��^
Ocategoricalqnetwork_categoricalqnetwork_dense_2_biasadd_readvariableop_resource:	�	
mul_x
identity	��TCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOp�SCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense/MatMul/ReadVariableOp�VCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense_1/BiasAdd/ReadVariableOp�UCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense_1/MatMul/ReadVariableOp�FCategoricalQNetwork/CategoricalQNetwork/dense_2/BiasAdd/ReadVariableOp�ECategoricalQNetwork/CategoricalQNetwork/dense_2/MatMul/ReadVariableOp�
ECategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"����   �
GCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/flatten/ReshapeReshapeobservationNCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/flatten/Const:output:0*
T0*'
_output_shapes
:����������
BCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense/CastCastPCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/flatten/Reshape:output:0*

DstT0*

SrcT0*'
_output_shapes
:����������
SCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense/MatMul/ReadVariableOpReadVariableOp\categoricalqnetwork_categoricalqnetwork_encodingnetwork_dense_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype0�
DCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense/MatMulMatMulFCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense/Cast:y:0[CategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
TCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOpReadVariableOp]categoricalqnetwork_categoricalqnetwork_encodingnetwork_dense_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
ECategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense/BiasAddBiasAddNCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense/MatMul:product:0\CategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
BCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense/ReluReluNCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense/BiasAdd:output:0*
T0*(
_output_shapes
:�����������
UCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense_1/MatMul/ReadVariableOpReadVariableOp^categoricalqnetwork_categoricalqnetwork_encodingnetwork_dense_1_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
FCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense_1/MatMulMatMulPCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense/Relu:activations:0]CategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
VCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense_1/BiasAdd/ReadVariableOpReadVariableOp_categoricalqnetwork_categoricalqnetwork_encodingnetwork_dense_1_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
GCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense_1/BiasAddBiasAddPCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense_1/MatMul:product:0^CategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
DCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense_1/ReluReluPCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense_1/BiasAdd:output:0*
T0*(
_output_shapes
:�����������
ECategoricalQNetwork/CategoricalQNetwork/dense_2/MatMul/ReadVariableOpReadVariableOpNcategoricalqnetwork_categoricalqnetwork_dense_2_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
6CategoricalQNetwork/CategoricalQNetwork/dense_2/MatMulMatMulRCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense_1/Relu:activations:0MCategoricalQNetwork/CategoricalQNetwork/dense_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
FCategoricalQNetwork/CategoricalQNetwork/dense_2/BiasAdd/ReadVariableOpReadVariableOpOcategoricalqnetwork_categoricalqnetwork_dense_2_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
7CategoricalQNetwork/CategoricalQNetwork/dense_2/BiasAddBiasAdd@CategoricalQNetwork/CategoricalQNetwork/dense_2/MatMul:product:0NCategoricalQNetwork/CategoricalQNetwork/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������v
!CategoricalQNetwork/Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"����   Q   �
CategoricalQNetwork/ReshapeReshape@CategoricalQNetwork/CategoricalQNetwork/dense_2/BiasAdd:output:0*CategoricalQNetwork/Reshape/shape:output:0*
T0*+
_output_shapes
:���������Qn
SoftmaxSoftmax$CategoricalQNetwork/Reshape:output:0*
T0*+
_output_shapes
:���������QZ
mulMulmul_xSoftmax:softmax:0*
T0*+
_output_shapes
:���������Q`
Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
���������e
SumSummul:z:0Sum/reduction_indices:output:0*
T0*'
_output_shapes
:���������l
!Categorical/mode/ArgMax/dimensionConst*
_output_shapes
: *
dtype0*
valueB :
����������
Categorical/mode/ArgMaxArgMaxSum:output:0*Categorical/mode/ArgMax/dimension:output:0*
T0*#
_output_shapes
:���������T
Deterministic/atolConst*
_output_shapes
: *
dtype0	*
value	B	 R T
Deterministic/rtolConst*
_output_shapes
: *
dtype0	*
value	B	 R d
!Deterministic/sample/sample_shapeConst*
_output_shapes
: *
dtype0*
valueB x
Deterministic/sample/ShapeShape Categorical/mode/ArgMax:output:0*
T0	*
_output_shapes
::��\
Deterministic/sample/ConstConst*
_output_shapes
: *
dtype0*
value	B : r
(Deterministic/sample/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: t
*Deterministic/sample/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:t
*Deterministic/sample/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
"Deterministic/sample/strided_sliceStridedSlice#Deterministic/sample/Shape:output:01Deterministic/sample/strided_slice/stack:output:03Deterministic/sample/strided_slice/stack_1:output:03Deterministic/sample/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_maskh
%Deterministic/sample/BroadcastArgs/s0Const*
_output_shapes
: *
dtype0*
valueB j
'Deterministic/sample/BroadcastArgs/s0_1Const*
_output_shapes
: *
dtype0*
valueB �
"Deterministic/sample/BroadcastArgsBroadcastArgs0Deterministic/sample/BroadcastArgs/s0_1:output:0+Deterministic/sample/strided_slice:output:0*
_output_shapes
:n
$Deterministic/sample/concat/values_0Const*
_output_shapes
:*
dtype0*
valueB:g
$Deterministic/sample/concat/values_2Const*
_output_shapes
: *
dtype0*
valueB b
 Deterministic/sample/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Deterministic/sample/concatConcatV2-Deterministic/sample/concat/values_0:output:0'Deterministic/sample/BroadcastArgs:r0:0-Deterministic/sample/concat/values_2:output:0)Deterministic/sample/concat/axis:output:0*
N*
T0*
_output_shapes
:�
 Deterministic/sample/BroadcastToBroadcastTo Categorical/mode/ArgMax:output:0$Deterministic/sample/concat:output:0*
T0	*'
_output_shapes
:���������e
"Deterministic/sample/Shape_1/ConstConst*
_output_shapes
: *
dtype0*
valueB f
Deterministic/sample/Shape_1Const*
_output_shapes
:*
dtype0*
valueB: �
Deterministic/sample/Shape_2Shape)Deterministic/sample/BroadcastTo:output:0*
T0	*
_output_shapes
::��t
*Deterministic/sample/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:v
,Deterministic/sample/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: v
,Deterministic/sample/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
$Deterministic/sample/strided_slice_1StridedSlice%Deterministic/sample/Shape_2:output:03Deterministic/sample/strided_slice_1/stack:output:05Deterministic/sample/strided_slice_1/stack_1:output:05Deterministic/sample/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_maskd
"Deterministic/sample/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Deterministic/sample/concat_1ConcatV2*Deterministic/sample/sample_shape:output:0-Deterministic/sample/strided_slice_1:output:0+Deterministic/sample/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
Deterministic/sample/ReshapeReshape)Deterministic/sample/BroadcastTo:output:0&Deterministic/sample/concat_1:output:0*
T0	*#
_output_shapes
:���������Y
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0	*
value	B	 R�
clip_by_value/MinimumMinimum%Deterministic/sample/Reshape:output:0 clip_by_value/Minimum/y:output:0*
T0	*#
_output_shapes
:���������Q
clip_by_value/yConst*
_output_shapes
: *
dtype0	*
value	B	 R {
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0	*#
_output_shapes
:���������\
IdentityIdentityclip_by_value:z:0^NoOp*
T0	*#
_output_shapes
:����������
NoOpNoOpU^CategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOpT^CategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense/MatMul/ReadVariableOpW^CategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense_1/BiasAdd/ReadVariableOpV^CategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense_1/MatMul/ReadVariableOpG^CategoricalQNetwork/CategoricalQNetwork/dense_2/BiasAdd/ReadVariableOpF^CategoricalQNetwork/CategoricalQNetwork/dense_2/MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*e
_input_shapesT
R:���������:���������:���������:���������: : : : : : :Q2�
TCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOpTCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOp2�
SCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense/MatMul/ReadVariableOpSCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense/MatMul/ReadVariableOp2�
VCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense_1/BiasAdd/ReadVariableOpVCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense_1/BiasAdd/ReadVariableOp2�
UCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense_1/MatMul/ReadVariableOpUCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense_1/MatMul/ReadVariableOp2�
FCategoricalQNetwork/CategoricalQNetwork/dense_2/BiasAdd/ReadVariableOpFCategoricalQNetwork/CategoricalQNetwork/dense_2/BiasAdd/ReadVariableOp2�
ECategoricalQNetwork/CategoricalQNetwork/dense_2/MatMul/ReadVariableOpECategoricalQNetwork/CategoricalQNetwork/dense_2/MatMul/ReadVariableOp: 


_output_shapes
:Q:(	$
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:TP
'
_output_shapes
:���������
%
_user_specified_nameobservation:MI
#
_output_shapes
:���������
"
_user_specified_name
discount:KG
#
_output_shapes
:���������
 
_user_specified_namereward:N J
#
_output_shapes
:���������
#
_user_specified_name	step_type
�
�
=__inference_signature_wrapper_function_with_signature_2415700
discount
observation

reward
	step_type
unknown:	�
	unknown_0:	�
	unknown_1:
��
	unknown_2:	�
	unknown_3:
��
	unknown_4:	�
	unknown_5
identity	��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCall	step_typerewarddiscountobservationunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5*
Tin
2*
Tout
2	*
_collective_manager_ids
 *#
_output_shapes
:���������*(
_read_only_resource_inputs

	*-
config_proto

CPU

GPU 2J 8� *4
f/R-
+__inference_function_with_signature_2415677k
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0	*#
_output_shapes
:���������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*e
_input_shapesT
R:���������:���������:���������:���������: : : : : : :Q22
StatefulPartitionedCallStatefulPartitionedCall: 


_output_shapes
:Q:'	#
!
_user_specified_name	2415694:'#
!
_user_specified_name	2415692:'#
!
_user_specified_name	2415690:'#
!
_user_specified_name	2415688:'#
!
_user_specified_name	2415686:'#
!
_user_specified_name	2415684:PL
#
_output_shapes
:���������
%
_user_specified_name0/step_type:MI
#
_output_shapes
:���������
"
_user_specified_name
0/reward:VR
'
_output_shapes
:���������
'
_user_specified_name0/observation:O K
#
_output_shapes
:���������
$
_user_specified_name
0/discount
�
}
=__inference_signature_wrapper_function_with_signature_2415723
unknown:	 
identity	��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallunknown*
Tin
2*
Tout
2	*
_collective_manager_ids
 *
_output_shapes
: *#
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *4
f/R-
+__inference_function_with_signature_2415716^
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0	*
_output_shapes
: <
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 22
StatefulPartitionedCallStatefulPartitionedCall:' #
!
_user_specified_name	2415719
�
-
+__inference_function_with_signature_2415725�
PartitionedCallPartitionedCall*	
Tin
 *

Tout
 *
_collective_manager_ids
 *
_output_shapes
 * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *%
f R
__inference_<lambda>_2415472*(
_construction_contextkEagerRuntime*
_input_shapes 
�K
�	
 __inference__traced_save_2415985
file_prefix)
read_disablecopyonread_variable:	 p
]read_1_disablecopyonread_categoricalqnetwork_categoricalqnetwork_encodingnetwork_dense_kernel:	�j
[read_2_disablecopyonread_categoricalqnetwork_categoricalqnetwork_encodingnetwork_dense_bias:	�s
_read_3_disablecopyonread_categoricalqnetwork_categoricalqnetwork_encodingnetwork_dense_1_kernel:
��l
]read_4_disablecopyonread_categoricalqnetwork_categoricalqnetwork_encodingnetwork_dense_1_bias:	�c
Oread_5_disablecopyonread_categoricalqnetwork_categoricalqnetwork_dense_2_kernel:
��\
Mread_6_disablecopyonread_categoricalqnetwork_categoricalqnetwork_dense_2_bias:	�
savev2_const_1
identity_15��MergeV2Checkpoints�Read/DisableCopyOnRead�Read/ReadVariableOp�Read_1/DisableCopyOnRead�Read_1/ReadVariableOp�Read_2/DisableCopyOnRead�Read_2/ReadVariableOp�Read_3/DisableCopyOnRead�Read_3/ReadVariableOp�Read_4/DisableCopyOnRead�Read_4/ReadVariableOp�Read_5/DisableCopyOnRead�Read_5/ReadVariableOp�Read_6/DisableCopyOnRead�Read_6/ReadVariableOpw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part�
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : �
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: q
Read/DisableCopyOnReadDisableCopyOnReadread_disablecopyonread_variable"/device:CPU:0*
_output_shapes
 �
Read/ReadVariableOpReadVariableOpread_disablecopyonread_variable^Read/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0	a
IdentityIdentityRead/ReadVariableOp:value:0"/device:CPU:0*
T0	*
_output_shapes
: Y

Identity_1IdentityIdentity:output:0"/device:CPU:0*
T0	*
_output_shapes
: �
Read_1/DisableCopyOnReadDisableCopyOnRead]read_1_disablecopyonread_categoricalqnetwork_categoricalqnetwork_encodingnetwork_dense_kernel"/device:CPU:0*
_output_shapes
 �
Read_1/ReadVariableOpReadVariableOp]read_1_disablecopyonread_categoricalqnetwork_categoricalqnetwork_encodingnetwork_dense_kernel^Read_1/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	�*
dtype0n

Identity_2IdentityRead_1/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	�d

Identity_3IdentityIdentity_2:output:0"/device:CPU:0*
T0*
_output_shapes
:	��
Read_2/DisableCopyOnReadDisableCopyOnRead[read_2_disablecopyonread_categoricalqnetwork_categoricalqnetwork_encodingnetwork_dense_bias"/device:CPU:0*
_output_shapes
 �
Read_2/ReadVariableOpReadVariableOp[read_2_disablecopyonread_categoricalqnetwork_categoricalqnetwork_encodingnetwork_dense_bias^Read_2/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0j

Identity_4IdentityRead_2/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�`

Identity_5IdentityIdentity_4:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_3/DisableCopyOnReadDisableCopyOnRead_read_3_disablecopyonread_categoricalqnetwork_categoricalqnetwork_encodingnetwork_dense_1_kernel"/device:CPU:0*
_output_shapes
 �
Read_3/ReadVariableOpReadVariableOp_read_3_disablecopyonread_categoricalqnetwork_categoricalqnetwork_encodingnetwork_dense_1_kernel^Read_3/DisableCopyOnRead"/device:CPU:0* 
_output_shapes
:
��*
dtype0o

Identity_6IdentityRead_3/ReadVariableOp:value:0"/device:CPU:0*
T0* 
_output_shapes
:
��e

Identity_7IdentityIdentity_6:output:0"/device:CPU:0*
T0* 
_output_shapes
:
���
Read_4/DisableCopyOnReadDisableCopyOnRead]read_4_disablecopyonread_categoricalqnetwork_categoricalqnetwork_encodingnetwork_dense_1_bias"/device:CPU:0*
_output_shapes
 �
Read_4/ReadVariableOpReadVariableOp]read_4_disablecopyonread_categoricalqnetwork_categoricalqnetwork_encodingnetwork_dense_1_bias^Read_4/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0j

Identity_8IdentityRead_4/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�`

Identity_9IdentityIdentity_8:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_5/DisableCopyOnReadDisableCopyOnReadOread_5_disablecopyonread_categoricalqnetwork_categoricalqnetwork_dense_2_kernel"/device:CPU:0*
_output_shapes
 �
Read_5/ReadVariableOpReadVariableOpOread_5_disablecopyonread_categoricalqnetwork_categoricalqnetwork_dense_2_kernel^Read_5/DisableCopyOnRead"/device:CPU:0* 
_output_shapes
:
��*
dtype0p
Identity_10IdentityRead_5/ReadVariableOp:value:0"/device:CPU:0*
T0* 
_output_shapes
:
��g
Identity_11IdentityIdentity_10:output:0"/device:CPU:0*
T0* 
_output_shapes
:
���
Read_6/DisableCopyOnReadDisableCopyOnReadMread_6_disablecopyonread_categoricalqnetwork_categoricalqnetwork_dense_2_bias"/device:CPU:0*
_output_shapes
 �
Read_6/ReadVariableOpReadVariableOpMread_6_disablecopyonread_categoricalqnetwork_categoricalqnetwork_dense_2_bias^Read_6/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0k
Identity_12IdentityRead_6/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_13IdentityIdentity_12:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*�
value�B�B%train_step/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/0/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/1/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/2/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/3/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/4/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/5/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH}
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*#
valueBB B B B B B B B �
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0Identity_1:output:0Identity_3:output:0Identity_5:output:0Identity_7:output:0Identity_9:output:0Identity_11:output:0Identity_13:output:0savev2_const_1"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtypes

2	�
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:�
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 i
Identity_14Identityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: U
Identity_15IdentityIdentity_14:output:0^NoOp*
T0*
_output_shapes
: �
NoOpNoOp^MergeV2Checkpoints^Read/DisableCopyOnRead^Read/ReadVariableOp^Read_1/DisableCopyOnRead^Read_1/ReadVariableOp^Read_2/DisableCopyOnRead^Read_2/ReadVariableOp^Read_3/DisableCopyOnRead^Read_3/ReadVariableOp^Read_4/DisableCopyOnRead^Read_4/ReadVariableOp^Read_5/DisableCopyOnRead^Read_5/ReadVariableOp^Read_6/DisableCopyOnRead^Read_6/ReadVariableOp*
_output_shapes
 "#
identity_15Identity_15:output:0*(
_construction_contextkEagerRuntime*%
_input_shapes
: : : : : : : : : 2(
MergeV2CheckpointsMergeV2Checkpoints20
Read/DisableCopyOnReadRead/DisableCopyOnRead2*
Read/ReadVariableOpRead/ReadVariableOp24
Read_1/DisableCopyOnReadRead_1/DisableCopyOnRead2.
Read_1/ReadVariableOpRead_1/ReadVariableOp24
Read_2/DisableCopyOnReadRead_2/DisableCopyOnRead2.
Read_2/ReadVariableOpRead_2/ReadVariableOp24
Read_3/DisableCopyOnReadRead_3/DisableCopyOnRead2.
Read_3/ReadVariableOpRead_3/ReadVariableOp24
Read_4/DisableCopyOnReadRead_4/DisableCopyOnRead2.
Read_4/ReadVariableOpRead_4/ReadVariableOp24
Read_5/DisableCopyOnReadRead_5/DisableCopyOnRead2.
Read_5/ReadVariableOpRead_5/ReadVariableOp24
Read_6/DisableCopyOnReadRead_6/DisableCopyOnRead2.
Read_6/ReadVariableOpRead_6/ReadVariableOp:?;

_output_shapes
: 
!
_user_specified_name	Const_1:TP
N
_user_specified_name64CategoricalQNetwork/CategoricalQNetwork/dense_2/bias:VR
P
_user_specified_name86CategoricalQNetwork/CategoricalQNetwork/dense_2/kernel:d`
^
_user_specified_nameFDCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense_1/bias:fb
`
_user_specified_nameHFCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense_1/kernel:b^
\
_user_specified_nameDBCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense/bias:d`
^
_user_specified_nameFDCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense/kernel:($
"
_user_specified_name
Variable:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
�
7
%__inference_get_initial_state_2415914

batch_size*(
_construction_contextkEagerRuntime*
_input_shapes
: :B >

_output_shapes
: 
$
_user_specified_name
batch_size
�g
�

)__inference_polymorphic_action_fn_2415868
time_step_step_type
time_step_reward
time_step_discount
time_step_observationo
\categoricalqnetwork_categoricalqnetwork_encodingnetwork_dense_matmul_readvariableop_resource:	�l
]categoricalqnetwork_categoricalqnetwork_encodingnetwork_dense_biasadd_readvariableop_resource:	�r
^categoricalqnetwork_categoricalqnetwork_encodingnetwork_dense_1_matmul_readvariableop_resource:
��n
_categoricalqnetwork_categoricalqnetwork_encodingnetwork_dense_1_biasadd_readvariableop_resource:	�b
Ncategoricalqnetwork_categoricalqnetwork_dense_2_matmul_readvariableop_resource:
��^
Ocategoricalqnetwork_categoricalqnetwork_dense_2_biasadd_readvariableop_resource:	�	
mul_x
identity	��TCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOp�SCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense/MatMul/ReadVariableOp�VCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense_1/BiasAdd/ReadVariableOp�UCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense_1/MatMul/ReadVariableOp�FCategoricalQNetwork/CategoricalQNetwork/dense_2/BiasAdd/ReadVariableOp�ECategoricalQNetwork/CategoricalQNetwork/dense_2/MatMul/ReadVariableOp�
ECategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"����   �
GCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/flatten/ReshapeReshapetime_step_observationNCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/flatten/Const:output:0*
T0*'
_output_shapes
:����������
BCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense/CastCastPCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/flatten/Reshape:output:0*

DstT0*

SrcT0*'
_output_shapes
:����������
SCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense/MatMul/ReadVariableOpReadVariableOp\categoricalqnetwork_categoricalqnetwork_encodingnetwork_dense_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype0�
DCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense/MatMulMatMulFCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense/Cast:y:0[CategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
TCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOpReadVariableOp]categoricalqnetwork_categoricalqnetwork_encodingnetwork_dense_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
ECategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense/BiasAddBiasAddNCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense/MatMul:product:0\CategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
BCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense/ReluReluNCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense/BiasAdd:output:0*
T0*(
_output_shapes
:�����������
UCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense_1/MatMul/ReadVariableOpReadVariableOp^categoricalqnetwork_categoricalqnetwork_encodingnetwork_dense_1_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
FCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense_1/MatMulMatMulPCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense/Relu:activations:0]CategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
VCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense_1/BiasAdd/ReadVariableOpReadVariableOp_categoricalqnetwork_categoricalqnetwork_encodingnetwork_dense_1_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
GCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense_1/BiasAddBiasAddPCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense_1/MatMul:product:0^CategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
DCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense_1/ReluReluPCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense_1/BiasAdd:output:0*
T0*(
_output_shapes
:�����������
ECategoricalQNetwork/CategoricalQNetwork/dense_2/MatMul/ReadVariableOpReadVariableOpNcategoricalqnetwork_categoricalqnetwork_dense_2_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
6CategoricalQNetwork/CategoricalQNetwork/dense_2/MatMulMatMulRCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense_1/Relu:activations:0MCategoricalQNetwork/CategoricalQNetwork/dense_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
FCategoricalQNetwork/CategoricalQNetwork/dense_2/BiasAdd/ReadVariableOpReadVariableOpOcategoricalqnetwork_categoricalqnetwork_dense_2_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
7CategoricalQNetwork/CategoricalQNetwork/dense_2/BiasAddBiasAdd@CategoricalQNetwork/CategoricalQNetwork/dense_2/MatMul:product:0NCategoricalQNetwork/CategoricalQNetwork/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������v
!CategoricalQNetwork/Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"����   Q   �
CategoricalQNetwork/ReshapeReshape@CategoricalQNetwork/CategoricalQNetwork/dense_2/BiasAdd:output:0*CategoricalQNetwork/Reshape/shape:output:0*
T0*+
_output_shapes
:���������Qn
SoftmaxSoftmax$CategoricalQNetwork/Reshape:output:0*
T0*+
_output_shapes
:���������QZ
mulMulmul_xSoftmax:softmax:0*
T0*+
_output_shapes
:���������Q`
Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
���������e
SumSummul:z:0Sum/reduction_indices:output:0*
T0*'
_output_shapes
:���������l
!Categorical/mode/ArgMax/dimensionConst*
_output_shapes
: *
dtype0*
valueB :
����������
Categorical/mode/ArgMaxArgMaxSum:output:0*Categorical/mode/ArgMax/dimension:output:0*
T0*#
_output_shapes
:���������T
Deterministic/atolConst*
_output_shapes
: *
dtype0	*
value	B	 R T
Deterministic/rtolConst*
_output_shapes
: *
dtype0	*
value	B	 R d
!Deterministic/sample/sample_shapeConst*
_output_shapes
: *
dtype0*
valueB x
Deterministic/sample/ShapeShape Categorical/mode/ArgMax:output:0*
T0	*
_output_shapes
::��\
Deterministic/sample/ConstConst*
_output_shapes
: *
dtype0*
value	B : r
(Deterministic/sample/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: t
*Deterministic/sample/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:t
*Deterministic/sample/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
"Deterministic/sample/strided_sliceStridedSlice#Deterministic/sample/Shape:output:01Deterministic/sample/strided_slice/stack:output:03Deterministic/sample/strided_slice/stack_1:output:03Deterministic/sample/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_maskh
%Deterministic/sample/BroadcastArgs/s0Const*
_output_shapes
: *
dtype0*
valueB j
'Deterministic/sample/BroadcastArgs/s0_1Const*
_output_shapes
: *
dtype0*
valueB �
"Deterministic/sample/BroadcastArgsBroadcastArgs0Deterministic/sample/BroadcastArgs/s0_1:output:0+Deterministic/sample/strided_slice:output:0*
_output_shapes
:n
$Deterministic/sample/concat/values_0Const*
_output_shapes
:*
dtype0*
valueB:g
$Deterministic/sample/concat/values_2Const*
_output_shapes
: *
dtype0*
valueB b
 Deterministic/sample/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Deterministic/sample/concatConcatV2-Deterministic/sample/concat/values_0:output:0'Deterministic/sample/BroadcastArgs:r0:0-Deterministic/sample/concat/values_2:output:0)Deterministic/sample/concat/axis:output:0*
N*
T0*
_output_shapes
:�
 Deterministic/sample/BroadcastToBroadcastTo Categorical/mode/ArgMax:output:0$Deterministic/sample/concat:output:0*
T0	*'
_output_shapes
:���������e
"Deterministic/sample/Shape_1/ConstConst*
_output_shapes
: *
dtype0*
valueB f
Deterministic/sample/Shape_1Const*
_output_shapes
:*
dtype0*
valueB: �
Deterministic/sample/Shape_2Shape)Deterministic/sample/BroadcastTo:output:0*
T0	*
_output_shapes
::��t
*Deterministic/sample/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:v
,Deterministic/sample/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: v
,Deterministic/sample/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
$Deterministic/sample/strided_slice_1StridedSlice%Deterministic/sample/Shape_2:output:03Deterministic/sample/strided_slice_1/stack:output:05Deterministic/sample/strided_slice_1/stack_1:output:05Deterministic/sample/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_maskd
"Deterministic/sample/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Deterministic/sample/concat_1ConcatV2*Deterministic/sample/sample_shape:output:0-Deterministic/sample/strided_slice_1:output:0+Deterministic/sample/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
Deterministic/sample/ReshapeReshape)Deterministic/sample/BroadcastTo:output:0&Deterministic/sample/concat_1:output:0*
T0	*#
_output_shapes
:���������Y
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0	*
value	B	 R�
clip_by_value/MinimumMinimum%Deterministic/sample/Reshape:output:0 clip_by_value/Minimum/y:output:0*
T0	*#
_output_shapes
:���������Q
clip_by_value/yConst*
_output_shapes
: *
dtype0	*
value	B	 R {
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0	*#
_output_shapes
:���������\
IdentityIdentityclip_by_value:z:0^NoOp*
T0	*#
_output_shapes
:����������
NoOpNoOpU^CategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOpT^CategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense/MatMul/ReadVariableOpW^CategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense_1/BiasAdd/ReadVariableOpV^CategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense_1/MatMul/ReadVariableOpG^CategoricalQNetwork/CategoricalQNetwork/dense_2/BiasAdd/ReadVariableOpF^CategoricalQNetwork/CategoricalQNetwork/dense_2/MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*e
_input_shapesT
R:���������:���������:���������:���������: : : : : : :Q2�
TCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOpTCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOp2�
SCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense/MatMul/ReadVariableOpSCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense/MatMul/ReadVariableOp2�
VCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense_1/BiasAdd/ReadVariableOpVCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense_1/BiasAdd/ReadVariableOp2�
UCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense_1/MatMul/ReadVariableOpUCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense_1/MatMul/ReadVariableOp2�
FCategoricalQNetwork/CategoricalQNetwork/dense_2/BiasAdd/ReadVariableOpFCategoricalQNetwork/CategoricalQNetwork/dense_2/BiasAdd/ReadVariableOp2�
ECategoricalQNetwork/CategoricalQNetwork/dense_2/MatMul/ReadVariableOpECategoricalQNetwork/CategoricalQNetwork/dense_2/MatMul/ReadVariableOp: 


_output_shapes
:Q:(	$
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:^Z
'
_output_shapes
:���������
/
_user_specified_nametime_step_observation:WS
#
_output_shapes
:���������
,
_user_specified_nametime_step_discount:UQ
#
_output_shapes
:���������
*
_user_specified_nametime_step_reward:X T
#
_output_shapes
:���������
-
_user_specified_nametime_step_step_type
]

__inference_<lambda>_2415472*(
_construction_contextkEagerRuntime*
_input_shapes 
�
c
__inference_<lambda>_2415470!
readvariableop_resource:	 
identity	��ReadVariableOp^
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0	T
IdentityIdentityReadVariableOp:value:0^NoOp*
T0	*
_output_shapes
: 3
NoOpNoOp^ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2 
ReadVariableOpReadVariableOp:( $
"
_user_specified_name
resource
�
7
%__inference_get_initial_state_2415705

batch_size*(
_construction_contextkEagerRuntime*
_input_shapes
: :B >

_output_shapes
: 
$
_user_specified_name
batch_size
�
k
+__inference_function_with_signature_2415716
unknown:	 
identity	��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallunknown*
Tin
2*
Tout
2	*
_collective_manager_ids
 *
_output_shapes
: *#
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *%
f R
__inference_<lambda>_2415470^
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0	*
_output_shapes
: <
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 22
StatefulPartitionedCallStatefulPartitionedCall:' #
!
_user_specified_name	2415712
�
�
+__inference_function_with_signature_2415677
	step_type

reward
discount
observation
unknown:	�
	unknown_0:	�
	unknown_1:
��
	unknown_2:	�
	unknown_3:
��
	unknown_4:	�
	unknown_5
identity	��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCall	step_typerewarddiscountobservationunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5*
Tin
2*
Tout
2	*
_collective_manager_ids
 *#
_output_shapes
:���������*(
_read_only_resource_inputs

	*-
config_proto

CPU

GPU 2J 8� *2
f-R+
)__inference_polymorphic_action_fn_2415660k
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0	*#
_output_shapes
:���������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*e
_input_shapesT
R:���������:���������:���������:���������: : : : : : :Q22
StatefulPartitionedCallStatefulPartitionedCall: 


_output_shapes
:Q:'	#
!
_user_specified_name	2415671:'#
!
_user_specified_name	2415669:'#
!
_user_specified_name	2415667:'#
!
_user_specified_name	2415665:'#
!
_user_specified_name	2415663:'#
!
_user_specified_name	2415661:VR
'
_output_shapes
:���������
'
_user_specified_name0/observation:OK
#
_output_shapes
:���������
$
_user_specified_name
0/discount:MI
#
_output_shapes
:���������
"
_user_specified_name
0/reward:P L
#
_output_shapes
:���������
%
_user_specified_name0/step_type
�
=
+__inference_function_with_signature_2415706

batch_size�
PartitionedCallPartitionedCall
batch_size*
Tin
2*

Tout
 *
_collective_manager_ids
 *
_output_shapes
 * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *.
f)R'
%__inference_get_initial_state_2415705*(
_construction_contextkEagerRuntime*
_input_shapes
: :B >

_output_shapes
: 
$
_user_specified_name
batch_size"�L
saver_filename:0StatefulPartitionedCall_2:0StatefulPartitionedCall_38"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*�
action�
4

0/discount&
action_0_discount:0���������
>
0/observation-
action_0_observation:0���������
0
0/reward$
action_0_reward:0���������
6
0/step_type'
action_0_step_type:0���������6
action,
StatefulPartitionedCall:0	���������tensorflow/serving/predict*e
get_initial_stateP
2

batch_size$
get_initial_state_batch_size:0 tensorflow/serving/predict*,
get_metadatatensorflow/serving/predict*Z
get_train_stepH*
int64!
StatefulPartitionedCall_1:0	 tensorflow/serving/predict:�u
�

train_step
metadata
model_variables
_all_assets

action
distribution
get_initial_state
get_metadata
	get_train_step


signatures"
_generic_user_object
:	 (2Variable
 "
trackable_dict_wrapper
K
0
1
2
3
4
5"
trackable_tuple_wrapper
5
_wrapped_policy"
trackable_dict_wrapper
�
trace_0
trace_12�
)__inference_polymorphic_action_fn_2415798
)__inference_polymorphic_action_fn_2415868�
���
FullArgSpec(
args �
j	time_step
jpolicy_state
varargs
 
varkw
 
defaults�
� 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 ztrace_0ztrace_1
�
trace_02�
/__inference_polymorphic_distribution_fn_2415911�
���
FullArgSpec(
args �
j	time_step
jpolicy_state
varargs
 
varkw
 
defaults�
� 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 ztrace_0
�
trace_02�
%__inference_get_initial_state_2415914�
���
FullArgSpec
args�
j
batch_size
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 ztrace_0
�B�
__inference_<lambda>_2415472"�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� 
�B�
__inference_<lambda>_2415470"�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� 
`

action
get_initial_state
get_train_step
get_metadata"
signature_map
W:U	�2DCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense/kernel
Q:O�2BCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense/bias
Z:X
��2FCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense_1/kernel
S:Q�2DCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense_1/bias
J:H
��26CategoricalQNetwork/CategoricalQNetwork/dense_2/kernel
C:A�24CategoricalQNetwork/CategoricalQNetwork/dense_2/bias
.

_q_network"
_generic_user_object
�
	capture_6B�
)__inference_polymorphic_action_fn_2415798	step_typerewarddiscountobservation"�
���
FullArgSpec(
args �
j	time_step
jpolicy_state
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z	capture_6
�
	capture_6B�
)__inference_polymorphic_action_fn_2415868time_step_step_typetime_step_rewardtime_step_discounttime_step_observation"�
���
FullArgSpec(
args �
j	time_step
jpolicy_state
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z	capture_6
�
	capture_6B�
/__inference_polymorphic_distribution_fn_2415911	step_typerewarddiscountobservation"�
���
FullArgSpec(
args �
j	time_step
jpolicy_state
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z	capture_6
�B�
%__inference_get_initial_state_2415914
batch_size"�
���
FullArgSpec
args�
j
batch_size
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�
	capture_6B�
=__inference_signature_wrapper_function_with_signature_2415700
0/discount0/observation0/reward0/step_type"�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 [

kwonlyargsM�J
jarg_0_discount
jarg_0_observation
jarg_0_reward
jarg_0_step_type
kwonlydefaults
 
annotations� *
 z	capture_6
�B�
=__inference_signature_wrapper_function_with_signature_2415710
batch_size"�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs�
j
batch_size
kwonlydefaults
 
annotations� *
 
�B�
=__inference_signature_wrapper_function_with_signature_2415723"�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� 
�B�
=__inference_signature_wrapper_function_with_signature_2415728"�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� 
�
	variables
trainable_variables
regularization_losses
	keras_api
 __call__
*!&call_and_return_all_conditional_losses
"
_q_network"
_tf_keras_layer
J
Constjtf.TrackableConstant
J
0
1
2
3
4
5"
trackable_list_wrapper
J
0
1
2
3
4
5"
trackable_list_wrapper
 "
trackable_list_wrapper
�
#non_trainable_variables

$layers
%metrics
&layer_regularization_losses
'layer_metrics
	variables
trainable_variables
regularization_losses
 __call__
*!&call_and_return_all_conditional_losses
&!"call_and_return_conditional_losses"
_generic_user_object
�2��
���
FullArgSpecD
args<�9
jobservation
j	step_type
jnetwork_state

jtraining
varargs
 
varkw
 
defaults�

 
� 
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpecD
args<�9
jobservation
j	step_type
jnetwork_state

jtraining
varargs
 
varkw
 
defaults�

 
� 
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�
(	variables
)trainable_variables
*regularization_losses
+	keras_api
,__call__
*-&call_and_return_all_conditional_losses
._encoder
/_q_value_layer"
_tf_keras_layer
 "
trackable_list_wrapper
'
"0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
J
0
1
2
3
4
5"
trackable_list_wrapper
J
0
1
2
3
4
5"
trackable_list_wrapper
 "
trackable_list_wrapper
�
0non_trainable_variables

1layers
2metrics
3layer_regularization_losses
4layer_metrics
(	variables
)trainable_variables
*regularization_losses
,__call__
*-&call_and_return_all_conditional_losses
&-"call_and_return_conditional_losses"
_generic_user_object
�2��
���
FullArgSpecD
args<�9
jobservation
j	step_type
jnetwork_state

jtraining
varargs
 
varkw
 
defaults�

 
� 
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpecD
args<�9
jobservation
j	step_type
jnetwork_state

jtraining
varargs
 
varkw
 
defaults�

 
� 
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�
5	variables
6trainable_variables
7regularization_losses
8	keras_api
9__call__
*:&call_and_return_all_conditional_losses
;_postprocessing_layers"
_tf_keras_layer
�
<	variables
=trainable_variables
>regularization_losses
?	keras_api
@__call__
*A&call_and_return_all_conditional_losses

kernel
bias"
_tf_keras_layer
 "
trackable_list_wrapper
.
.0
/1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
<
0
1
2
3"
trackable_list_wrapper
<
0
1
2
3"
trackable_list_wrapper
 "
trackable_list_wrapper
�
Bnon_trainable_variables

Clayers
Dmetrics
Elayer_regularization_losses
Flayer_metrics
5	variables
6trainable_variables
7regularization_losses
9__call__
*:&call_and_return_all_conditional_losses
&:"call_and_return_conditional_losses"
_generic_user_object
�2��
���
FullArgSpecD
args<�9
jobservation
j	step_type
jnetwork_state

jtraining
varargs
 
varkw
 
defaults�

 
� 
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpecD
args<�9
jobservation
j	step_type
jnetwork_state

jtraining
varargs
 
varkw
 
defaults�

 
� 
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
5
G0
H1
I2"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
Jnon_trainable_variables

Klayers
Lmetrics
Mlayer_regularization_losses
Nlayer_metrics
<	variables
=trainable_variables
>regularization_losses
@__call__
*A&call_and_return_all_conditional_losses
&A"call_and_return_conditional_losses"
_generic_user_object
�2��
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
5
G0
H1
I2"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�
O	variables
Ptrainable_variables
Qregularization_losses
R	keras_api
S__call__
*T&call_and_return_all_conditional_losses"
_tf_keras_layer
�
U	variables
Vtrainable_variables
Wregularization_losses
X	keras_api
Y__call__
*Z&call_and_return_all_conditional_losses

kernel
bias"
_tf_keras_layer
�
[	variables
\trainable_variables
]regularization_losses
^	keras_api
___call__
*`&call_and_return_all_conditional_losses

kernel
bias"
_tf_keras_layer
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
anon_trainable_variables

blayers
cmetrics
dlayer_regularization_losses
elayer_metrics
O	variables
Ptrainable_variables
Qregularization_losses
S__call__
*T&call_and_return_all_conditional_losses
&T"call_and_return_conditional_losses"
_generic_user_object
�2��
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
fnon_trainable_variables

glayers
hmetrics
ilayer_regularization_losses
jlayer_metrics
U	variables
Vtrainable_variables
Wregularization_losses
Y__call__
*Z&call_and_return_all_conditional_losses
&Z"call_and_return_conditional_losses"
_generic_user_object
�2��
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
knon_trainable_variables

llayers
mmetrics
nlayer_regularization_losses
olayer_metrics
[	variables
\trainable_variables
]regularization_losses
___call__
*`&call_and_return_all_conditional_losses
&`"call_and_return_conditional_losses"
_generic_user_object
�2��
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapperD
__inference_<lambda>_2415470$�

� 
� "�
unknown 	4
__inference_<lambda>_2415472�

� 
� "� R
%__inference_get_initial_state_2415914)"�
�
�

batch_size 
� "� �
)__inference_polymorphic_action_fn_2415798����
���
���
TimeStep,
	step_type�
	step_type���������&
reward�
reward���������*
discount�
discount���������4
observation%�"
observation���������
� 
� "R�O

PolicyStep&
action�
action���������	
state� 
info� �
)__inference_polymorphic_action_fn_2415868����
���
���
TimeStep6
	step_type)�&
time_step_step_type���������0
reward&�#
time_step_reward���������4
discount(�%
time_step_discount���������>
observation/�,
time_step_observation���������
� 
� "R�O

PolicyStep&
action�
action���������	
state� 
info� �
/__inference_polymorphic_distribution_fn_2415911����
���
���
TimeStep,
	step_type�
	step_type���������&
reward�
reward���������*
discount�
discount���������4
observation%�"
observation���������
� 
� "���

PolicyStep�
action������
`
B�?

atol� 	

loc����������	

rtol� 	
L�I

allow_nan_statsp

namejDeterministic_1_1

validate_argsp 
�
j
parameters
� 
�
jname+tfp.distributions.Deterministic_ACTTypeSpec 
state� 
info� �
=__inference_signature_wrapper_function_with_signature_2415700����
� 
���
2
arg_0_discount �

0/discount���������
<
arg_0_observation'�$
0/observation���������
.
arg_0_reward�
0/reward���������
4
arg_0_step_type!�
0/step_type���������"+�(
&
action�
action���������	x
=__inference_signature_wrapper_function_with_signature_241571070�-
� 
&�#
!

batch_size�

batch_size "� q
=__inference_signature_wrapper_function_with_signature_24157230�

� 
� "�

int64�
int64 	U
=__inference_signature_wrapper_function_with_signature_2415728�

� 
� "� 