§®
Ôª
D
AddV2
x"T
y"T
z"T"
Ttype:
2	
B
AssignVariableOp
resource
value"dtype"
dtypetype
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(
?
Mul
x"T
y"T
z"T"
Ttype:
2	

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype
E
Relu
features"T
activations"T"
Ttype:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
.
Rsqrt
x"T
y"T"
Ttype:

2
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
0
Sigmoid
x"T
y"T"
Ttype:

2
¾
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
executor_typestring 
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
<
Sub
x"T
y"T
z"T"
Ttype:
2	

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 "serve*2.6.22v2.6.1-9-gc2363d6d0258ë

batch_normalization/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:#**
shared_namebatch_normalization/gamma

-batch_normalization/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization/gamma*
_output_shapes
:#*
dtype0

batch_normalization/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:#*)
shared_namebatch_normalization/beta

,batch_normalization/beta/Read/ReadVariableOpReadVariableOpbatch_normalization/beta*
_output_shapes
:#*
dtype0

batch_normalization/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:#*0
shared_name!batch_normalization/moving_mean

3batch_normalization/moving_mean/Read/ReadVariableOpReadVariableOpbatch_normalization/moving_mean*
_output_shapes
:#*
dtype0

#batch_normalization/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:#*4
shared_name%#batch_normalization/moving_variance

7batch_normalization/moving_variance/Read/ReadVariableOpReadVariableOp#batch_normalization/moving_variance*
_output_shapes
:#*
dtype0
u
dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	#ú*
shared_namedense/kernel
n
 dense/kernel/Read/ReadVariableOpReadVariableOpdense/kernel*
_output_shapes
:	#ú*
dtype0
m

dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:ú*
shared_name
dense/bias
f
dense/bias/Read/ReadVariableOpReadVariableOp
dense/bias*
_output_shapes	
:ú*
dtype0

batch_normalization_1/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:ú*,
shared_namebatch_normalization_1/gamma

/batch_normalization_1/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_1/gamma*
_output_shapes	
:ú*
dtype0

batch_normalization_1/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:ú*+
shared_namebatch_normalization_1/beta

.batch_normalization_1/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_1/beta*
_output_shapes	
:ú*
dtype0

!batch_normalization_1/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:ú*2
shared_name#!batch_normalization_1/moving_mean

5batch_normalization_1/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_1/moving_mean*
_output_shapes	
:ú*
dtype0
£
%batch_normalization_1/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:ú*6
shared_name'%batch_normalization_1/moving_variance

9batch_normalization_1/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_1/moving_variance*
_output_shapes	
:ú*
dtype0
z
dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
úú*
shared_namedense_1/kernel
s
"dense_1/kernel/Read/ReadVariableOpReadVariableOpdense_1/kernel* 
_output_shapes
:
úú*
dtype0
q
dense_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:ú*
shared_namedense_1/bias
j
 dense_1/bias/Read/ReadVariableOpReadVariableOpdense_1/bias*
_output_shapes	
:ú*
dtype0

batch_normalization_2/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:ú*,
shared_namebatch_normalization_2/gamma

/batch_normalization_2/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_2/gamma*
_output_shapes	
:ú*
dtype0

batch_normalization_2/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:ú*+
shared_namebatch_normalization_2/beta

.batch_normalization_2/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_2/beta*
_output_shapes	
:ú*
dtype0

!batch_normalization_2/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:ú*2
shared_name#!batch_normalization_2/moving_mean

5batch_normalization_2/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_2/moving_mean*
_output_shapes	
:ú*
dtype0
£
%batch_normalization_2/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:ú*6
shared_name'%batch_normalization_2/moving_variance

9batch_normalization_2/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_2/moving_variance*
_output_shapes	
:ú*
dtype0
y
dense_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	ú*
shared_namedense_2/kernel
r
"dense_2/kernel/Read/ReadVariableOpReadVariableOpdense_2/kernel*
_output_shapes
:	ú*
dtype0
p
dense_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_2/bias
i
 dense_2/bias/Read/ReadVariableOpReadVariableOpdense_2/bias*
_output_shapes
:*
dtype0
`
beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namebeta_1
Y
beta_1/Read/ReadVariableOpReadVariableOpbeta_1*
_output_shapes
: *
dtype0
`
beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namebeta_2
Y
beta_2/Read/ReadVariableOpReadVariableOpbeta_2*
_output_shapes
: *
dtype0
^
decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namedecay
W
decay/Read/ReadVariableOpReadVariableOpdecay*
_output_shapes
: *
dtype0
n
learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namelearning_rate
g
!learning_rate/Read/ReadVariableOpReadVariableOplearning_rate*
_output_shapes
: *
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
b
total_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_2
[
total_2/Read/ReadVariableOpReadVariableOptotal_2*
_output_shapes
: *
dtype0
b
count_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_2
[
count_2/Read/ReadVariableOpReadVariableOpcount_2*
_output_shapes
: *
dtype0
b
total_3VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_3
[
total_3/Read/ReadVariableOpReadVariableOptotal_3*
_output_shapes
: *
dtype0
b
count_3VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_3
[
count_3/Read/ReadVariableOpReadVariableOpcount_3*
_output_shapes
: *
dtype0

 Adam/batch_normalization/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:#*1
shared_name" Adam/batch_normalization/gamma/m

4Adam/batch_normalization/gamma/m/Read/ReadVariableOpReadVariableOp Adam/batch_normalization/gamma/m*
_output_shapes
:#*
dtype0

Adam/batch_normalization/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:#*0
shared_name!Adam/batch_normalization/beta/m

3Adam/batch_normalization/beta/m/Read/ReadVariableOpReadVariableOpAdam/batch_normalization/beta/m*
_output_shapes
:#*
dtype0

Adam/dense/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	#ú*$
shared_nameAdam/dense/kernel/m
|
'Adam/dense/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense/kernel/m*
_output_shapes
:	#ú*
dtype0
{
Adam/dense/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:ú*"
shared_nameAdam/dense/bias/m
t
%Adam/dense/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense/bias/m*
_output_shapes	
:ú*
dtype0

"Adam/batch_normalization_1/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:ú*3
shared_name$"Adam/batch_normalization_1/gamma/m

6Adam/batch_normalization_1/gamma/m/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_1/gamma/m*
_output_shapes	
:ú*
dtype0

!Adam/batch_normalization_1/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:ú*2
shared_name#!Adam/batch_normalization_1/beta/m

5Adam/batch_normalization_1/beta/m/Read/ReadVariableOpReadVariableOp!Adam/batch_normalization_1/beta/m*
_output_shapes	
:ú*
dtype0

Adam/dense_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
úú*&
shared_nameAdam/dense_1/kernel/m

)Adam/dense_1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_1/kernel/m* 
_output_shapes
:
úú*
dtype0

Adam/dense_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:ú*$
shared_nameAdam/dense_1/bias/m
x
'Adam/dense_1/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_1/bias/m*
_output_shapes	
:ú*
dtype0

"Adam/batch_normalization_2/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:ú*3
shared_name$"Adam/batch_normalization_2/gamma/m

6Adam/batch_normalization_2/gamma/m/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_2/gamma/m*
_output_shapes	
:ú*
dtype0

!Adam/batch_normalization_2/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:ú*2
shared_name#!Adam/batch_normalization_2/beta/m

5Adam/batch_normalization_2/beta/m/Read/ReadVariableOpReadVariableOp!Adam/batch_normalization_2/beta/m*
_output_shapes	
:ú*
dtype0

Adam/dense_2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	ú*&
shared_nameAdam/dense_2/kernel/m

)Adam/dense_2/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_2/kernel/m*
_output_shapes
:	ú*
dtype0
~
Adam/dense_2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/dense_2/bias/m
w
'Adam/dense_2/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_2/bias/m*
_output_shapes
:*
dtype0

 Adam/batch_normalization/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:#*1
shared_name" Adam/batch_normalization/gamma/v

4Adam/batch_normalization/gamma/v/Read/ReadVariableOpReadVariableOp Adam/batch_normalization/gamma/v*
_output_shapes
:#*
dtype0

Adam/batch_normalization/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:#*0
shared_name!Adam/batch_normalization/beta/v

3Adam/batch_normalization/beta/v/Read/ReadVariableOpReadVariableOpAdam/batch_normalization/beta/v*
_output_shapes
:#*
dtype0

Adam/dense/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	#ú*$
shared_nameAdam/dense/kernel/v
|
'Adam/dense/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense/kernel/v*
_output_shapes
:	#ú*
dtype0
{
Adam/dense/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:ú*"
shared_nameAdam/dense/bias/v
t
%Adam/dense/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense/bias/v*
_output_shapes	
:ú*
dtype0

"Adam/batch_normalization_1/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:ú*3
shared_name$"Adam/batch_normalization_1/gamma/v

6Adam/batch_normalization_1/gamma/v/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_1/gamma/v*
_output_shapes	
:ú*
dtype0

!Adam/batch_normalization_1/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:ú*2
shared_name#!Adam/batch_normalization_1/beta/v

5Adam/batch_normalization_1/beta/v/Read/ReadVariableOpReadVariableOp!Adam/batch_normalization_1/beta/v*
_output_shapes	
:ú*
dtype0

Adam/dense_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
úú*&
shared_nameAdam/dense_1/kernel/v

)Adam/dense_1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_1/kernel/v* 
_output_shapes
:
úú*
dtype0

Adam/dense_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:ú*$
shared_nameAdam/dense_1/bias/v
x
'Adam/dense_1/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_1/bias/v*
_output_shapes	
:ú*
dtype0

"Adam/batch_normalization_2/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:ú*3
shared_name$"Adam/batch_normalization_2/gamma/v

6Adam/batch_normalization_2/gamma/v/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_2/gamma/v*
_output_shapes	
:ú*
dtype0

!Adam/batch_normalization_2/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:ú*2
shared_name#!Adam/batch_normalization_2/beta/v

5Adam/batch_normalization_2/beta/v/Read/ReadVariableOpReadVariableOp!Adam/batch_normalization_2/beta/v*
_output_shapes	
:ú*
dtype0

Adam/dense_2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	ú*&
shared_nameAdam/dense_2/kernel/v

)Adam/dense_2/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_2/kernel/v*
_output_shapes
:	ú*
dtype0
~
Adam/dense_2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/dense_2/bias/v
w
'Adam/dense_2/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_2/bias/v*
_output_shapes
:*
dtype0

NoOpNoOp
ÉR
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*R
valueúQB÷Q BðQ

layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer_with_weights-2
layer-4
layer_with_weights-3
layer-5
layer-6
layer_with_weights-4
layer-7
	layer_with_weights-5
	layer-8

	optimizer
trainable_variables
regularization_losses
	variables
	keras_api

signatures
 

axis
	gamma
beta
moving_mean
moving_variance
trainable_variables
regularization_losses
	variables
	keras_api
h

kernel
bias
trainable_variables
regularization_losses
	variables
	keras_api
R
trainable_variables
 regularization_losses
!	variables
"	keras_api

#axis
	$gamma
%beta
&moving_mean
'moving_variance
(trainable_variables
)regularization_losses
*	variables
+	keras_api
h

,kernel
-bias
.trainable_variables
/regularization_losses
0	variables
1	keras_api
R
2trainable_variables
3regularization_losses
4	variables
5	keras_api

6axis
	7gamma
8beta
9moving_mean
:moving_variance
;trainable_variables
<regularization_losses
=	variables
>	keras_api
h

?kernel
@bias
Atrainable_variables
Bregularization_losses
C	variables
D	keras_api
°

Ebeta_1

Fbeta_2
	Gdecay
Hlearning_rate
Iitermmmm$m%m,m-m7m8m?m@mvvvv$v%v,v -v¡7v¢8v£?v¤@v¥
V
0
1
2
3
$4
%5
,6
-7
78
89
?10
@11
 

0
1
2
3
4
5
$6
%7
&8
'9
,10
-11
712
813
914
:15
?16
@17
­

Jlayers
Klayer_regularization_losses
Lnon_trainable_variables
Mmetrics
trainable_variables
Nlayer_metrics
regularization_losses
	variables
 
 
db
VARIABLE_VALUEbatch_normalization/gamma5layer_with_weights-0/gamma/.ATTRIBUTES/VARIABLE_VALUE
b`
VARIABLE_VALUEbatch_normalization/beta4layer_with_weights-0/beta/.ATTRIBUTES/VARIABLE_VALUE
pn
VARIABLE_VALUEbatch_normalization/moving_mean;layer_with_weights-0/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUE#batch_normalization/moving_variance?layer_with_weights-0/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

0
1
 

0
1
2
3
­

Olayers
Player_regularization_losses
Qnon_trainable_variables
Rmetrics
trainable_variables
Slayer_metrics
regularization_losses
	variables
XV
VARIABLE_VALUEdense/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUE
dense/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1
 

0
1
­

Tlayers
Ulayer_regularization_losses
Vnon_trainable_variables
Wmetrics
trainable_variables
Xlayer_metrics
regularization_losses
	variables
 
 
 
­

Ylayers
Zlayer_regularization_losses
[non_trainable_variables
\metrics
trainable_variables
]layer_metrics
 regularization_losses
!	variables
 
fd
VARIABLE_VALUEbatch_normalization_1/gamma5layer_with_weights-2/gamma/.ATTRIBUTES/VARIABLE_VALUE
db
VARIABLE_VALUEbatch_normalization_1/beta4layer_with_weights-2/beta/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUE!batch_normalization_1/moving_mean;layer_with_weights-2/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE%batch_normalization_1/moving_variance?layer_with_weights-2/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

$0
%1
 

$0
%1
&2
'3
­

^layers
_layer_regularization_losses
`non_trainable_variables
ametrics
(trainable_variables
blayer_metrics
)regularization_losses
*	variables
ZX
VARIABLE_VALUEdense_1/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_1/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE

,0
-1
 

,0
-1
­

clayers
dlayer_regularization_losses
enon_trainable_variables
fmetrics
.trainable_variables
glayer_metrics
/regularization_losses
0	variables
 
 
 
­

hlayers
ilayer_regularization_losses
jnon_trainable_variables
kmetrics
2trainable_variables
llayer_metrics
3regularization_losses
4	variables
 
fd
VARIABLE_VALUEbatch_normalization_2/gamma5layer_with_weights-4/gamma/.ATTRIBUTES/VARIABLE_VALUE
db
VARIABLE_VALUEbatch_normalization_2/beta4layer_with_weights-4/beta/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUE!batch_normalization_2/moving_mean;layer_with_weights-4/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE%batch_normalization_2/moving_variance?layer_with_weights-4/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

70
81
 

70
81
92
:3
­

mlayers
nlayer_regularization_losses
onon_trainable_variables
pmetrics
;trainable_variables
qlayer_metrics
<regularization_losses
=	variables
ZX
VARIABLE_VALUEdense_2/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_2/bias4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE

?0
@1
 

?0
@1
­

rlayers
slayer_regularization_losses
tnon_trainable_variables
umetrics
Atrainable_variables
vlayer_metrics
Bregularization_losses
C	variables
GE
VARIABLE_VALUEbeta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
GE
VARIABLE_VALUEbeta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
EC
VARIABLE_VALUEdecay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUElearning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
HF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
?
0
1
2
3
4
5
6
7
	8
 
*
0
1
&2
'3
94
:5

w0
x1
y2
z3
 
 
 

0
1
 
 
 
 
 
 
 
 
 
 
 
 
 
 

&0
'1
 
 
 
 
 
 
 
 
 
 
 
 
 
 

90
:1
 
 
 
 
 
 
 
4
	{total
	|count
}	variables
~	keras_api
H
	total

count

_fn_kwargs
	variables
	keras_api
I

total

count

_fn_kwargs
	variables
	keras_api
I

total

count

_fn_kwargs
	variables
	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

{0
|1

}	variables
QO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

	variables
QO
VARIABLE_VALUEtotal_24keras_api/metrics/2/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_24keras_api/metrics/2/count/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

	variables
QO
VARIABLE_VALUEtotal_34keras_api/metrics/3/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_34keras_api/metrics/3/count/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

	variables

VARIABLE_VALUE Adam/batch_normalization/gamma/mQlayer_with_weights-0/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUEAdam/batch_normalization/beta/mPlayer_with_weights-0/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/dense/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/dense/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE"Adam/batch_normalization_1/gamma/mQlayer_with_weights-2/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE!Adam/batch_normalization_1/beta/mPlayer_with_weights-2/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/dense_1/kernel/mRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_1/bias/mPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE"Adam/batch_normalization_2/gamma/mQlayer_with_weights-4/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE!Adam/batch_normalization_2/beta/mPlayer_with_weights-4/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/dense_2/kernel/mRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_2/bias/mPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE Adam/batch_normalization/gamma/vQlayer_with_weights-0/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUEAdam/batch_normalization/beta/vPlayer_with_weights-0/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/dense/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/dense/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE"Adam/batch_normalization_1/gamma/vQlayer_with_weights-2/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE!Adam/batch_normalization_1/beta/vPlayer_with_weights-2/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/dense_1/kernel/vRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_1/bias/vPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE"Adam/batch_normalization_2/gamma/vQlayer_with_weights-4/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE!Adam/batch_normalization_2/beta/vPlayer_with_weights-4/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/dense_2/kernel/vRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_2/bias/vPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
z
serving_default_input_1Placeholder*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#*
dtype0*
shape:ÿÿÿÿÿÿÿÿÿ#
®
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1#batch_normalization/moving_variancebatch_normalization/gammabatch_normalization/moving_meanbatch_normalization/betadense/kernel
dense/bias%batch_normalization_1/moving_variancebatch_normalization_1/gamma!batch_normalization_1/moving_meanbatch_normalization_1/betadense_1/kerneldense_1/bias%batch_normalization_2/moving_variancebatch_normalization_2/gamma!batch_normalization_2/moving_meanbatch_normalization_2/betadense_2/kerneldense_2/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*4
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *.
f)R'
%__inference_signature_wrapper_2279405
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 

StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename-batch_normalization/gamma/Read/ReadVariableOp,batch_normalization/beta/Read/ReadVariableOp3batch_normalization/moving_mean/Read/ReadVariableOp7batch_normalization/moving_variance/Read/ReadVariableOp dense/kernel/Read/ReadVariableOpdense/bias/Read/ReadVariableOp/batch_normalization_1/gamma/Read/ReadVariableOp.batch_normalization_1/beta/Read/ReadVariableOp5batch_normalization_1/moving_mean/Read/ReadVariableOp9batch_normalization_1/moving_variance/Read/ReadVariableOp"dense_1/kernel/Read/ReadVariableOp dense_1/bias/Read/ReadVariableOp/batch_normalization_2/gamma/Read/ReadVariableOp.batch_normalization_2/beta/Read/ReadVariableOp5batch_normalization_2/moving_mean/Read/ReadVariableOp9batch_normalization_2/moving_variance/Read/ReadVariableOp"dense_2/kernel/Read/ReadVariableOp dense_2/bias/Read/ReadVariableOpbeta_1/Read/ReadVariableOpbeta_2/Read/ReadVariableOpdecay/Read/ReadVariableOp!learning_rate/Read/ReadVariableOpAdam/iter/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOptotal_2/Read/ReadVariableOpcount_2/Read/ReadVariableOptotal_3/Read/ReadVariableOpcount_3/Read/ReadVariableOp4Adam/batch_normalization/gamma/m/Read/ReadVariableOp3Adam/batch_normalization/beta/m/Read/ReadVariableOp'Adam/dense/kernel/m/Read/ReadVariableOp%Adam/dense/bias/m/Read/ReadVariableOp6Adam/batch_normalization_1/gamma/m/Read/ReadVariableOp5Adam/batch_normalization_1/beta/m/Read/ReadVariableOp)Adam/dense_1/kernel/m/Read/ReadVariableOp'Adam/dense_1/bias/m/Read/ReadVariableOp6Adam/batch_normalization_2/gamma/m/Read/ReadVariableOp5Adam/batch_normalization_2/beta/m/Read/ReadVariableOp)Adam/dense_2/kernel/m/Read/ReadVariableOp'Adam/dense_2/bias/m/Read/ReadVariableOp4Adam/batch_normalization/gamma/v/Read/ReadVariableOp3Adam/batch_normalization/beta/v/Read/ReadVariableOp'Adam/dense/kernel/v/Read/ReadVariableOp%Adam/dense/bias/v/Read/ReadVariableOp6Adam/batch_normalization_1/gamma/v/Read/ReadVariableOp5Adam/batch_normalization_1/beta/v/Read/ReadVariableOp)Adam/dense_1/kernel/v/Read/ReadVariableOp'Adam/dense_1/bias/v/Read/ReadVariableOp6Adam/batch_normalization_2/gamma/v/Read/ReadVariableOp5Adam/batch_normalization_2/beta/v/Read/ReadVariableOp)Adam/dense_2/kernel/v/Read/ReadVariableOp'Adam/dense_2/bias/v/Read/ReadVariableOpConst*D
Tin=
;29	*
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
GPU 2J 8 *)
f$R"
 __inference__traced_save_2280251
¼
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamebatch_normalization/gammabatch_normalization/betabatch_normalization/moving_mean#batch_normalization/moving_variancedense/kernel
dense/biasbatch_normalization_1/gammabatch_normalization_1/beta!batch_normalization_1/moving_mean%batch_normalization_1/moving_variancedense_1/kerneldense_1/biasbatch_normalization_2/gammabatch_normalization_2/beta!batch_normalization_2/moving_mean%batch_normalization_2/moving_variancedense_2/kerneldense_2/biasbeta_1beta_2decaylearning_rate	Adam/itertotalcounttotal_1count_1total_2count_2total_3count_3 Adam/batch_normalization/gamma/mAdam/batch_normalization/beta/mAdam/dense/kernel/mAdam/dense/bias/m"Adam/batch_normalization_1/gamma/m!Adam/batch_normalization_1/beta/mAdam/dense_1/kernel/mAdam/dense_1/bias/m"Adam/batch_normalization_2/gamma/m!Adam/batch_normalization_2/beta/mAdam/dense_2/kernel/mAdam/dense_2/bias/m Adam/batch_normalization/gamma/vAdam/batch_normalization/beta/vAdam/dense/kernel/vAdam/dense/bias/v"Adam/batch_normalization_1/gamma/v!Adam/batch_normalization_1/beta/vAdam/dense_1/kernel/vAdam/dense_1/bias/v"Adam/batch_normalization_2/gamma/v!Adam/batch_normalization_2/beta/vAdam/dense_2/kernel/vAdam/dense_2/bias/v*C
Tin<
:28*
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
GPU 2J 8 *,
f'R%
#__inference__traced_restore_2280426×ÿ
å
e
I__inference_activation_1_layer_call_and_return_conditional_losses_2278919

inputs
identityO
ReluReluinputs*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿú2
Relug
IdentityIdentityRelu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿú2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:ÿÿÿÿÿÿÿÿÿú:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿú
 
_user_specified_nameinputs
Ë*
é
P__inference_batch_normalization_layer_call_and_return_conditional_losses_2278436

inputs5
'assignmovingavg_readvariableop_resource:#7
)assignmovingavg_1_readvariableop_resource:#3
%batchnorm_mul_readvariableop_resource:#/
!batchnorm_readvariableop_resource:#
identity¢AssignMovingAvg¢AssignMovingAvg/ReadVariableOp¢AssignMovingAvg_1¢ AssignMovingAvg_1/ReadVariableOp¢batchnorm/ReadVariableOp¢batchnorm/mul/ReadVariableOp
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2 
moments/mean/reduction_indices
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:#*
	keep_dims(2
moments/mean|
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:#2
moments/StopGradient¤
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#2
moments/SquaredDifference
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2$
"moments/variance/reduction_indices²
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:#*
	keep_dims(2
moments/variance
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:#*
squeeze_dims
 2
moments/Squeeze
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:#*
squeeze_dims
 2
moments/Squeeze_1s
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2
AssignMovingAvg/decay¤
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes
:#*
dtype02 
AssignMovingAvg/ReadVariableOp
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:#2
AssignMovingAvg/sub
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:#2
AssignMovingAvg/mul¿
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype02
AssignMovingAvgw
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2
AssignMovingAvg_1/decayª
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes
:#*
dtype02"
 AssignMovingAvg_1/ReadVariableOp 
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:#2
AssignMovingAvg_1/sub
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:#2
AssignMovingAvg_1/mulÉ
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02
AssignMovingAvg_1g
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2
batchnorm/add/y
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:#2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:#2
batchnorm/Rsqrt
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:#*
dtype02
batchnorm/mul/ReadVariableOp
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:#2
batchnorm/mulv
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#2
batchnorm/mul_1{
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:#2
batchnorm/mul_2
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:#*
dtype02
batchnorm/ReadVariableOp
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:#2
batchnorm/sub
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#2
batchnorm/add_1n
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#2

Identityò
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ#: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#
 
_user_specified_nameinputs
ô
¯
P__inference_batch_normalization_layer_call_and_return_conditional_losses_2279719

inputs/
!batchnorm_readvariableop_resource:#3
%batchnorm_mul_readvariableop_resource:#1
#batchnorm_readvariableop_1_resource:#1
#batchnorm_readvariableop_2_resource:#
identity¢batchnorm/ReadVariableOp¢batchnorm/ReadVariableOp_1¢batchnorm/ReadVariableOp_2¢batchnorm/mul/ReadVariableOp
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:#*
dtype02
batchnorm/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2
batchnorm/add/y
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:#2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:#2
batchnorm/Rsqrt
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:#*
dtype02
batchnorm/mul/ReadVariableOp
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:#2
batchnorm/mulv
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#2
batchnorm/mul_1
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:#*
dtype02
batchnorm/ReadVariableOp_1
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:#2
batchnorm/mul_2
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:#*
dtype02
batchnorm/ReadVariableOp_2
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:#2
batchnorm/sub
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#2
batchnorm/add_1n
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#2

IdentityÂ
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ#: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#
 
_user_specified_nameinputs
«B
	
B__inference_model_layer_call_and_return_conditional_losses_2278960

inputs)
batch_normalization_2278845:#)
batch_normalization_2278847:#)
batch_normalization_2278849:#)
batch_normalization_2278851:# 
dense_2278871:	#ú
dense_2278873:	ú,
batch_normalization_1_2278883:	ú,
batch_normalization_1_2278885:	ú,
batch_normalization_1_2278887:	ú,
batch_normalization_1_2278889:	ú#
dense_1_2278909:
úú
dense_1_2278911:	ú,
batch_normalization_2_2278921:	ú,
batch_normalization_2_2278923:	ú,
batch_normalization_2_2278925:	ú,
batch_normalization_2_2278927:	ú"
dense_2_2278942:	ú
dense_2_2278944:
identity¢+batch_normalization/StatefulPartitionedCall¢-batch_normalization_1/StatefulPartitionedCall¢-batch_normalization_2/StatefulPartitionedCall¢dense/StatefulPartitionedCall¢.dense/kernel/Regularizer/Square/ReadVariableOp¢dense_1/StatefulPartitionedCall¢0dense_1/kernel/Regularizer/Square/ReadVariableOp¢dense_2/StatefulPartitionedCall
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCallinputsbatch_normalization_2278845batch_normalization_2278847batch_normalization_2278849batch_normalization_2278851*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Y
fTRR
P__inference_batch_normalization_layer_call_and_return_conditional_losses_22783762-
+batch_normalization/StatefulPartitionedCall·
dense/StatefulPartitionedCallStatefulPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0dense_2278871dense_2278873*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿú*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dense_layer_call_and_return_conditional_losses_22788702
dense/StatefulPartitionedCallü
activation/PartitionedCallPartitionedCall&dense/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿú* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_activation_layer_call_and_return_conditional_losses_22788812
activation/PartitionedCall¸
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall#activation/PartitionedCall:output:0batch_normalization_1_2278883batch_normalization_1_2278885batch_normalization_1_2278887batch_normalization_1_2278889*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿú*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *[
fVRT
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_22785382/
-batch_normalization_1/StatefulPartitionedCallÃ
dense_1/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0dense_1_2278909dense_1_2278911*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿú*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dense_1_layer_call_and_return_conditional_losses_22789082!
dense_1/StatefulPartitionedCall
activation_1/PartitionedCallPartitionedCall(dense_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿú* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_activation_1_layer_call_and_return_conditional_losses_22789192
activation_1/PartitionedCallº
-batch_normalization_2/StatefulPartitionedCallStatefulPartitionedCall%activation_1/PartitionedCall:output:0batch_normalization_2_2278921batch_normalization_2_2278923batch_normalization_2_2278925batch_normalization_2_2278927*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿú*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *[
fVRT
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_22787002/
-batch_normalization_2/StatefulPartitionedCallÂ
dense_2/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_2/StatefulPartitionedCall:output:0dense_2_2278942dense_2_2278944*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dense_2_layer_call_and_return_conditional_losses_22789412!
dense_2/StatefulPartitionedCall¯
.dense/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_2278871*
_output_shapes
:	#ú*
dtype020
.dense/kernel/Regularizer/Square/ReadVariableOp®
dense/kernel/Regularizer/SquareSquare6dense/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	#ú2!
dense/kernel/Regularizer/Square
dense/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2 
dense/kernel/Regularizer/Const²
dense/kernel/Regularizer/SumSum#dense/kernel/Regularizer/Square:y:0'dense/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense/kernel/Regularizer/Sum
dense/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *3Yd<2 
dense/kernel/Regularizer/mul/x´
dense/kernel/Regularizer/mulMul'dense/kernel/Regularizer/mul/x:output:0%dense/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense/kernel/Regularizer/mul¶
0dense_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_1_2278909* 
_output_shapes
:
úú*
dtype022
0dense_1/kernel/Regularizer/Square/ReadVariableOpµ
!dense_1/kernel/Regularizer/SquareSquare8dense_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
úú2#
!dense_1/kernel/Regularizer/Square
 dense_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_1/kernel/Regularizer/Constº
dense_1/kernel/Regularizer/SumSum%dense_1/kernel/Regularizer/Square:y:0)dense_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/Sum
 dense_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *3Yd<2"
 dense_1/kernel/Regularizer/mul/x¼
dense_1/kernel/Regularizer/mulMul)dense_1/kernel/Regularizer/mul/x:output:0'dense_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/mul
IdentityIdentity(dense_2/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity¤
NoOpNoOp,^batch_normalization/StatefulPartitionedCall.^batch_normalization_1/StatefulPartitionedCall.^batch_normalization_2/StatefulPartitionedCall^dense/StatefulPartitionedCall/^dense/kernel/Regularizer/Square/ReadVariableOp ^dense_1/StatefulPartitionedCall1^dense_1/kernel/Regularizer/Square/ReadVariableOp ^dense_2/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:ÿÿÿÿÿÿÿÿÿ#: : : : : : : : : : : : : : : : : : 2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2^
-batch_normalization_1/StatefulPartitionedCall-batch_normalization_1/StatefulPartitionedCall2^
-batch_normalization_2/StatefulPartitionedCall-batch_normalization_2/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2`
.dense/kernel/Regularizer/Square/ReadVariableOp.dense/kernel/Regularizer/Square/ReadVariableOp2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2d
0dense_1/kernel/Regularizer/Square/ReadVariableOp0dense_1/kernel/Regularizer/Square/ReadVariableOp2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#
 
_user_specified_nameinputs
ø

)__inference_dense_1_layer_call_fn_2279931

inputs
unknown:
úú
	unknown_0:	ú
identity¢StatefulPartitionedCallõ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿú*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dense_1_layer_call_and_return_conditional_losses_22789082
StatefulPartitionedCall|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿú2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿú: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿú
 
_user_specified_nameinputs
¨B
	
B__inference_model_layer_call_and_return_conditional_losses_2279344
input_1)
batch_normalization_2279287:#)
batch_normalization_2279289:#)
batch_normalization_2279291:#)
batch_normalization_2279293:# 
dense_2279296:	#ú
dense_2279298:	ú,
batch_normalization_1_2279302:	ú,
batch_normalization_1_2279304:	ú,
batch_normalization_1_2279306:	ú,
batch_normalization_1_2279308:	ú#
dense_1_2279311:
úú
dense_1_2279313:	ú,
batch_normalization_2_2279317:	ú,
batch_normalization_2_2279319:	ú,
batch_normalization_2_2279321:	ú,
batch_normalization_2_2279323:	ú"
dense_2_2279326:	ú
dense_2_2279328:
identity¢+batch_normalization/StatefulPartitionedCall¢-batch_normalization_1/StatefulPartitionedCall¢-batch_normalization_2/StatefulPartitionedCall¢dense/StatefulPartitionedCall¢.dense/kernel/Regularizer/Square/ReadVariableOp¢dense_1/StatefulPartitionedCall¢0dense_1/kernel/Regularizer/Square/ReadVariableOp¢dense_2/StatefulPartitionedCall
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCallinput_1batch_normalization_2279287batch_normalization_2279289batch_normalization_2279291batch_normalization_2279293*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Y
fTRR
P__inference_batch_normalization_layer_call_and_return_conditional_losses_22784362-
+batch_normalization/StatefulPartitionedCall·
dense/StatefulPartitionedCallStatefulPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0dense_2279296dense_2279298*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿú*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dense_layer_call_and_return_conditional_losses_22788702
dense/StatefulPartitionedCallü
activation/PartitionedCallPartitionedCall&dense/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿú* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_activation_layer_call_and_return_conditional_losses_22788812
activation/PartitionedCall¶
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall#activation/PartitionedCall:output:0batch_normalization_1_2279302batch_normalization_1_2279304batch_normalization_1_2279306batch_normalization_1_2279308*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿú*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *[
fVRT
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_22785982/
-batch_normalization_1/StatefulPartitionedCallÃ
dense_1/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0dense_1_2279311dense_1_2279313*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿú*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dense_1_layer_call_and_return_conditional_losses_22789082!
dense_1/StatefulPartitionedCall
activation_1/PartitionedCallPartitionedCall(dense_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿú* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_activation_1_layer_call_and_return_conditional_losses_22789192
activation_1/PartitionedCall¸
-batch_normalization_2/StatefulPartitionedCallStatefulPartitionedCall%activation_1/PartitionedCall:output:0batch_normalization_2_2279317batch_normalization_2_2279319batch_normalization_2_2279321batch_normalization_2_2279323*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿú*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *[
fVRT
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_22787602/
-batch_normalization_2/StatefulPartitionedCallÂ
dense_2/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_2/StatefulPartitionedCall:output:0dense_2_2279326dense_2_2279328*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dense_2_layer_call_and_return_conditional_losses_22789412!
dense_2/StatefulPartitionedCall¯
.dense/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_2279296*
_output_shapes
:	#ú*
dtype020
.dense/kernel/Regularizer/Square/ReadVariableOp®
dense/kernel/Regularizer/SquareSquare6dense/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	#ú2!
dense/kernel/Regularizer/Square
dense/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2 
dense/kernel/Regularizer/Const²
dense/kernel/Regularizer/SumSum#dense/kernel/Regularizer/Square:y:0'dense/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense/kernel/Regularizer/Sum
dense/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *3Yd<2 
dense/kernel/Regularizer/mul/x´
dense/kernel/Regularizer/mulMul'dense/kernel/Regularizer/mul/x:output:0%dense/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense/kernel/Regularizer/mul¶
0dense_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_1_2279311* 
_output_shapes
:
úú*
dtype022
0dense_1/kernel/Regularizer/Square/ReadVariableOpµ
!dense_1/kernel/Regularizer/SquareSquare8dense_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
úú2#
!dense_1/kernel/Regularizer/Square
 dense_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_1/kernel/Regularizer/Constº
dense_1/kernel/Regularizer/SumSum%dense_1/kernel/Regularizer/Square:y:0)dense_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/Sum
 dense_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *3Yd<2"
 dense_1/kernel/Regularizer/mul/x¼
dense_1/kernel/Regularizer/mulMul)dense_1/kernel/Regularizer/mul/x:output:0'dense_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/mul
IdentityIdentity(dense_2/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity¤
NoOpNoOp,^batch_normalization/StatefulPartitionedCall.^batch_normalization_1/StatefulPartitionedCall.^batch_normalization_2/StatefulPartitionedCall^dense/StatefulPartitionedCall/^dense/kernel/Regularizer/Square/ReadVariableOp ^dense_1/StatefulPartitionedCall1^dense_1/kernel/Regularizer/Square/ReadVariableOp ^dense_2/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:ÿÿÿÿÿÿÿÿÿ#: : : : : : : : : : : : : : : : : : 2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2^
-batch_normalization_1/StatefulPartitionedCall-batch_normalization_1/StatefulPartitionedCall2^
-batch_normalization_2/StatefulPartitionedCall-batch_normalization_2/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2`
.dense/kernel/Regularizer/Square/ReadVariableOp.dense/kernel/Regularizer/Square/ReadVariableOp2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2d
0dense_1/kernel/Regularizer/Square/ReadVariableOp0dense_1/kernel/Regularizer/Square/ReadVariableOp2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall:P L
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#
!
_user_specified_name	input_1
ß
Ö
7__inference_batch_normalization_2_layer_call_fn_2280021

inputs
unknown:	ú
	unknown_0:	ú
	unknown_1:	ú
	unknown_2:	ú
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿú*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *[
fVRT
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_22787602
StatefulPartitionedCall|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿú2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿú: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿú
 
_user_specified_nameinputs

«
D__inference_dense_1_layer_call_and_return_conditional_losses_2278908

inputs2
matmul_readvariableop_resource:
úú.
biasadd_readvariableop_resource:	ú
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp¢0dense_1/kernel/Regularizer/Square/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
úú*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿú2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:ú*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿú2	
BiasAddÅ
0dense_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
úú*
dtype022
0dense_1/kernel/Regularizer/Square/ReadVariableOpµ
!dense_1/kernel/Regularizer/SquareSquare8dense_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
úú2#
!dense_1/kernel/Regularizer/Square
 dense_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_1/kernel/Regularizer/Constº
dense_1/kernel/Regularizer/SumSum%dense_1/kernel/Regularizer/Square:y:0)dense_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/Sum
 dense_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *3Yd<2"
 dense_1/kernel/Regularizer/mul/x¼
dense_1/kernel/Regularizer/mulMul)dense_1/kernel/Regularizer/mul/x:output:0'dense_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/mull
IdentityIdentityBiasAdd:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿú2

Identity²
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp1^dense_1/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿú: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2d
0dense_1/kernel/Regularizer/Square/ReadVariableOp0dense_1/kernel/Regularizer/Square/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿú
 
_user_specified_nameinputs
ì
¦
B__inference_dense_layer_call_and_return_conditional_losses_2278870

inputs1
matmul_readvariableop_resource:	#ú.
biasadd_readvariableop_resource:	ú
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp¢.dense/kernel/Regularizer/Square/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	#ú*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿú2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:ú*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿú2	
BiasAddÀ
.dense/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	#ú*
dtype020
.dense/kernel/Regularizer/Square/ReadVariableOp®
dense/kernel/Regularizer/SquareSquare6dense/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	#ú2!
dense/kernel/Regularizer/Square
dense/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2 
dense/kernel/Regularizer/Const²
dense/kernel/Regularizer/SumSum#dense/kernel/Regularizer/Square:y:0'dense/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense/kernel/Regularizer/Sum
dense/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *3Yd<2 
dense/kernel/Regularizer/mul/x´
dense/kernel/Regularizer/mulMul'dense/kernel/Regularizer/mul/x:output:0%dense/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense/kernel/Regularizer/mull
IdentityIdentityBiasAdd:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿú2

Identity°
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp/^dense/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ#: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2`
.dense/kernel/Regularizer/Square/ReadVariableOp.dense/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#
 
_user_specified_nameinputs
ô
¯
P__inference_batch_normalization_layer_call_and_return_conditional_losses_2278376

inputs/
!batchnorm_readvariableop_resource:#3
%batchnorm_mul_readvariableop_resource:#1
#batchnorm_readvariableop_1_resource:#1
#batchnorm_readvariableop_2_resource:#
identity¢batchnorm/ReadVariableOp¢batchnorm/ReadVariableOp_1¢batchnorm/ReadVariableOp_2¢batchnorm/mul/ReadVariableOp
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:#*
dtype02
batchnorm/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2
batchnorm/add/y
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:#2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:#2
batchnorm/Rsqrt
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:#*
dtype02
batchnorm/mul/ReadVariableOp
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:#2
batchnorm/mulv
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#2
batchnorm/mul_1
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:#*
dtype02
batchnorm/ReadVariableOp_1
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:#2
batchnorm/mul_2
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:#*
dtype02
batchnorm/ReadVariableOp_2
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:#2
batchnorm/sub
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#2
batchnorm/add_1n
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#2

IdentityÂ
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ#: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#
 
_user_specified_nameinputs
é*
ï
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_2278598

inputs6
'assignmovingavg_readvariableop_resource:	ú8
)assignmovingavg_1_readvariableop_resource:	ú4
%batchnorm_mul_readvariableop_resource:	ú0
!batchnorm_readvariableop_resource:	ú
identity¢AssignMovingAvg¢AssignMovingAvg/ReadVariableOp¢AssignMovingAvg_1¢ AssignMovingAvg_1/ReadVariableOp¢batchnorm/ReadVariableOp¢batchnorm/mul/ReadVariableOp
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2 
moments/mean/reduction_indices
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	ú*
	keep_dims(2
moments/mean}
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes
:	ú2
moments/StopGradient¥
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿú2
moments/SquaredDifference
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2$
"moments/variance/reduction_indices³
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	ú*
	keep_dims(2
moments/variance
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes	
:ú*
squeeze_dims
 2
moments/Squeeze
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes	
:ú*
squeeze_dims
 2
moments/Squeeze_1s
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2
AssignMovingAvg/decay¥
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes	
:ú*
dtype02 
AssignMovingAvg/ReadVariableOp
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes	
:ú2
AssignMovingAvg/sub
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes	
:ú2
AssignMovingAvg/mul¿
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype02
AssignMovingAvgw
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2
AssignMovingAvg_1/decay«
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes	
:ú*
dtype02"
 AssignMovingAvg_1/ReadVariableOp¡
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes	
:ú2
AssignMovingAvg_1/sub
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes	
:ú2
AssignMovingAvg_1/mulÉ
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02
AssignMovingAvg_1g
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2
batchnorm/add/y
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:ú2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:ú2
batchnorm/Rsqrt
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:ú*
dtype02
batchnorm/mul/ReadVariableOp
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:ú2
batchnorm/mulw
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿú2
batchnorm/mul_1|
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes	
:ú2
batchnorm/mul_2
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:ú*
dtype02
batchnorm/ReadVariableOp
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:ú2
batchnorm/sub
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿú2
batchnorm/add_1o
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿú2

Identityò
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿú: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿú
 
_user_specified_nameinputs
Ê
J
.__inference_activation_1_layer_call_fn_2279941

inputs
identityÈ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿú* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_activation_1_layer_call_and_return_conditional_losses_22789192
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿú2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:ÿÿÿÿÿÿÿÿÿú:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿú
 
_user_specified_nameinputs
ô

)__inference_dense_2_layer_call_fn_2280041

inputs
unknown:	ú
	unknown_0:
identity¢StatefulPartitionedCallô
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dense_2_layer_call_and_return_conditional_losses_22789412
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿú: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿú
 
_user_specified_nameinputs

µ
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_2278700

inputs0
!batchnorm_readvariableop_resource:	ú4
%batchnorm_mul_readvariableop_resource:	ú2
#batchnorm_readvariableop_1_resource:	ú2
#batchnorm_readvariableop_2_resource:	ú
identity¢batchnorm/ReadVariableOp¢batchnorm/ReadVariableOp_1¢batchnorm/ReadVariableOp_2¢batchnorm/mul/ReadVariableOp
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:ú*
dtype02
batchnorm/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2
batchnorm/add/y
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:ú2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:ú2
batchnorm/Rsqrt
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:ú*
dtype02
batchnorm/mul/ReadVariableOp
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:ú2
batchnorm/mulw
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿú2
batchnorm/mul_1
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes	
:ú*
dtype02
batchnorm/ReadVariableOp_1
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes	
:ú2
batchnorm/mul_2
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes	
:ú*
dtype02
batchnorm/ReadVariableOp_2
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:ú2
batchnorm/sub
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿú2
batchnorm/add_1o
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿú2

IdentityÂ
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿú: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿú
 
_user_specified_nameinputs
á
Ö
7__inference_batch_normalization_1_layer_call_fn_2279887

inputs
unknown:	ú
	unknown_0:	ú
	unknown_1:	ú
	unknown_2:	ú
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿú*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *[
fVRT
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_22785382
StatefulPartitionedCall|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿú2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿú: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿú
 
_user_specified_nameinputs
å
e
I__inference_activation_1_layer_call_and_return_conditional_losses_2279936

inputs
identityO
ReluReluinputs*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿú2
Relug
IdentityIdentityRelu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿú2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:ÿÿÿÿÿÿÿÿÿú:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿú
 
_user_specified_nameinputs
Õ
Ð
5__inference_batch_normalization_layer_call_fn_2279766

inputs
unknown:#
	unknown_0:#
	unknown_1:#
	unknown_2:#
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Y
fTRR
P__inference_batch_normalization_layer_call_and_return_conditional_losses_22783762
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ#: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#
 
_user_specified_nameinputs
ì
¦
B__inference_dense_layer_call_and_return_conditional_losses_2279801

inputs1
matmul_readvariableop_resource:	#ú.
biasadd_readvariableop_resource:	ú
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp¢.dense/kernel/Regularizer/Square/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	#ú*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿú2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:ú*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿú2	
BiasAddÀ
.dense/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	#ú*
dtype020
.dense/kernel/Regularizer/Square/ReadVariableOp®
dense/kernel/Regularizer/SquareSquare6dense/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	#ú2!
dense/kernel/Regularizer/Square
dense/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2 
dense/kernel/Regularizer/Const²
dense/kernel/Regularizer/SumSum#dense/kernel/Regularizer/Square:y:0'dense/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense/kernel/Regularizer/Sum
dense/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *3Yd<2 
dense/kernel/Regularizer/mul/x´
dense/kernel/Regularizer/mulMul'dense/kernel/Regularizer/mul/x:output:0%dense/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense/kernel/Regularizer/mull
IdentityIdentityBiasAdd:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿú2

Identity°
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp/^dense/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ#: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2`
.dense/kernel/Regularizer/Square/ReadVariableOp.dense/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#
 
_user_specified_nameinputs
é*
ï
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_2279995

inputs6
'assignmovingavg_readvariableop_resource:	ú8
)assignmovingavg_1_readvariableop_resource:	ú4
%batchnorm_mul_readvariableop_resource:	ú0
!batchnorm_readvariableop_resource:	ú
identity¢AssignMovingAvg¢AssignMovingAvg/ReadVariableOp¢AssignMovingAvg_1¢ AssignMovingAvg_1/ReadVariableOp¢batchnorm/ReadVariableOp¢batchnorm/mul/ReadVariableOp
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2 
moments/mean/reduction_indices
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	ú*
	keep_dims(2
moments/mean}
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes
:	ú2
moments/StopGradient¥
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿú2
moments/SquaredDifference
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2$
"moments/variance/reduction_indices³
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	ú*
	keep_dims(2
moments/variance
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes	
:ú*
squeeze_dims
 2
moments/Squeeze
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes	
:ú*
squeeze_dims
 2
moments/Squeeze_1s
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2
AssignMovingAvg/decay¥
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes	
:ú*
dtype02 
AssignMovingAvg/ReadVariableOp
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes	
:ú2
AssignMovingAvg/sub
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes	
:ú2
AssignMovingAvg/mul¿
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype02
AssignMovingAvgw
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2
AssignMovingAvg_1/decay«
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes	
:ú*
dtype02"
 AssignMovingAvg_1/ReadVariableOp¡
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes	
:ú2
AssignMovingAvg_1/sub
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes	
:ú2
AssignMovingAvg_1/mulÉ
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02
AssignMovingAvg_1g
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2
batchnorm/add/y
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:ú2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:ú2
batchnorm/Rsqrt
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:ú*
dtype02
batchnorm/mul/ReadVariableOp
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:ú2
batchnorm/mulw
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿú2
batchnorm/mul_1|
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes	
:ú2
batchnorm/mul_2
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:ú*
dtype02
batchnorm/ReadVariableOp
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:ú2
batchnorm/sub
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿú2
batchnorm/add_1o
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿú2

Identityò
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿú: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿú
 
_user_specified_nameinputs

Á
'__inference_model_layer_call_fn_2279699

inputs
unknown:#
	unknown_0:#
	unknown_1:#
	unknown_2:#
	unknown_3:	#ú
	unknown_4:	ú
	unknown_5:	ú
	unknown_6:	ú
	unknown_7:	ú
	unknown_8:	ú
	unknown_9:
úú

unknown_10:	ú

unknown_11:	ú

unknown_12:	ú

unknown_13:	ú

unknown_14:	ú

unknown_15:	ú

unknown_16:
identity¢StatefulPartitionedCallÃ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_model_layer_call_and_return_conditional_losses_22791442
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:ÿÿÿÿÿÿÿÿÿ#: : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#
 
_user_specified_nameinputs
Ë*
é
P__inference_batch_normalization_layer_call_and_return_conditional_losses_2279753

inputs5
'assignmovingavg_readvariableop_resource:#7
)assignmovingavg_1_readvariableop_resource:#3
%batchnorm_mul_readvariableop_resource:#/
!batchnorm_readvariableop_resource:#
identity¢AssignMovingAvg¢AssignMovingAvg/ReadVariableOp¢AssignMovingAvg_1¢ AssignMovingAvg_1/ReadVariableOp¢batchnorm/ReadVariableOp¢batchnorm/mul/ReadVariableOp
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2 
moments/mean/reduction_indices
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:#*
	keep_dims(2
moments/mean|
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:#2
moments/StopGradient¤
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#2
moments/SquaredDifference
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2$
"moments/variance/reduction_indices²
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:#*
	keep_dims(2
moments/variance
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:#*
squeeze_dims
 2
moments/Squeeze
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:#*
squeeze_dims
 2
moments/Squeeze_1s
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2
AssignMovingAvg/decay¤
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes
:#*
dtype02 
AssignMovingAvg/ReadVariableOp
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:#2
AssignMovingAvg/sub
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:#2
AssignMovingAvg/mul¿
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype02
AssignMovingAvgw
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2
AssignMovingAvg_1/decayª
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes
:#*
dtype02"
 AssignMovingAvg_1/ReadVariableOp 
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:#2
AssignMovingAvg_1/sub
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:#2
AssignMovingAvg_1/mulÉ
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02
AssignMovingAvg_1g
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2
batchnorm/add/y
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:#2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:#2
batchnorm/Rsqrt
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:#*
dtype02
batchnorm/mul/ReadVariableOp
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:#2
batchnorm/mulv
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#2
batchnorm/mul_1{
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:#2
batchnorm/mul_2
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:#*
dtype02
batchnorm/ReadVariableOp
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:#2
batchnorm/sub
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#2
batchnorm/add_1n
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#2

Identityò
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ#: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#
 
_user_specified_nameinputs
¥B
	
B__inference_model_layer_call_and_return_conditional_losses_2279144

inputs)
batch_normalization_2279087:#)
batch_normalization_2279089:#)
batch_normalization_2279091:#)
batch_normalization_2279093:# 
dense_2279096:	#ú
dense_2279098:	ú,
batch_normalization_1_2279102:	ú,
batch_normalization_1_2279104:	ú,
batch_normalization_1_2279106:	ú,
batch_normalization_1_2279108:	ú#
dense_1_2279111:
úú
dense_1_2279113:	ú,
batch_normalization_2_2279117:	ú,
batch_normalization_2_2279119:	ú,
batch_normalization_2_2279121:	ú,
batch_normalization_2_2279123:	ú"
dense_2_2279126:	ú
dense_2_2279128:
identity¢+batch_normalization/StatefulPartitionedCall¢-batch_normalization_1/StatefulPartitionedCall¢-batch_normalization_2/StatefulPartitionedCall¢dense/StatefulPartitionedCall¢.dense/kernel/Regularizer/Square/ReadVariableOp¢dense_1/StatefulPartitionedCall¢0dense_1/kernel/Regularizer/Square/ReadVariableOp¢dense_2/StatefulPartitionedCall
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCallinputsbatch_normalization_2279087batch_normalization_2279089batch_normalization_2279091batch_normalization_2279093*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Y
fTRR
P__inference_batch_normalization_layer_call_and_return_conditional_losses_22784362-
+batch_normalization/StatefulPartitionedCall·
dense/StatefulPartitionedCallStatefulPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0dense_2279096dense_2279098*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿú*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dense_layer_call_and_return_conditional_losses_22788702
dense/StatefulPartitionedCallü
activation/PartitionedCallPartitionedCall&dense/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿú* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_activation_layer_call_and_return_conditional_losses_22788812
activation/PartitionedCall¶
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall#activation/PartitionedCall:output:0batch_normalization_1_2279102batch_normalization_1_2279104batch_normalization_1_2279106batch_normalization_1_2279108*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿú*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *[
fVRT
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_22785982/
-batch_normalization_1/StatefulPartitionedCallÃ
dense_1/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0dense_1_2279111dense_1_2279113*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿú*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dense_1_layer_call_and_return_conditional_losses_22789082!
dense_1/StatefulPartitionedCall
activation_1/PartitionedCallPartitionedCall(dense_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿú* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_activation_1_layer_call_and_return_conditional_losses_22789192
activation_1/PartitionedCall¸
-batch_normalization_2/StatefulPartitionedCallStatefulPartitionedCall%activation_1/PartitionedCall:output:0batch_normalization_2_2279117batch_normalization_2_2279119batch_normalization_2_2279121batch_normalization_2_2279123*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿú*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *[
fVRT
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_22787602/
-batch_normalization_2/StatefulPartitionedCallÂ
dense_2/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_2/StatefulPartitionedCall:output:0dense_2_2279126dense_2_2279128*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dense_2_layer_call_and_return_conditional_losses_22789412!
dense_2/StatefulPartitionedCall¯
.dense/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_2279096*
_output_shapes
:	#ú*
dtype020
.dense/kernel/Regularizer/Square/ReadVariableOp®
dense/kernel/Regularizer/SquareSquare6dense/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	#ú2!
dense/kernel/Regularizer/Square
dense/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2 
dense/kernel/Regularizer/Const²
dense/kernel/Regularizer/SumSum#dense/kernel/Regularizer/Square:y:0'dense/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense/kernel/Regularizer/Sum
dense/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *3Yd<2 
dense/kernel/Regularizer/mul/x´
dense/kernel/Regularizer/mulMul'dense/kernel/Regularizer/mul/x:output:0%dense/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense/kernel/Regularizer/mul¶
0dense_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_1_2279111* 
_output_shapes
:
úú*
dtype022
0dense_1/kernel/Regularizer/Square/ReadVariableOpµ
!dense_1/kernel/Regularizer/SquareSquare8dense_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
úú2#
!dense_1/kernel/Regularizer/Square
 dense_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_1/kernel/Regularizer/Constº
dense_1/kernel/Regularizer/SumSum%dense_1/kernel/Regularizer/Square:y:0)dense_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/Sum
 dense_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *3Yd<2"
 dense_1/kernel/Regularizer/mul/x¼
dense_1/kernel/Regularizer/mulMul)dense_1/kernel/Regularizer/mul/x:output:0'dense_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/mul
IdentityIdentity(dense_2/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity¤
NoOpNoOp,^batch_normalization/StatefulPartitionedCall.^batch_normalization_1/StatefulPartitionedCall.^batch_normalization_2/StatefulPartitionedCall^dense/StatefulPartitionedCall/^dense/kernel/Regularizer/Square/ReadVariableOp ^dense_1/StatefulPartitionedCall1^dense_1/kernel/Regularizer/Square/ReadVariableOp ^dense_2/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:ÿÿÿÿÿÿÿÿÿ#: : : : : : : : : : : : : : : : : : 2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2^
-batch_normalization_1/StatefulPartitionedCall-batch_normalization_1/StatefulPartitionedCall2^
-batch_normalization_2/StatefulPartitionedCall-batch_normalization_2/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2`
.dense/kernel/Regularizer/Square/ReadVariableOp.dense/kernel/Regularizer/Square/ReadVariableOp2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2d
0dense_1/kernel/Regularizer/Square/ReadVariableOp0dense_1/kernel/Regularizer/Square/ReadVariableOp2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#
 
_user_specified_nameinputs

ö
D__inference_dense_2_layer_call_and_return_conditional_losses_2278941

inputs1
matmul_readvariableop_resource:	ú-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	ú*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAdda
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
Sigmoidf
IdentityIdentitySigmoid:y:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿú: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿú
 
_user_specified_nameinputs

«
D__inference_dense_1_layer_call_and_return_conditional_losses_2279922

inputs2
matmul_readvariableop_resource:
úú.
biasadd_readvariableop_resource:	ú
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp¢0dense_1/kernel/Regularizer/Square/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
úú*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿú2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:ú*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿú2	
BiasAddÅ
0dense_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
úú*
dtype022
0dense_1/kernel/Regularizer/Square/ReadVariableOpµ
!dense_1/kernel/Regularizer/SquareSquare8dense_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
úú2#
!dense_1/kernel/Regularizer/Square
 dense_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_1/kernel/Regularizer/Constº
dense_1/kernel/Regularizer/SumSum%dense_1/kernel/Regularizer/Square:y:0)dense_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/Sum
 dense_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *3Yd<2"
 dense_1/kernel/Regularizer/mul/x¼
dense_1/kernel/Regularizer/mulMul)dense_1/kernel/Regularizer/mul/x:output:0'dense_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/mull
IdentityIdentityBiasAdd:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿú2

Identity²
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp1^dense_1/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿú: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2d
0dense_1/kernel/Regularizer/Square/ReadVariableOp0dense_1/kernel/Regularizer/Square/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿú
 
_user_specified_nameinputs
á
Ö
7__inference_batch_normalization_2_layer_call_fn_2280008

inputs
unknown:	ú
	unknown_0:	ú
	unknown_1:	ú
	unknown_2:	ú
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿú*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *[
fVRT
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_22787002
StatefulPartitionedCall|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿú2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿú: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿú
 
_user_specified_nameinputs
Æ
H
,__inference_activation_layer_call_fn_2279820

inputs
identityÆ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿú* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_activation_layer_call_and_return_conditional_losses_22788812
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿú2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:ÿÿÿÿÿÿÿÿÿú:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿú
 
_user_specified_nameinputs
ò
À
%__inference_signature_wrapper_2279405
input_1
unknown:#
	unknown_0:#
	unknown_1:#
	unknown_2:#
	unknown_3:	#ú
	unknown_4:	ú
	unknown_5:	ú
	unknown_6:	ú
	unknown_7:	ú
	unknown_8:	ú
	unknown_9:
úú

unknown_10:	ú

unknown_11:	ú

unknown_12:	ú

unknown_13:	ú

unknown_14:	ú

unknown_15:	ú

unknown_16:
identity¢StatefulPartitionedCallª
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*4
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *+
f&R$
"__inference__wrapped_model_22783522
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:ÿÿÿÿÿÿÿÿÿ#: : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#
!
_user_specified_name	input_1
Êo

 __inference__traced_save_2280251
file_prefix8
4savev2_batch_normalization_gamma_read_readvariableop7
3savev2_batch_normalization_beta_read_readvariableop>
:savev2_batch_normalization_moving_mean_read_readvariableopB
>savev2_batch_normalization_moving_variance_read_readvariableop+
'savev2_dense_kernel_read_readvariableop)
%savev2_dense_bias_read_readvariableop:
6savev2_batch_normalization_1_gamma_read_readvariableop9
5savev2_batch_normalization_1_beta_read_readvariableop@
<savev2_batch_normalization_1_moving_mean_read_readvariableopD
@savev2_batch_normalization_1_moving_variance_read_readvariableop-
)savev2_dense_1_kernel_read_readvariableop+
'savev2_dense_1_bias_read_readvariableop:
6savev2_batch_normalization_2_gamma_read_readvariableop9
5savev2_batch_normalization_2_beta_read_readvariableop@
<savev2_batch_normalization_2_moving_mean_read_readvariableopD
@savev2_batch_normalization_2_moving_variance_read_readvariableop-
)savev2_dense_2_kernel_read_readvariableop+
'savev2_dense_2_bias_read_readvariableop%
!savev2_beta_1_read_readvariableop%
!savev2_beta_2_read_readvariableop$
 savev2_decay_read_readvariableop,
(savev2_learning_rate_read_readvariableop(
$savev2_adam_iter_read_readvariableop	$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop&
"savev2_total_2_read_readvariableop&
"savev2_count_2_read_readvariableop&
"savev2_total_3_read_readvariableop&
"savev2_count_3_read_readvariableop?
;savev2_adam_batch_normalization_gamma_m_read_readvariableop>
:savev2_adam_batch_normalization_beta_m_read_readvariableop2
.savev2_adam_dense_kernel_m_read_readvariableop0
,savev2_adam_dense_bias_m_read_readvariableopA
=savev2_adam_batch_normalization_1_gamma_m_read_readvariableop@
<savev2_adam_batch_normalization_1_beta_m_read_readvariableop4
0savev2_adam_dense_1_kernel_m_read_readvariableop2
.savev2_adam_dense_1_bias_m_read_readvariableopA
=savev2_adam_batch_normalization_2_gamma_m_read_readvariableop@
<savev2_adam_batch_normalization_2_beta_m_read_readvariableop4
0savev2_adam_dense_2_kernel_m_read_readvariableop2
.savev2_adam_dense_2_bias_m_read_readvariableop?
;savev2_adam_batch_normalization_gamma_v_read_readvariableop>
:savev2_adam_batch_normalization_beta_v_read_readvariableop2
.savev2_adam_dense_kernel_v_read_readvariableop0
,savev2_adam_dense_bias_v_read_readvariableopA
=savev2_adam_batch_normalization_1_gamma_v_read_readvariableop@
<savev2_adam_batch_normalization_1_beta_v_read_readvariableop4
0savev2_adam_dense_1_kernel_v_read_readvariableop2
.savev2_adam_dense_1_bias_v_read_readvariableopA
=savev2_adam_batch_normalization_2_gamma_v_read_readvariableop@
<savev2_adam_batch_normalization_2_beta_v_read_readvariableop4
0savev2_adam_dense_2_kernel_v_read_readvariableop2
.savev2_adam_dense_2_bias_v_read_readvariableop
savev2_const

identity_1¢MergeV2Checkpoints
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Constl
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part2	
Const_1
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard¦
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:8*
dtype0*
valueB8B5layer_with_weights-0/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-0/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-0/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-2/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-2/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-2/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-4/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-4/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-4/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/3/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/3/count/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-0/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-2/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-4/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-0/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-2/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-4/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_namesù
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:8*
dtype0*
valuezBx8B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices¬
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:04savev2_batch_normalization_gamma_read_readvariableop3savev2_batch_normalization_beta_read_readvariableop:savev2_batch_normalization_moving_mean_read_readvariableop>savev2_batch_normalization_moving_variance_read_readvariableop'savev2_dense_kernel_read_readvariableop%savev2_dense_bias_read_readvariableop6savev2_batch_normalization_1_gamma_read_readvariableop5savev2_batch_normalization_1_beta_read_readvariableop<savev2_batch_normalization_1_moving_mean_read_readvariableop@savev2_batch_normalization_1_moving_variance_read_readvariableop)savev2_dense_1_kernel_read_readvariableop'savev2_dense_1_bias_read_readvariableop6savev2_batch_normalization_2_gamma_read_readvariableop5savev2_batch_normalization_2_beta_read_readvariableop<savev2_batch_normalization_2_moving_mean_read_readvariableop@savev2_batch_normalization_2_moving_variance_read_readvariableop)savev2_dense_2_kernel_read_readvariableop'savev2_dense_2_bias_read_readvariableop!savev2_beta_1_read_readvariableop!savev2_beta_2_read_readvariableop savev2_decay_read_readvariableop(savev2_learning_rate_read_readvariableop$savev2_adam_iter_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop"savev2_total_2_read_readvariableop"savev2_count_2_read_readvariableop"savev2_total_3_read_readvariableop"savev2_count_3_read_readvariableop;savev2_adam_batch_normalization_gamma_m_read_readvariableop:savev2_adam_batch_normalization_beta_m_read_readvariableop.savev2_adam_dense_kernel_m_read_readvariableop,savev2_adam_dense_bias_m_read_readvariableop=savev2_adam_batch_normalization_1_gamma_m_read_readvariableop<savev2_adam_batch_normalization_1_beta_m_read_readvariableop0savev2_adam_dense_1_kernel_m_read_readvariableop.savev2_adam_dense_1_bias_m_read_readvariableop=savev2_adam_batch_normalization_2_gamma_m_read_readvariableop<savev2_adam_batch_normalization_2_beta_m_read_readvariableop0savev2_adam_dense_2_kernel_m_read_readvariableop.savev2_adam_dense_2_bias_m_read_readvariableop;savev2_adam_batch_normalization_gamma_v_read_readvariableop:savev2_adam_batch_normalization_beta_v_read_readvariableop.savev2_adam_dense_kernel_v_read_readvariableop,savev2_adam_dense_bias_v_read_readvariableop=savev2_adam_batch_normalization_1_gamma_v_read_readvariableop<savev2_adam_batch_normalization_1_beta_v_read_readvariableop0savev2_adam_dense_1_kernel_v_read_readvariableop.savev2_adam_dense_1_bias_v_read_readvariableop=savev2_adam_batch_normalization_2_gamma_v_read_readvariableop<savev2_adam_batch_normalization_2_beta_v_read_readvariableop0savev2_adam_dense_2_kernel_v_read_readvariableop.savev2_adam_dense_2_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *F
dtypes<
:28	2
SaveV2º
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes¡
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identity_

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: 2

Identity_1c
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"!

identity_1Identity_1:output:0*õ
_input_shapesã
à: :#:#:#:#:	#ú:ú:ú:ú:ú:ú:
úú:ú:ú:ú:ú:ú:	ú:: : : : : : : : : : : : : :#:#:	#ú:ú:ú:ú:
úú:ú:ú:ú:	ú::#:#:	#ú:ú:ú:ú:
úú:ú:ú:ú:	ú:: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix: 

_output_shapes
:#: 

_output_shapes
:#: 

_output_shapes
:#: 

_output_shapes
:#:%!

_output_shapes
:	#ú:!

_output_shapes	
:ú:!

_output_shapes	
:ú:!

_output_shapes	
:ú:!	

_output_shapes	
:ú:!


_output_shapes	
:ú:&"
 
_output_shapes
:
úú:!

_output_shapes	
:ú:!

_output_shapes	
:ú:!

_output_shapes	
:ú:!

_output_shapes	
:ú:!

_output_shapes	
:ú:%!

_output_shapes
:	ú: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :  

_output_shapes
:#: !

_output_shapes
:#:%"!

_output_shapes
:	#ú:!#

_output_shapes	
:ú:!$

_output_shapes	
:ú:!%

_output_shapes	
:ú:&&"
 
_output_shapes
:
úú:!'

_output_shapes	
:ú:!(

_output_shapes	
:ú:!)

_output_shapes	
:ú:%*!

_output_shapes
:	ú: +

_output_shapes
:: ,

_output_shapes
:#: -

_output_shapes
:#:%.!

_output_shapes
:	#ú:!/

_output_shapes	
:ú:!0

_output_shapes	
:ú:!1

_output_shapes	
:ú:&2"
 
_output_shapes
:
úú:!3

_output_shapes	
:ú:!4

_output_shapes	
:ú:!5

_output_shapes	
:ú:%6!

_output_shapes
:	ú: 7

_output_shapes
::8

_output_shapes
: 
é*
ï
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_2279874

inputs6
'assignmovingavg_readvariableop_resource:	ú8
)assignmovingavg_1_readvariableop_resource:	ú4
%batchnorm_mul_readvariableop_resource:	ú0
!batchnorm_readvariableop_resource:	ú
identity¢AssignMovingAvg¢AssignMovingAvg/ReadVariableOp¢AssignMovingAvg_1¢ AssignMovingAvg_1/ReadVariableOp¢batchnorm/ReadVariableOp¢batchnorm/mul/ReadVariableOp
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2 
moments/mean/reduction_indices
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	ú*
	keep_dims(2
moments/mean}
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes
:	ú2
moments/StopGradient¥
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿú2
moments/SquaredDifference
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2$
"moments/variance/reduction_indices³
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	ú*
	keep_dims(2
moments/variance
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes	
:ú*
squeeze_dims
 2
moments/Squeeze
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes	
:ú*
squeeze_dims
 2
moments/Squeeze_1s
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2
AssignMovingAvg/decay¥
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes	
:ú*
dtype02 
AssignMovingAvg/ReadVariableOp
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes	
:ú2
AssignMovingAvg/sub
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes	
:ú2
AssignMovingAvg/mul¿
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype02
AssignMovingAvgw
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2
AssignMovingAvg_1/decay«
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes	
:ú*
dtype02"
 AssignMovingAvg_1/ReadVariableOp¡
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes	
:ú2
AssignMovingAvg_1/sub
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes	
:ú2
AssignMovingAvg_1/mulÉ
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02
AssignMovingAvg_1g
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2
batchnorm/add/y
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:ú2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:ú2
batchnorm/Rsqrt
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:ú*
dtype02
batchnorm/mul/ReadVariableOp
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:ú2
batchnorm/mulw
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿú2
batchnorm/mul_1|
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes	
:ú2
batchnorm/mul_2
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:ú*
dtype02
batchnorm/ReadVariableOp
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:ú2
batchnorm/sub
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿú2
batchnorm/add_1o
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿú2

Identityò
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿú: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿú
 
_user_specified_nameinputs
ß
Ö
7__inference_batch_normalization_1_layer_call_fn_2279900

inputs
unknown:	ú
	unknown_0:	ú
	unknown_1:	ú
	unknown_2:	ú
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿú*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *[
fVRT
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_22785982
StatefulPartitionedCall|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿú2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿú: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿú
 
_user_specified_nameinputs

ö
D__inference_dense_2_layer_call_and_return_conditional_losses_2280032

inputs1
matmul_readvariableop_resource:	ú-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	ú*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAdda
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
Sigmoidf
IdentityIdentitySigmoid:y:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿú: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿú
 
_user_specified_nameinputs

µ
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_2279840

inputs0
!batchnorm_readvariableop_resource:	ú4
%batchnorm_mul_readvariableop_resource:	ú2
#batchnorm_readvariableop_1_resource:	ú2
#batchnorm_readvariableop_2_resource:	ú
identity¢batchnorm/ReadVariableOp¢batchnorm/ReadVariableOp_1¢batchnorm/ReadVariableOp_2¢batchnorm/mul/ReadVariableOp
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:ú*
dtype02
batchnorm/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2
batchnorm/add/y
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:ú2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:ú2
batchnorm/Rsqrt
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:ú*
dtype02
batchnorm/mul/ReadVariableOp
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:ú2
batchnorm/mulw
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿú2
batchnorm/mul_1
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes	
:ú*
dtype02
batchnorm/ReadVariableOp_1
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes	
:ú2
batchnorm/mul_2
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes	
:ú*
dtype02
batchnorm/ReadVariableOp_2
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:ú2
batchnorm/sub
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿú2
batchnorm/add_1o
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿú2

IdentityÂ
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿú: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿú
 
_user_specified_nameinputs
ô
­
__inference_loss_fn_0_2280052J
7dense_kernel_regularizer_square_readvariableop_resource:	#ú
identity¢.dense/kernel/Regularizer/Square/ReadVariableOpÙ
.dense/kernel/Regularizer/Square/ReadVariableOpReadVariableOp7dense_kernel_regularizer_square_readvariableop_resource*
_output_shapes
:	#ú*
dtype020
.dense/kernel/Regularizer/Square/ReadVariableOp®
dense/kernel/Regularizer/SquareSquare6dense/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	#ú2!
dense/kernel/Regularizer/Square
dense/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2 
dense/kernel/Regularizer/Const²
dense/kernel/Regularizer/SumSum#dense/kernel/Regularizer/Square:y:0'dense/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense/kernel/Regularizer/Sum
dense/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *3Yd<2 
dense/kernel/Regularizer/mul/x´
dense/kernel/Regularizer/mulMul'dense/kernel/Regularizer/mul/x:output:0%dense/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense/kernel/Regularizer/mulj
IdentityIdentity dense/kernel/Regularizer/mul:z:0^NoOp*
T0*
_output_shapes
: 2

Identity
NoOpNoOp/^dense/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2`
.dense/kernel/Regularizer/Square/ReadVariableOp.dense/kernel/Regularizer/Square/ReadVariableOp

µ
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_2279961

inputs0
!batchnorm_readvariableop_resource:	ú4
%batchnorm_mul_readvariableop_resource:	ú2
#batchnorm_readvariableop_1_resource:	ú2
#batchnorm_readvariableop_2_resource:	ú
identity¢batchnorm/ReadVariableOp¢batchnorm/ReadVariableOp_1¢batchnorm/ReadVariableOp_2¢batchnorm/mul/ReadVariableOp
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:ú*
dtype02
batchnorm/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2
batchnorm/add/y
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:ú2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:ú2
batchnorm/Rsqrt
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:ú*
dtype02
batchnorm/mul/ReadVariableOp
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:ú2
batchnorm/mulw
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿú2
batchnorm/mul_1
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes	
:ú*
dtype02
batchnorm/ReadVariableOp_1
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes	
:ú2
batchnorm/mul_2
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes	
:ú*
dtype02
batchnorm/ReadVariableOp_2
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:ú2
batchnorm/sub
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿú2
batchnorm/add_1o
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿú2

IdentityÂ
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿú: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿú
 
_user_specified_nameinputs
®B
	
B__inference_model_layer_call_and_return_conditional_losses_2279284
input_1)
batch_normalization_2279227:#)
batch_normalization_2279229:#)
batch_normalization_2279231:#)
batch_normalization_2279233:# 
dense_2279236:	#ú
dense_2279238:	ú,
batch_normalization_1_2279242:	ú,
batch_normalization_1_2279244:	ú,
batch_normalization_1_2279246:	ú,
batch_normalization_1_2279248:	ú#
dense_1_2279251:
úú
dense_1_2279253:	ú,
batch_normalization_2_2279257:	ú,
batch_normalization_2_2279259:	ú,
batch_normalization_2_2279261:	ú,
batch_normalization_2_2279263:	ú"
dense_2_2279266:	ú
dense_2_2279268:
identity¢+batch_normalization/StatefulPartitionedCall¢-batch_normalization_1/StatefulPartitionedCall¢-batch_normalization_2/StatefulPartitionedCall¢dense/StatefulPartitionedCall¢.dense/kernel/Regularizer/Square/ReadVariableOp¢dense_1/StatefulPartitionedCall¢0dense_1/kernel/Regularizer/Square/ReadVariableOp¢dense_2/StatefulPartitionedCall
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCallinput_1batch_normalization_2279227batch_normalization_2279229batch_normalization_2279231batch_normalization_2279233*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Y
fTRR
P__inference_batch_normalization_layer_call_and_return_conditional_losses_22783762-
+batch_normalization/StatefulPartitionedCall·
dense/StatefulPartitionedCallStatefulPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0dense_2279236dense_2279238*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿú*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dense_layer_call_and_return_conditional_losses_22788702
dense/StatefulPartitionedCallü
activation/PartitionedCallPartitionedCall&dense/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿú* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_activation_layer_call_and_return_conditional_losses_22788812
activation/PartitionedCall¸
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall#activation/PartitionedCall:output:0batch_normalization_1_2279242batch_normalization_1_2279244batch_normalization_1_2279246batch_normalization_1_2279248*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿú*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *[
fVRT
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_22785382/
-batch_normalization_1/StatefulPartitionedCallÃ
dense_1/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0dense_1_2279251dense_1_2279253*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿú*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dense_1_layer_call_and_return_conditional_losses_22789082!
dense_1/StatefulPartitionedCall
activation_1/PartitionedCallPartitionedCall(dense_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿú* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_activation_1_layer_call_and_return_conditional_losses_22789192
activation_1/PartitionedCallº
-batch_normalization_2/StatefulPartitionedCallStatefulPartitionedCall%activation_1/PartitionedCall:output:0batch_normalization_2_2279257batch_normalization_2_2279259batch_normalization_2_2279261batch_normalization_2_2279263*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿú*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *[
fVRT
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_22787002/
-batch_normalization_2/StatefulPartitionedCallÂ
dense_2/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_2/StatefulPartitionedCall:output:0dense_2_2279266dense_2_2279268*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dense_2_layer_call_and_return_conditional_losses_22789412!
dense_2/StatefulPartitionedCall¯
.dense/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_2279236*
_output_shapes
:	#ú*
dtype020
.dense/kernel/Regularizer/Square/ReadVariableOp®
dense/kernel/Regularizer/SquareSquare6dense/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	#ú2!
dense/kernel/Regularizer/Square
dense/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2 
dense/kernel/Regularizer/Const²
dense/kernel/Regularizer/SumSum#dense/kernel/Regularizer/Square:y:0'dense/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense/kernel/Regularizer/Sum
dense/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *3Yd<2 
dense/kernel/Regularizer/mul/x´
dense/kernel/Regularizer/mulMul'dense/kernel/Regularizer/mul/x:output:0%dense/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense/kernel/Regularizer/mul¶
0dense_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_1_2279251* 
_output_shapes
:
úú*
dtype022
0dense_1/kernel/Regularizer/Square/ReadVariableOpµ
!dense_1/kernel/Regularizer/SquareSquare8dense_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
úú2#
!dense_1/kernel/Regularizer/Square
 dense_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_1/kernel/Regularizer/Constº
dense_1/kernel/Regularizer/SumSum%dense_1/kernel/Regularizer/Square:y:0)dense_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/Sum
 dense_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *3Yd<2"
 dense_1/kernel/Regularizer/mul/x¼
dense_1/kernel/Regularizer/mulMul)dense_1/kernel/Regularizer/mul/x:output:0'dense_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/mul
IdentityIdentity(dense_2/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity¤
NoOpNoOp,^batch_normalization/StatefulPartitionedCall.^batch_normalization_1/StatefulPartitionedCall.^batch_normalization_2/StatefulPartitionedCall^dense/StatefulPartitionedCall/^dense/kernel/Regularizer/Square/ReadVariableOp ^dense_1/StatefulPartitionedCall1^dense_1/kernel/Regularizer/Square/ReadVariableOp ^dense_2/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:ÿÿÿÿÿÿÿÿÿ#: : : : : : : : : : : : : : : : : : 2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2^
-batch_normalization_1/StatefulPartitionedCall-batch_normalization_1/StatefulPartitionedCall2^
-batch_normalization_2/StatefulPartitionedCall-batch_normalization_2/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2`
.dense/kernel/Regularizer/Square/ReadVariableOp.dense/kernel/Regularizer/Square/ReadVariableOp2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2d
0dense_1/kernel/Regularizer/Square/ReadVariableOp0dense_1/kernel/Regularizer/Square/ReadVariableOp2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall:P L
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#
!
_user_specified_name	input_1
í
º"
#__inference__traced_restore_2280426
file_prefix8
*assignvariableop_batch_normalization_gamma:#9
+assignvariableop_1_batch_normalization_beta:#@
2assignvariableop_2_batch_normalization_moving_mean:#D
6assignvariableop_3_batch_normalization_moving_variance:#2
assignvariableop_4_dense_kernel:	#ú,
assignvariableop_5_dense_bias:	ú=
.assignvariableop_6_batch_normalization_1_gamma:	ú<
-assignvariableop_7_batch_normalization_1_beta:	úC
4assignvariableop_8_batch_normalization_1_moving_mean:	úG
8assignvariableop_9_batch_normalization_1_moving_variance:	ú6
"assignvariableop_10_dense_1_kernel:
úú/
 assignvariableop_11_dense_1_bias:	ú>
/assignvariableop_12_batch_normalization_2_gamma:	ú=
.assignvariableop_13_batch_normalization_2_beta:	úD
5assignvariableop_14_batch_normalization_2_moving_mean:	úH
9assignvariableop_15_batch_normalization_2_moving_variance:	ú5
"assignvariableop_16_dense_2_kernel:	ú.
 assignvariableop_17_dense_2_bias:$
assignvariableop_18_beta_1: $
assignvariableop_19_beta_2: #
assignvariableop_20_decay: +
!assignvariableop_21_learning_rate: '
assignvariableop_22_adam_iter:	 #
assignvariableop_23_total: #
assignvariableop_24_count: %
assignvariableop_25_total_1: %
assignvariableop_26_count_1: %
assignvariableop_27_total_2: %
assignvariableop_28_count_2: %
assignvariableop_29_total_3: %
assignvariableop_30_count_3: B
4assignvariableop_31_adam_batch_normalization_gamma_m:#A
3assignvariableop_32_adam_batch_normalization_beta_m:#:
'assignvariableop_33_adam_dense_kernel_m:	#ú4
%assignvariableop_34_adam_dense_bias_m:	úE
6assignvariableop_35_adam_batch_normalization_1_gamma_m:	úD
5assignvariableop_36_adam_batch_normalization_1_beta_m:	ú=
)assignvariableop_37_adam_dense_1_kernel_m:
úú6
'assignvariableop_38_adam_dense_1_bias_m:	úE
6assignvariableop_39_adam_batch_normalization_2_gamma_m:	úD
5assignvariableop_40_adam_batch_normalization_2_beta_m:	ú<
)assignvariableop_41_adam_dense_2_kernel_m:	ú5
'assignvariableop_42_adam_dense_2_bias_m:B
4assignvariableop_43_adam_batch_normalization_gamma_v:#A
3assignvariableop_44_adam_batch_normalization_beta_v:#:
'assignvariableop_45_adam_dense_kernel_v:	#ú4
%assignvariableop_46_adam_dense_bias_v:	úE
6assignvariableop_47_adam_batch_normalization_1_gamma_v:	úD
5assignvariableop_48_adam_batch_normalization_1_beta_v:	ú=
)assignvariableop_49_adam_dense_1_kernel_v:
úú6
'assignvariableop_50_adam_dense_1_bias_v:	úE
6assignvariableop_51_adam_batch_normalization_2_gamma_v:	úD
5assignvariableop_52_adam_batch_normalization_2_beta_v:	ú<
)assignvariableop_53_adam_dense_2_kernel_v:	ú5
'assignvariableop_54_adam_dense_2_bias_v:
identity_56¢AssignVariableOp¢AssignVariableOp_1¢AssignVariableOp_10¢AssignVariableOp_11¢AssignVariableOp_12¢AssignVariableOp_13¢AssignVariableOp_14¢AssignVariableOp_15¢AssignVariableOp_16¢AssignVariableOp_17¢AssignVariableOp_18¢AssignVariableOp_19¢AssignVariableOp_2¢AssignVariableOp_20¢AssignVariableOp_21¢AssignVariableOp_22¢AssignVariableOp_23¢AssignVariableOp_24¢AssignVariableOp_25¢AssignVariableOp_26¢AssignVariableOp_27¢AssignVariableOp_28¢AssignVariableOp_29¢AssignVariableOp_3¢AssignVariableOp_30¢AssignVariableOp_31¢AssignVariableOp_32¢AssignVariableOp_33¢AssignVariableOp_34¢AssignVariableOp_35¢AssignVariableOp_36¢AssignVariableOp_37¢AssignVariableOp_38¢AssignVariableOp_39¢AssignVariableOp_4¢AssignVariableOp_40¢AssignVariableOp_41¢AssignVariableOp_42¢AssignVariableOp_43¢AssignVariableOp_44¢AssignVariableOp_45¢AssignVariableOp_46¢AssignVariableOp_47¢AssignVariableOp_48¢AssignVariableOp_49¢AssignVariableOp_5¢AssignVariableOp_50¢AssignVariableOp_51¢AssignVariableOp_52¢AssignVariableOp_53¢AssignVariableOp_54¢AssignVariableOp_6¢AssignVariableOp_7¢AssignVariableOp_8¢AssignVariableOp_9
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:8*
dtype0*
valueB8B5layer_with_weights-0/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-0/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-0/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-2/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-2/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-2/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-4/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-4/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-4/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/3/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/3/count/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-0/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-2/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-4/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-0/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-2/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-4/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_namesÿ
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:8*
dtype0*
valuezBx8B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slicesÆ
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*ö
_output_shapesã
à::::::::::::::::::::::::::::::::::::::::::::::::::::::::*F
dtypes<
:28	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity©
AssignVariableOpAssignVariableOp*assignvariableop_batch_normalization_gammaIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1°
AssignVariableOp_1AssignVariableOp+assignvariableop_1_batch_normalization_betaIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2·
AssignVariableOp_2AssignVariableOp2assignvariableop_2_batch_normalization_moving_meanIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3»
AssignVariableOp_3AssignVariableOp6assignvariableop_3_batch_normalization_moving_varianceIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4¤
AssignVariableOp_4AssignVariableOpassignvariableop_4_dense_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5¢
AssignVariableOp_5AssignVariableOpassignvariableop_5_dense_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6³
AssignVariableOp_6AssignVariableOp.assignvariableop_6_batch_normalization_1_gammaIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7²
AssignVariableOp_7AssignVariableOp-assignvariableop_7_batch_normalization_1_betaIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8¹
AssignVariableOp_8AssignVariableOp4assignvariableop_8_batch_normalization_1_moving_meanIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9½
AssignVariableOp_9AssignVariableOp8assignvariableop_9_batch_normalization_1_moving_varianceIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10ª
AssignVariableOp_10AssignVariableOp"assignvariableop_10_dense_1_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11¨
AssignVariableOp_11AssignVariableOp assignvariableop_11_dense_1_biasIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12·
AssignVariableOp_12AssignVariableOp/assignvariableop_12_batch_normalization_2_gammaIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13¶
AssignVariableOp_13AssignVariableOp.assignvariableop_13_batch_normalization_2_betaIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14½
AssignVariableOp_14AssignVariableOp5assignvariableop_14_batch_normalization_2_moving_meanIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15Á
AssignVariableOp_15AssignVariableOp9assignvariableop_15_batch_normalization_2_moving_varianceIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16ª
AssignVariableOp_16AssignVariableOp"assignvariableop_16_dense_2_kernelIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17¨
AssignVariableOp_17AssignVariableOp assignvariableop_17_dense_2_biasIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18¢
AssignVariableOp_18AssignVariableOpassignvariableop_18_beta_1Identity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19¢
AssignVariableOp_19AssignVariableOpassignvariableop_19_beta_2Identity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20¡
AssignVariableOp_20AssignVariableOpassignvariableop_20_decayIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21©
AssignVariableOp_21AssignVariableOp!assignvariableop_21_learning_rateIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0	*
_output_shapes
:2
Identity_22¥
AssignVariableOp_22AssignVariableOpassignvariableop_22_adam_iterIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23¡
AssignVariableOp_23AssignVariableOpassignvariableop_23_totalIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24¡
AssignVariableOp_24AssignVariableOpassignvariableop_24_countIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25£
AssignVariableOp_25AssignVariableOpassignvariableop_25_total_1Identity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26£
AssignVariableOp_26AssignVariableOpassignvariableop_26_count_1Identity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27£
AssignVariableOp_27AssignVariableOpassignvariableop_27_total_2Identity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28£
AssignVariableOp_28AssignVariableOpassignvariableop_28_count_2Identity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29£
AssignVariableOp_29AssignVariableOpassignvariableop_29_total_3Identity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30£
AssignVariableOp_30AssignVariableOpassignvariableop_30_count_3Identity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31¼
AssignVariableOp_31AssignVariableOp4assignvariableop_31_adam_batch_normalization_gamma_mIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32»
AssignVariableOp_32AssignVariableOp3assignvariableop_32_adam_batch_normalization_beta_mIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33¯
AssignVariableOp_33AssignVariableOp'assignvariableop_33_adam_dense_kernel_mIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34­
AssignVariableOp_34AssignVariableOp%assignvariableop_34_adam_dense_bias_mIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35¾
AssignVariableOp_35AssignVariableOp6assignvariableop_35_adam_batch_normalization_1_gamma_mIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_36½
AssignVariableOp_36AssignVariableOp5assignvariableop_36_adam_batch_normalization_1_beta_mIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_36n
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:2
Identity_37±
AssignVariableOp_37AssignVariableOp)assignvariableop_37_adam_dense_1_kernel_mIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_37n
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:2
Identity_38¯
AssignVariableOp_38AssignVariableOp'assignvariableop_38_adam_dense_1_bias_mIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_38n
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:2
Identity_39¾
AssignVariableOp_39AssignVariableOp6assignvariableop_39_adam_batch_normalization_2_gamma_mIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_39n
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:2
Identity_40½
AssignVariableOp_40AssignVariableOp5assignvariableop_40_adam_batch_normalization_2_beta_mIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_40n
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:2
Identity_41±
AssignVariableOp_41AssignVariableOp)assignvariableop_41_adam_dense_2_kernel_mIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_41n
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:2
Identity_42¯
AssignVariableOp_42AssignVariableOp'assignvariableop_42_adam_dense_2_bias_mIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_42n
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:2
Identity_43¼
AssignVariableOp_43AssignVariableOp4assignvariableop_43_adam_batch_normalization_gamma_vIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_43n
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:2
Identity_44»
AssignVariableOp_44AssignVariableOp3assignvariableop_44_adam_batch_normalization_beta_vIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_44n
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:2
Identity_45¯
AssignVariableOp_45AssignVariableOp'assignvariableop_45_adam_dense_kernel_vIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_45n
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:2
Identity_46­
AssignVariableOp_46AssignVariableOp%assignvariableop_46_adam_dense_bias_vIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_46n
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:2
Identity_47¾
AssignVariableOp_47AssignVariableOp6assignvariableop_47_adam_batch_normalization_1_gamma_vIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_47n
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:2
Identity_48½
AssignVariableOp_48AssignVariableOp5assignvariableop_48_adam_batch_normalization_1_beta_vIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_48n
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:2
Identity_49±
AssignVariableOp_49AssignVariableOp)assignvariableop_49_adam_dense_1_kernel_vIdentity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_49n
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:2
Identity_50¯
AssignVariableOp_50AssignVariableOp'assignvariableop_50_adam_dense_1_bias_vIdentity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_50n
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:2
Identity_51¾
AssignVariableOp_51AssignVariableOp6assignvariableop_51_adam_batch_normalization_2_gamma_vIdentity_51:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_51n
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:2
Identity_52½
AssignVariableOp_52AssignVariableOp5assignvariableop_52_adam_batch_normalization_2_beta_vIdentity_52:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_52n
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:2
Identity_53±
AssignVariableOp_53AssignVariableOp)assignvariableop_53_adam_dense_2_kernel_vIdentity_53:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_53n
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:2
Identity_54¯
AssignVariableOp_54AssignVariableOp'assignvariableop_54_adam_dense_2_bias_vIdentity_54:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_549
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp

Identity_55Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_55f
Identity_56IdentityIdentity_55:output:0^NoOp_1*
T0*
_output_shapes
: 2
Identity_56

NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 2
NoOp_1"#
identity_56Identity_56:output:0*
_input_shapesr
p: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_44AssignVariableOp_442*
AssignVariableOp_45AssignVariableOp_452*
AssignVariableOp_46AssignVariableOp_462*
AssignVariableOp_47AssignVariableOp_472*
AssignVariableOp_48AssignVariableOp_482*
AssignVariableOp_49AssignVariableOp_492(
AssignVariableOp_5AssignVariableOp_52*
AssignVariableOp_50AssignVariableOp_502*
AssignVariableOp_51AssignVariableOp_512*
AssignVariableOp_52AssignVariableOp_522*
AssignVariableOp_53AssignVariableOp_532*
AssignVariableOp_54AssignVariableOp_542(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix

Á
'__inference_model_layer_call_fn_2279658

inputs
unknown:#
	unknown_0:#
	unknown_1:#
	unknown_2:#
	unknown_3:	#ú
	unknown_4:	ú
	unknown_5:	ú
	unknown_6:	ú
	unknown_7:	ú
	unknown_8:	ú
	unknown_9:
úú

unknown_10:	ú

unknown_11:	ú

unknown_12:	ú

unknown_13:	ú

unknown_14:	ú

unknown_15:	ú

unknown_16:
identity¢StatefulPartitionedCallÉ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*4
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_model_layer_call_and_return_conditional_losses_22789602
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:ÿÿÿÿÿÿÿÿÿ#: : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#
 
_user_specified_nameinputs
ñ

'__inference_dense_layer_call_fn_2279810

inputs
unknown:	#ú
	unknown_0:	ú
identity¢StatefulPartitionedCalló
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿú*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dense_layer_call_and_return_conditional_losses_22788702
StatefulPartitionedCall|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿú2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ#: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#
 
_user_specified_nameinputs
ã
c
G__inference_activation_layer_call_and_return_conditional_losses_2279815

inputs
identityO
ReluReluinputs*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿú2
Relug
IdentityIdentityRelu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿú2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:ÿÿÿÿÿÿÿÿÿú:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿú
 
_user_specified_nameinputs

Â
'__inference_model_layer_call_fn_2279224
input_1
unknown:#
	unknown_0:#
	unknown_1:#
	unknown_2:#
	unknown_3:	#ú
	unknown_4:	ú
	unknown_5:	ú
	unknown_6:	ú
	unknown_7:	ú
	unknown_8:	ú
	unknown_9:
úú

unknown_10:	ú

unknown_11:	ú

unknown_12:	ú

unknown_13:	ú

unknown_14:	ú

unknown_15:	ú

unknown_16:
identity¢StatefulPartitionedCallÄ
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_model_layer_call_and_return_conditional_losses_22791442
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:ÿÿÿÿÿÿÿÿÿ#: : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#
!
_user_specified_name	input_1

¡
B__inference_model_layer_call_and_return_conditional_losses_2279490

inputsC
5batch_normalization_batchnorm_readvariableop_resource:#G
9batch_normalization_batchnorm_mul_readvariableop_resource:#E
7batch_normalization_batchnorm_readvariableop_1_resource:#E
7batch_normalization_batchnorm_readvariableop_2_resource:#7
$dense_matmul_readvariableop_resource:	#ú4
%dense_biasadd_readvariableop_resource:	úF
7batch_normalization_1_batchnorm_readvariableop_resource:	úJ
;batch_normalization_1_batchnorm_mul_readvariableop_resource:	úH
9batch_normalization_1_batchnorm_readvariableop_1_resource:	úH
9batch_normalization_1_batchnorm_readvariableop_2_resource:	ú:
&dense_1_matmul_readvariableop_resource:
úú6
'dense_1_biasadd_readvariableop_resource:	úF
7batch_normalization_2_batchnorm_readvariableop_resource:	úJ
;batch_normalization_2_batchnorm_mul_readvariableop_resource:	úH
9batch_normalization_2_batchnorm_readvariableop_1_resource:	úH
9batch_normalization_2_batchnorm_readvariableop_2_resource:	ú9
&dense_2_matmul_readvariableop_resource:	ú5
'dense_2_biasadd_readvariableop_resource:
identity¢,batch_normalization/batchnorm/ReadVariableOp¢.batch_normalization/batchnorm/ReadVariableOp_1¢.batch_normalization/batchnorm/ReadVariableOp_2¢0batch_normalization/batchnorm/mul/ReadVariableOp¢.batch_normalization_1/batchnorm/ReadVariableOp¢0batch_normalization_1/batchnorm/ReadVariableOp_1¢0batch_normalization_1/batchnorm/ReadVariableOp_2¢2batch_normalization_1/batchnorm/mul/ReadVariableOp¢.batch_normalization_2/batchnorm/ReadVariableOp¢0batch_normalization_2/batchnorm/ReadVariableOp_1¢0batch_normalization_2/batchnorm/ReadVariableOp_2¢2batch_normalization_2/batchnorm/mul/ReadVariableOp¢dense/BiasAdd/ReadVariableOp¢dense/MatMul/ReadVariableOp¢.dense/kernel/Regularizer/Square/ReadVariableOp¢dense_1/BiasAdd/ReadVariableOp¢dense_1/MatMul/ReadVariableOp¢0dense_1/kernel/Regularizer/Square/ReadVariableOp¢dense_2/BiasAdd/ReadVariableOp¢dense_2/MatMul/ReadVariableOpÎ
,batch_normalization/batchnorm/ReadVariableOpReadVariableOp5batch_normalization_batchnorm_readvariableop_resource*
_output_shapes
:#*
dtype02.
,batch_normalization/batchnorm/ReadVariableOp
#batch_normalization/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2%
#batch_normalization/batchnorm/add/yØ
!batch_normalization/batchnorm/addAddV24batch_normalization/batchnorm/ReadVariableOp:value:0,batch_normalization/batchnorm/add/y:output:0*
T0*
_output_shapes
:#2#
!batch_normalization/batchnorm/add
#batch_normalization/batchnorm/RsqrtRsqrt%batch_normalization/batchnorm/add:z:0*
T0*
_output_shapes
:#2%
#batch_normalization/batchnorm/RsqrtÚ
0batch_normalization/batchnorm/mul/ReadVariableOpReadVariableOp9batch_normalization_batchnorm_mul_readvariableop_resource*
_output_shapes
:#*
dtype022
0batch_normalization/batchnorm/mul/ReadVariableOpÕ
!batch_normalization/batchnorm/mulMul'batch_normalization/batchnorm/Rsqrt:y:08batch_normalization/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:#2#
!batch_normalization/batchnorm/mul²
#batch_normalization/batchnorm/mul_1Mulinputs%batch_normalization/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#2%
#batch_normalization/batchnorm/mul_1Ô
.batch_normalization/batchnorm/ReadVariableOp_1ReadVariableOp7batch_normalization_batchnorm_readvariableop_1_resource*
_output_shapes
:#*
dtype020
.batch_normalization/batchnorm/ReadVariableOp_1Õ
#batch_normalization/batchnorm/mul_2Mul6batch_normalization/batchnorm/ReadVariableOp_1:value:0%batch_normalization/batchnorm/mul:z:0*
T0*
_output_shapes
:#2%
#batch_normalization/batchnorm/mul_2Ô
.batch_normalization/batchnorm/ReadVariableOp_2ReadVariableOp7batch_normalization_batchnorm_readvariableop_2_resource*
_output_shapes
:#*
dtype020
.batch_normalization/batchnorm/ReadVariableOp_2Ó
!batch_normalization/batchnorm/subSub6batch_normalization/batchnorm/ReadVariableOp_2:value:0'batch_normalization/batchnorm/mul_2:z:0*
T0*
_output_shapes
:#2#
!batch_normalization/batchnorm/subÕ
#batch_normalization/batchnorm/add_1AddV2'batch_normalization/batchnorm/mul_1:z:0%batch_normalization/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#2%
#batch_normalization/batchnorm/add_1 
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes
:	#ú*
dtype02
dense/MatMul/ReadVariableOp§
dense/MatMulMatMul'batch_normalization/batchnorm/add_1:z:0#dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿú2
dense/MatMul
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes	
:ú*
dtype02
dense/BiasAdd/ReadVariableOp
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿú2
dense/BiasAddu
activation/ReluReludense/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿú2
activation/ReluÕ
.batch_normalization_1/batchnorm/ReadVariableOpReadVariableOp7batch_normalization_1_batchnorm_readvariableop_resource*
_output_shapes	
:ú*
dtype020
.batch_normalization_1/batchnorm/ReadVariableOp
%batch_normalization_1/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2'
%batch_normalization_1/batchnorm/add/yá
#batch_normalization_1/batchnorm/addAddV26batch_normalization_1/batchnorm/ReadVariableOp:value:0.batch_normalization_1/batchnorm/add/y:output:0*
T0*
_output_shapes	
:ú2%
#batch_normalization_1/batchnorm/add¦
%batch_normalization_1/batchnorm/RsqrtRsqrt'batch_normalization_1/batchnorm/add:z:0*
T0*
_output_shapes	
:ú2'
%batch_normalization_1/batchnorm/Rsqrtá
2batch_normalization_1/batchnorm/mul/ReadVariableOpReadVariableOp;batch_normalization_1_batchnorm_mul_readvariableop_resource*
_output_shapes	
:ú*
dtype024
2batch_normalization_1/batchnorm/mul/ReadVariableOpÞ
#batch_normalization_1/batchnorm/mulMul)batch_normalization_1/batchnorm/Rsqrt:y:0:batch_normalization_1/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:ú2%
#batch_normalization_1/batchnorm/mulÐ
%batch_normalization_1/batchnorm/mul_1Mulactivation/Relu:activations:0'batch_normalization_1/batchnorm/mul:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿú2'
%batch_normalization_1/batchnorm/mul_1Û
0batch_normalization_1/batchnorm/ReadVariableOp_1ReadVariableOp9batch_normalization_1_batchnorm_readvariableop_1_resource*
_output_shapes	
:ú*
dtype022
0batch_normalization_1/batchnorm/ReadVariableOp_1Þ
%batch_normalization_1/batchnorm/mul_2Mul8batch_normalization_1/batchnorm/ReadVariableOp_1:value:0'batch_normalization_1/batchnorm/mul:z:0*
T0*
_output_shapes	
:ú2'
%batch_normalization_1/batchnorm/mul_2Û
0batch_normalization_1/batchnorm/ReadVariableOp_2ReadVariableOp9batch_normalization_1_batchnorm_readvariableop_2_resource*
_output_shapes	
:ú*
dtype022
0batch_normalization_1/batchnorm/ReadVariableOp_2Ü
#batch_normalization_1/batchnorm/subSub8batch_normalization_1/batchnorm/ReadVariableOp_2:value:0)batch_normalization_1/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:ú2%
#batch_normalization_1/batchnorm/subÞ
%batch_normalization_1/batchnorm/add_1AddV2)batch_normalization_1/batchnorm/mul_1:z:0'batch_normalization_1/batchnorm/sub:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿú2'
%batch_normalization_1/batchnorm/add_1§
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource* 
_output_shapes
:
úú*
dtype02
dense_1/MatMul/ReadVariableOp¯
dense_1/MatMulMatMul)batch_normalization_1/batchnorm/add_1:z:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿú2
dense_1/MatMul¥
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes	
:ú*
dtype02 
dense_1/BiasAdd/ReadVariableOp¢
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿú2
dense_1/BiasAdd{
activation_1/ReluReludense_1/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿú2
activation_1/ReluÕ
.batch_normalization_2/batchnorm/ReadVariableOpReadVariableOp7batch_normalization_2_batchnorm_readvariableop_resource*
_output_shapes	
:ú*
dtype020
.batch_normalization_2/batchnorm/ReadVariableOp
%batch_normalization_2/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2'
%batch_normalization_2/batchnorm/add/yá
#batch_normalization_2/batchnorm/addAddV26batch_normalization_2/batchnorm/ReadVariableOp:value:0.batch_normalization_2/batchnorm/add/y:output:0*
T0*
_output_shapes	
:ú2%
#batch_normalization_2/batchnorm/add¦
%batch_normalization_2/batchnorm/RsqrtRsqrt'batch_normalization_2/batchnorm/add:z:0*
T0*
_output_shapes	
:ú2'
%batch_normalization_2/batchnorm/Rsqrtá
2batch_normalization_2/batchnorm/mul/ReadVariableOpReadVariableOp;batch_normalization_2_batchnorm_mul_readvariableop_resource*
_output_shapes	
:ú*
dtype024
2batch_normalization_2/batchnorm/mul/ReadVariableOpÞ
#batch_normalization_2/batchnorm/mulMul)batch_normalization_2/batchnorm/Rsqrt:y:0:batch_normalization_2/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:ú2%
#batch_normalization_2/batchnorm/mulÒ
%batch_normalization_2/batchnorm/mul_1Mulactivation_1/Relu:activations:0'batch_normalization_2/batchnorm/mul:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿú2'
%batch_normalization_2/batchnorm/mul_1Û
0batch_normalization_2/batchnorm/ReadVariableOp_1ReadVariableOp9batch_normalization_2_batchnorm_readvariableop_1_resource*
_output_shapes	
:ú*
dtype022
0batch_normalization_2/batchnorm/ReadVariableOp_1Þ
%batch_normalization_2/batchnorm/mul_2Mul8batch_normalization_2/batchnorm/ReadVariableOp_1:value:0'batch_normalization_2/batchnorm/mul:z:0*
T0*
_output_shapes	
:ú2'
%batch_normalization_2/batchnorm/mul_2Û
0batch_normalization_2/batchnorm/ReadVariableOp_2ReadVariableOp9batch_normalization_2_batchnorm_readvariableop_2_resource*
_output_shapes	
:ú*
dtype022
0batch_normalization_2/batchnorm/ReadVariableOp_2Ü
#batch_normalization_2/batchnorm/subSub8batch_normalization_2/batchnorm/ReadVariableOp_2:value:0)batch_normalization_2/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:ú2%
#batch_normalization_2/batchnorm/subÞ
%batch_normalization_2/batchnorm/add_1AddV2)batch_normalization_2/batchnorm/mul_1:z:0'batch_normalization_2/batchnorm/sub:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿú2'
%batch_normalization_2/batchnorm/add_1¦
dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource*
_output_shapes
:	ú*
dtype02
dense_2/MatMul/ReadVariableOp®
dense_2/MatMulMatMul)batch_normalization_2/batchnorm/add_1:z:0%dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_2/MatMul¤
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_2/BiasAdd/ReadVariableOp¡
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_2/BiasAddy
dense_2/SigmoidSigmoiddense_2/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_2/SigmoidÆ
.dense/kernel/Regularizer/Square/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes
:	#ú*
dtype020
.dense/kernel/Regularizer/Square/ReadVariableOp®
dense/kernel/Regularizer/SquareSquare6dense/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	#ú2!
dense/kernel/Regularizer/Square
dense/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2 
dense/kernel/Regularizer/Const²
dense/kernel/Regularizer/SumSum#dense/kernel/Regularizer/Square:y:0'dense/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense/kernel/Regularizer/Sum
dense/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *3Yd<2 
dense/kernel/Regularizer/mul/x´
dense/kernel/Regularizer/mulMul'dense/kernel/Regularizer/mul/x:output:0%dense/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense/kernel/Regularizer/mulÍ
0dense_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource* 
_output_shapes
:
úú*
dtype022
0dense_1/kernel/Regularizer/Square/ReadVariableOpµ
!dense_1/kernel/Regularizer/SquareSquare8dense_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
úú2#
!dense_1/kernel/Regularizer/Square
 dense_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_1/kernel/Regularizer/Constº
dense_1/kernel/Regularizer/SumSum%dense_1/kernel/Regularizer/Square:y:0)dense_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/Sum
 dense_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *3Yd<2"
 dense_1/kernel/Regularizer/mul/x¼
dense_1/kernel/Regularizer/mulMul)dense_1/kernel/Regularizer/mul/x:output:0'dense_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/muln
IdentityIdentitydense_2/Sigmoid:y:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

IdentityÍ
NoOpNoOp-^batch_normalization/batchnorm/ReadVariableOp/^batch_normalization/batchnorm/ReadVariableOp_1/^batch_normalization/batchnorm/ReadVariableOp_21^batch_normalization/batchnorm/mul/ReadVariableOp/^batch_normalization_1/batchnorm/ReadVariableOp1^batch_normalization_1/batchnorm/ReadVariableOp_11^batch_normalization_1/batchnorm/ReadVariableOp_23^batch_normalization_1/batchnorm/mul/ReadVariableOp/^batch_normalization_2/batchnorm/ReadVariableOp1^batch_normalization_2/batchnorm/ReadVariableOp_11^batch_normalization_2/batchnorm/ReadVariableOp_23^batch_normalization_2/batchnorm/mul/ReadVariableOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp/^dense/kernel/Regularizer/Square/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp1^dense_1/kernel/Regularizer/Square/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:ÿÿÿÿÿÿÿÿÿ#: : : : : : : : : : : : : : : : : : 2\
,batch_normalization/batchnorm/ReadVariableOp,batch_normalization/batchnorm/ReadVariableOp2`
.batch_normalization/batchnorm/ReadVariableOp_1.batch_normalization/batchnorm/ReadVariableOp_12`
.batch_normalization/batchnorm/ReadVariableOp_2.batch_normalization/batchnorm/ReadVariableOp_22d
0batch_normalization/batchnorm/mul/ReadVariableOp0batch_normalization/batchnorm/mul/ReadVariableOp2`
.batch_normalization_1/batchnorm/ReadVariableOp.batch_normalization_1/batchnorm/ReadVariableOp2d
0batch_normalization_1/batchnorm/ReadVariableOp_10batch_normalization_1/batchnorm/ReadVariableOp_12d
0batch_normalization_1/batchnorm/ReadVariableOp_20batch_normalization_1/batchnorm/ReadVariableOp_22h
2batch_normalization_1/batchnorm/mul/ReadVariableOp2batch_normalization_1/batchnorm/mul/ReadVariableOp2`
.batch_normalization_2/batchnorm/ReadVariableOp.batch_normalization_2/batchnorm/ReadVariableOp2d
0batch_normalization_2/batchnorm/ReadVariableOp_10batch_normalization_2/batchnorm/ReadVariableOp_12d
0batch_normalization_2/batchnorm/ReadVariableOp_20batch_normalization_2/batchnorm/ReadVariableOp_22h
2batch_normalization_2/batchnorm/mul/ReadVariableOp2batch_normalization_2/batchnorm/mul/ReadVariableOp2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2`
.dense/kernel/Regularizer/Square/ReadVariableOp.dense/kernel/Regularizer/Square/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp2d
0dense_1/kernel/Regularizer/Square/ReadVariableOp0dense_1/kernel/Regularizer/Square/ReadVariableOp2@
dense_2/BiasAdd/ReadVariableOpdense_2/BiasAdd/ReadVariableOp2>
dense_2/MatMul/ReadVariableOpdense_2/MatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#
 
_user_specified_nameinputs
÷
ö
"__inference__wrapped_model_2278352
input_1I
;model_batch_normalization_batchnorm_readvariableop_resource:#M
?model_batch_normalization_batchnorm_mul_readvariableop_resource:#K
=model_batch_normalization_batchnorm_readvariableop_1_resource:#K
=model_batch_normalization_batchnorm_readvariableop_2_resource:#=
*model_dense_matmul_readvariableop_resource:	#ú:
+model_dense_biasadd_readvariableop_resource:	úL
=model_batch_normalization_1_batchnorm_readvariableop_resource:	úP
Amodel_batch_normalization_1_batchnorm_mul_readvariableop_resource:	úN
?model_batch_normalization_1_batchnorm_readvariableop_1_resource:	úN
?model_batch_normalization_1_batchnorm_readvariableop_2_resource:	ú@
,model_dense_1_matmul_readvariableop_resource:
úú<
-model_dense_1_biasadd_readvariableop_resource:	úL
=model_batch_normalization_2_batchnorm_readvariableop_resource:	úP
Amodel_batch_normalization_2_batchnorm_mul_readvariableop_resource:	úN
?model_batch_normalization_2_batchnorm_readvariableop_1_resource:	úN
?model_batch_normalization_2_batchnorm_readvariableop_2_resource:	ú?
,model_dense_2_matmul_readvariableop_resource:	ú;
-model_dense_2_biasadd_readvariableop_resource:
identity¢2model/batch_normalization/batchnorm/ReadVariableOp¢4model/batch_normalization/batchnorm/ReadVariableOp_1¢4model/batch_normalization/batchnorm/ReadVariableOp_2¢6model/batch_normalization/batchnorm/mul/ReadVariableOp¢4model/batch_normalization_1/batchnorm/ReadVariableOp¢6model/batch_normalization_1/batchnorm/ReadVariableOp_1¢6model/batch_normalization_1/batchnorm/ReadVariableOp_2¢8model/batch_normalization_1/batchnorm/mul/ReadVariableOp¢4model/batch_normalization_2/batchnorm/ReadVariableOp¢6model/batch_normalization_2/batchnorm/ReadVariableOp_1¢6model/batch_normalization_2/batchnorm/ReadVariableOp_2¢8model/batch_normalization_2/batchnorm/mul/ReadVariableOp¢"model/dense/BiasAdd/ReadVariableOp¢!model/dense/MatMul/ReadVariableOp¢$model/dense_1/BiasAdd/ReadVariableOp¢#model/dense_1/MatMul/ReadVariableOp¢$model/dense_2/BiasAdd/ReadVariableOp¢#model/dense_2/MatMul/ReadVariableOpà
2model/batch_normalization/batchnorm/ReadVariableOpReadVariableOp;model_batch_normalization_batchnorm_readvariableop_resource*
_output_shapes
:#*
dtype024
2model/batch_normalization/batchnorm/ReadVariableOp
)model/batch_normalization/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2+
)model/batch_normalization/batchnorm/add/yð
'model/batch_normalization/batchnorm/addAddV2:model/batch_normalization/batchnorm/ReadVariableOp:value:02model/batch_normalization/batchnorm/add/y:output:0*
T0*
_output_shapes
:#2)
'model/batch_normalization/batchnorm/add±
)model/batch_normalization/batchnorm/RsqrtRsqrt+model/batch_normalization/batchnorm/add:z:0*
T0*
_output_shapes
:#2+
)model/batch_normalization/batchnorm/Rsqrtì
6model/batch_normalization/batchnorm/mul/ReadVariableOpReadVariableOp?model_batch_normalization_batchnorm_mul_readvariableop_resource*
_output_shapes
:#*
dtype028
6model/batch_normalization/batchnorm/mul/ReadVariableOpí
'model/batch_normalization/batchnorm/mulMul-model/batch_normalization/batchnorm/Rsqrt:y:0>model/batch_normalization/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:#2)
'model/batch_normalization/batchnorm/mulÅ
)model/batch_normalization/batchnorm/mul_1Mulinput_1+model/batch_normalization/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#2+
)model/batch_normalization/batchnorm/mul_1æ
4model/batch_normalization/batchnorm/ReadVariableOp_1ReadVariableOp=model_batch_normalization_batchnorm_readvariableop_1_resource*
_output_shapes
:#*
dtype026
4model/batch_normalization/batchnorm/ReadVariableOp_1í
)model/batch_normalization/batchnorm/mul_2Mul<model/batch_normalization/batchnorm/ReadVariableOp_1:value:0+model/batch_normalization/batchnorm/mul:z:0*
T0*
_output_shapes
:#2+
)model/batch_normalization/batchnorm/mul_2æ
4model/batch_normalization/batchnorm/ReadVariableOp_2ReadVariableOp=model_batch_normalization_batchnorm_readvariableop_2_resource*
_output_shapes
:#*
dtype026
4model/batch_normalization/batchnorm/ReadVariableOp_2ë
'model/batch_normalization/batchnorm/subSub<model/batch_normalization/batchnorm/ReadVariableOp_2:value:0-model/batch_normalization/batchnorm/mul_2:z:0*
T0*
_output_shapes
:#2)
'model/batch_normalization/batchnorm/subí
)model/batch_normalization/batchnorm/add_1AddV2-model/batch_normalization/batchnorm/mul_1:z:0+model/batch_normalization/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#2+
)model/batch_normalization/batchnorm/add_1²
!model/dense/MatMul/ReadVariableOpReadVariableOp*model_dense_matmul_readvariableop_resource*
_output_shapes
:	#ú*
dtype02#
!model/dense/MatMul/ReadVariableOp¿
model/dense/MatMulMatMul-model/batch_normalization/batchnorm/add_1:z:0)model/dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿú2
model/dense/MatMul±
"model/dense/BiasAdd/ReadVariableOpReadVariableOp+model_dense_biasadd_readvariableop_resource*
_output_shapes	
:ú*
dtype02$
"model/dense/BiasAdd/ReadVariableOp²
model/dense/BiasAddBiasAddmodel/dense/MatMul:product:0*model/dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿú2
model/dense/BiasAdd
model/activation/ReluRelumodel/dense/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿú2
model/activation/Reluç
4model/batch_normalization_1/batchnorm/ReadVariableOpReadVariableOp=model_batch_normalization_1_batchnorm_readvariableop_resource*
_output_shapes	
:ú*
dtype026
4model/batch_normalization_1/batchnorm/ReadVariableOp
+model/batch_normalization_1/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2-
+model/batch_normalization_1/batchnorm/add/yù
)model/batch_normalization_1/batchnorm/addAddV2<model/batch_normalization_1/batchnorm/ReadVariableOp:value:04model/batch_normalization_1/batchnorm/add/y:output:0*
T0*
_output_shapes	
:ú2+
)model/batch_normalization_1/batchnorm/add¸
+model/batch_normalization_1/batchnorm/RsqrtRsqrt-model/batch_normalization_1/batchnorm/add:z:0*
T0*
_output_shapes	
:ú2-
+model/batch_normalization_1/batchnorm/Rsqrtó
8model/batch_normalization_1/batchnorm/mul/ReadVariableOpReadVariableOpAmodel_batch_normalization_1_batchnorm_mul_readvariableop_resource*
_output_shapes	
:ú*
dtype02:
8model/batch_normalization_1/batchnorm/mul/ReadVariableOpö
)model/batch_normalization_1/batchnorm/mulMul/model/batch_normalization_1/batchnorm/Rsqrt:y:0@model/batch_normalization_1/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:ú2+
)model/batch_normalization_1/batchnorm/mulè
+model/batch_normalization_1/batchnorm/mul_1Mul#model/activation/Relu:activations:0-model/batch_normalization_1/batchnorm/mul:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿú2-
+model/batch_normalization_1/batchnorm/mul_1í
6model/batch_normalization_1/batchnorm/ReadVariableOp_1ReadVariableOp?model_batch_normalization_1_batchnorm_readvariableop_1_resource*
_output_shapes	
:ú*
dtype028
6model/batch_normalization_1/batchnorm/ReadVariableOp_1ö
+model/batch_normalization_1/batchnorm/mul_2Mul>model/batch_normalization_1/batchnorm/ReadVariableOp_1:value:0-model/batch_normalization_1/batchnorm/mul:z:0*
T0*
_output_shapes	
:ú2-
+model/batch_normalization_1/batchnorm/mul_2í
6model/batch_normalization_1/batchnorm/ReadVariableOp_2ReadVariableOp?model_batch_normalization_1_batchnorm_readvariableop_2_resource*
_output_shapes	
:ú*
dtype028
6model/batch_normalization_1/batchnorm/ReadVariableOp_2ô
)model/batch_normalization_1/batchnorm/subSub>model/batch_normalization_1/batchnorm/ReadVariableOp_2:value:0/model/batch_normalization_1/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:ú2+
)model/batch_normalization_1/batchnorm/subö
+model/batch_normalization_1/batchnorm/add_1AddV2/model/batch_normalization_1/batchnorm/mul_1:z:0-model/batch_normalization_1/batchnorm/sub:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿú2-
+model/batch_normalization_1/batchnorm/add_1¹
#model/dense_1/MatMul/ReadVariableOpReadVariableOp,model_dense_1_matmul_readvariableop_resource* 
_output_shapes
:
úú*
dtype02%
#model/dense_1/MatMul/ReadVariableOpÇ
model/dense_1/MatMulMatMul/model/batch_normalization_1/batchnorm/add_1:z:0+model/dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿú2
model/dense_1/MatMul·
$model/dense_1/BiasAdd/ReadVariableOpReadVariableOp-model_dense_1_biasadd_readvariableop_resource*
_output_shapes	
:ú*
dtype02&
$model/dense_1/BiasAdd/ReadVariableOpº
model/dense_1/BiasAddBiasAddmodel/dense_1/MatMul:product:0,model/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿú2
model/dense_1/BiasAdd
model/activation_1/ReluRelumodel/dense_1/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿú2
model/activation_1/Reluç
4model/batch_normalization_2/batchnorm/ReadVariableOpReadVariableOp=model_batch_normalization_2_batchnorm_readvariableop_resource*
_output_shapes	
:ú*
dtype026
4model/batch_normalization_2/batchnorm/ReadVariableOp
+model/batch_normalization_2/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2-
+model/batch_normalization_2/batchnorm/add/yù
)model/batch_normalization_2/batchnorm/addAddV2<model/batch_normalization_2/batchnorm/ReadVariableOp:value:04model/batch_normalization_2/batchnorm/add/y:output:0*
T0*
_output_shapes	
:ú2+
)model/batch_normalization_2/batchnorm/add¸
+model/batch_normalization_2/batchnorm/RsqrtRsqrt-model/batch_normalization_2/batchnorm/add:z:0*
T0*
_output_shapes	
:ú2-
+model/batch_normalization_2/batchnorm/Rsqrtó
8model/batch_normalization_2/batchnorm/mul/ReadVariableOpReadVariableOpAmodel_batch_normalization_2_batchnorm_mul_readvariableop_resource*
_output_shapes	
:ú*
dtype02:
8model/batch_normalization_2/batchnorm/mul/ReadVariableOpö
)model/batch_normalization_2/batchnorm/mulMul/model/batch_normalization_2/batchnorm/Rsqrt:y:0@model/batch_normalization_2/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:ú2+
)model/batch_normalization_2/batchnorm/mulê
+model/batch_normalization_2/batchnorm/mul_1Mul%model/activation_1/Relu:activations:0-model/batch_normalization_2/batchnorm/mul:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿú2-
+model/batch_normalization_2/batchnorm/mul_1í
6model/batch_normalization_2/batchnorm/ReadVariableOp_1ReadVariableOp?model_batch_normalization_2_batchnorm_readvariableop_1_resource*
_output_shapes	
:ú*
dtype028
6model/batch_normalization_2/batchnorm/ReadVariableOp_1ö
+model/batch_normalization_2/batchnorm/mul_2Mul>model/batch_normalization_2/batchnorm/ReadVariableOp_1:value:0-model/batch_normalization_2/batchnorm/mul:z:0*
T0*
_output_shapes	
:ú2-
+model/batch_normalization_2/batchnorm/mul_2í
6model/batch_normalization_2/batchnorm/ReadVariableOp_2ReadVariableOp?model_batch_normalization_2_batchnorm_readvariableop_2_resource*
_output_shapes	
:ú*
dtype028
6model/batch_normalization_2/batchnorm/ReadVariableOp_2ô
)model/batch_normalization_2/batchnorm/subSub>model/batch_normalization_2/batchnorm/ReadVariableOp_2:value:0/model/batch_normalization_2/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:ú2+
)model/batch_normalization_2/batchnorm/subö
+model/batch_normalization_2/batchnorm/add_1AddV2/model/batch_normalization_2/batchnorm/mul_1:z:0-model/batch_normalization_2/batchnorm/sub:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿú2-
+model/batch_normalization_2/batchnorm/add_1¸
#model/dense_2/MatMul/ReadVariableOpReadVariableOp,model_dense_2_matmul_readvariableop_resource*
_output_shapes
:	ú*
dtype02%
#model/dense_2/MatMul/ReadVariableOpÆ
model/dense_2/MatMulMatMul/model/batch_normalization_2/batchnorm/add_1:z:0+model/dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
model/dense_2/MatMul¶
$model/dense_2/BiasAdd/ReadVariableOpReadVariableOp-model_dense_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02&
$model/dense_2/BiasAdd/ReadVariableOp¹
model/dense_2/BiasAddBiasAddmodel/dense_2/MatMul:product:0,model/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
model/dense_2/BiasAdd
model/dense_2/SigmoidSigmoidmodel/dense_2/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
model/dense_2/Sigmoidt
IdentityIdentitymodel/dense_2/Sigmoid:y:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

IdentityÕ
NoOpNoOp3^model/batch_normalization/batchnorm/ReadVariableOp5^model/batch_normalization/batchnorm/ReadVariableOp_15^model/batch_normalization/batchnorm/ReadVariableOp_27^model/batch_normalization/batchnorm/mul/ReadVariableOp5^model/batch_normalization_1/batchnorm/ReadVariableOp7^model/batch_normalization_1/batchnorm/ReadVariableOp_17^model/batch_normalization_1/batchnorm/ReadVariableOp_29^model/batch_normalization_1/batchnorm/mul/ReadVariableOp5^model/batch_normalization_2/batchnorm/ReadVariableOp7^model/batch_normalization_2/batchnorm/ReadVariableOp_17^model/batch_normalization_2/batchnorm/ReadVariableOp_29^model/batch_normalization_2/batchnorm/mul/ReadVariableOp#^model/dense/BiasAdd/ReadVariableOp"^model/dense/MatMul/ReadVariableOp%^model/dense_1/BiasAdd/ReadVariableOp$^model/dense_1/MatMul/ReadVariableOp%^model/dense_2/BiasAdd/ReadVariableOp$^model/dense_2/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:ÿÿÿÿÿÿÿÿÿ#: : : : : : : : : : : : : : : : : : 2h
2model/batch_normalization/batchnorm/ReadVariableOp2model/batch_normalization/batchnorm/ReadVariableOp2l
4model/batch_normalization/batchnorm/ReadVariableOp_14model/batch_normalization/batchnorm/ReadVariableOp_12l
4model/batch_normalization/batchnorm/ReadVariableOp_24model/batch_normalization/batchnorm/ReadVariableOp_22p
6model/batch_normalization/batchnorm/mul/ReadVariableOp6model/batch_normalization/batchnorm/mul/ReadVariableOp2l
4model/batch_normalization_1/batchnorm/ReadVariableOp4model/batch_normalization_1/batchnorm/ReadVariableOp2p
6model/batch_normalization_1/batchnorm/ReadVariableOp_16model/batch_normalization_1/batchnorm/ReadVariableOp_12p
6model/batch_normalization_1/batchnorm/ReadVariableOp_26model/batch_normalization_1/batchnorm/ReadVariableOp_22t
8model/batch_normalization_1/batchnorm/mul/ReadVariableOp8model/batch_normalization_1/batchnorm/mul/ReadVariableOp2l
4model/batch_normalization_2/batchnorm/ReadVariableOp4model/batch_normalization_2/batchnorm/ReadVariableOp2p
6model/batch_normalization_2/batchnorm/ReadVariableOp_16model/batch_normalization_2/batchnorm/ReadVariableOp_12p
6model/batch_normalization_2/batchnorm/ReadVariableOp_26model/batch_normalization_2/batchnorm/ReadVariableOp_22t
8model/batch_normalization_2/batchnorm/mul/ReadVariableOp8model/batch_normalization_2/batchnorm/mul/ReadVariableOp2H
"model/dense/BiasAdd/ReadVariableOp"model/dense/BiasAdd/ReadVariableOp2F
!model/dense/MatMul/ReadVariableOp!model/dense/MatMul/ReadVariableOp2L
$model/dense_1/BiasAdd/ReadVariableOp$model/dense_1/BiasAdd/ReadVariableOp2J
#model/dense_1/MatMul/ReadVariableOp#model/dense_1/MatMul/ReadVariableOp2L
$model/dense_2/BiasAdd/ReadVariableOp$model/dense_2/BiasAdd/ReadVariableOp2J
#model/dense_2/MatMul/ReadVariableOp#model/dense_2/MatMul/ReadVariableOp:P L
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#
!
_user_specified_name	input_1
é*
ï
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_2278760

inputs6
'assignmovingavg_readvariableop_resource:	ú8
)assignmovingavg_1_readvariableop_resource:	ú4
%batchnorm_mul_readvariableop_resource:	ú0
!batchnorm_readvariableop_resource:	ú
identity¢AssignMovingAvg¢AssignMovingAvg/ReadVariableOp¢AssignMovingAvg_1¢ AssignMovingAvg_1/ReadVariableOp¢batchnorm/ReadVariableOp¢batchnorm/mul/ReadVariableOp
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2 
moments/mean/reduction_indices
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	ú*
	keep_dims(2
moments/mean}
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes
:	ú2
moments/StopGradient¥
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿú2
moments/SquaredDifference
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2$
"moments/variance/reduction_indices³
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	ú*
	keep_dims(2
moments/variance
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes	
:ú*
squeeze_dims
 2
moments/Squeeze
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes	
:ú*
squeeze_dims
 2
moments/Squeeze_1s
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2
AssignMovingAvg/decay¥
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes	
:ú*
dtype02 
AssignMovingAvg/ReadVariableOp
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes	
:ú2
AssignMovingAvg/sub
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes	
:ú2
AssignMovingAvg/mul¿
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype02
AssignMovingAvgw
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2
AssignMovingAvg_1/decay«
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes	
:ú*
dtype02"
 AssignMovingAvg_1/ReadVariableOp¡
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes	
:ú2
AssignMovingAvg_1/sub
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes	
:ú2
AssignMovingAvg_1/mulÉ
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02
AssignMovingAvg_1g
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2
batchnorm/add/y
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:ú2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:ú2
batchnorm/Rsqrt
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:ú*
dtype02
batchnorm/mul/ReadVariableOp
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:ú2
batchnorm/mulw
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿú2
batchnorm/mul_1|
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes	
:ú2
batchnorm/mul_2
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:ú*
dtype02
batchnorm/ReadVariableOp
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:ú2
batchnorm/sub
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿú2
batchnorm/add_1o
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿú2

Identityò
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿú: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿú
 
_user_specified_nameinputs
ã
c
G__inference_activation_layer_call_and_return_conditional_losses_2278881

inputs
identityO
ReluReluinputs*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿú2
Relug
IdentityIdentityRelu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿú2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:ÿÿÿÿÿÿÿÿÿú:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿú
 
_user_specified_nameinputs
Ó
Ð
5__inference_batch_normalization_layer_call_fn_2279779

inputs
unknown:#
	unknown_0:#
	unknown_1:#
	unknown_2:#
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Y
fTRR
P__inference_batch_normalization_layer_call_and_return_conditional_losses_22784362
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ#: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#
 
_user_specified_nameinputs

Â
'__inference_model_layer_call_fn_2278999
input_1
unknown:#
	unknown_0:#
	unknown_1:#
	unknown_2:#
	unknown_3:	#ú
	unknown_4:	ú
	unknown_5:	ú
	unknown_6:	ú
	unknown_7:	ú
	unknown_8:	ú
	unknown_9:
úú

unknown_10:	ú

unknown_11:	ú

unknown_12:	ú

unknown_13:	ú

unknown_14:	ú

unknown_15:	ú

unknown_16:
identity¢StatefulPartitionedCallÊ
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*4
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_model_layer_call_and_return_conditional_losses_22789602
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:ÿÿÿÿÿÿÿÿÿ#: : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#
!
_user_specified_name	input_1

µ
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_2278538

inputs0
!batchnorm_readvariableop_resource:	ú4
%batchnorm_mul_readvariableop_resource:	ú2
#batchnorm_readvariableop_1_resource:	ú2
#batchnorm_readvariableop_2_resource:	ú
identity¢batchnorm/ReadVariableOp¢batchnorm/ReadVariableOp_1¢batchnorm/ReadVariableOp_2¢batchnorm/mul/ReadVariableOp
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:ú*
dtype02
batchnorm/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2
batchnorm/add/y
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:ú2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:ú2
batchnorm/Rsqrt
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:ú*
dtype02
batchnorm/mul/ReadVariableOp
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:ú2
batchnorm/mulw
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿú2
batchnorm/mul_1
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes	
:ú*
dtype02
batchnorm/ReadVariableOp_1
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes	
:ú2
batchnorm/mul_2
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes	
:ú*
dtype02
batchnorm/ReadVariableOp_2
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:ú2
batchnorm/sub
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿú2
batchnorm/add_1o
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿú2

IdentityÂ
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿú: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿú
 
_user_specified_nameinputs
àä
Ï
B__inference_model_layer_call_and_return_conditional_losses_2279617

inputsI
;batch_normalization_assignmovingavg_readvariableop_resource:#K
=batch_normalization_assignmovingavg_1_readvariableop_resource:#G
9batch_normalization_batchnorm_mul_readvariableop_resource:#C
5batch_normalization_batchnorm_readvariableop_resource:#7
$dense_matmul_readvariableop_resource:	#ú4
%dense_biasadd_readvariableop_resource:	úL
=batch_normalization_1_assignmovingavg_readvariableop_resource:	úN
?batch_normalization_1_assignmovingavg_1_readvariableop_resource:	úJ
;batch_normalization_1_batchnorm_mul_readvariableop_resource:	úF
7batch_normalization_1_batchnorm_readvariableop_resource:	ú:
&dense_1_matmul_readvariableop_resource:
úú6
'dense_1_biasadd_readvariableop_resource:	úL
=batch_normalization_2_assignmovingavg_readvariableop_resource:	úN
?batch_normalization_2_assignmovingavg_1_readvariableop_resource:	úJ
;batch_normalization_2_batchnorm_mul_readvariableop_resource:	úF
7batch_normalization_2_batchnorm_readvariableop_resource:	ú9
&dense_2_matmul_readvariableop_resource:	ú5
'dense_2_biasadd_readvariableop_resource:
identity¢#batch_normalization/AssignMovingAvg¢2batch_normalization/AssignMovingAvg/ReadVariableOp¢%batch_normalization/AssignMovingAvg_1¢4batch_normalization/AssignMovingAvg_1/ReadVariableOp¢,batch_normalization/batchnorm/ReadVariableOp¢0batch_normalization/batchnorm/mul/ReadVariableOp¢%batch_normalization_1/AssignMovingAvg¢4batch_normalization_1/AssignMovingAvg/ReadVariableOp¢'batch_normalization_1/AssignMovingAvg_1¢6batch_normalization_1/AssignMovingAvg_1/ReadVariableOp¢.batch_normalization_1/batchnorm/ReadVariableOp¢2batch_normalization_1/batchnorm/mul/ReadVariableOp¢%batch_normalization_2/AssignMovingAvg¢4batch_normalization_2/AssignMovingAvg/ReadVariableOp¢'batch_normalization_2/AssignMovingAvg_1¢6batch_normalization_2/AssignMovingAvg_1/ReadVariableOp¢.batch_normalization_2/batchnorm/ReadVariableOp¢2batch_normalization_2/batchnorm/mul/ReadVariableOp¢dense/BiasAdd/ReadVariableOp¢dense/MatMul/ReadVariableOp¢.dense/kernel/Regularizer/Square/ReadVariableOp¢dense_1/BiasAdd/ReadVariableOp¢dense_1/MatMul/ReadVariableOp¢0dense_1/kernel/Regularizer/Square/ReadVariableOp¢dense_2/BiasAdd/ReadVariableOp¢dense_2/MatMul/ReadVariableOp²
2batch_normalization/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 24
2batch_normalization/moments/mean/reduction_indicesË
 batch_normalization/moments/meanMeaninputs;batch_normalization/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:#*
	keep_dims(2"
 batch_normalization/moments/mean¸
(batch_normalization/moments/StopGradientStopGradient)batch_normalization/moments/mean:output:0*
T0*
_output_shapes

:#2*
(batch_normalization/moments/StopGradientà
-batch_normalization/moments/SquaredDifferenceSquaredDifferenceinputs1batch_normalization/moments/StopGradient:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#2/
-batch_normalization/moments/SquaredDifferenceº
6batch_normalization/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 28
6batch_normalization/moments/variance/reduction_indices
$batch_normalization/moments/varianceMean1batch_normalization/moments/SquaredDifference:z:0?batch_normalization/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:#*
	keep_dims(2&
$batch_normalization/moments/variance¼
#batch_normalization/moments/SqueezeSqueeze)batch_normalization/moments/mean:output:0*
T0*
_output_shapes
:#*
squeeze_dims
 2%
#batch_normalization/moments/SqueezeÄ
%batch_normalization/moments/Squeeze_1Squeeze-batch_normalization/moments/variance:output:0*
T0*
_output_shapes
:#*
squeeze_dims
 2'
%batch_normalization/moments/Squeeze_1
)batch_normalization/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2+
)batch_normalization/AssignMovingAvg/decayà
2batch_normalization/AssignMovingAvg/ReadVariableOpReadVariableOp;batch_normalization_assignmovingavg_readvariableop_resource*
_output_shapes
:#*
dtype024
2batch_normalization/AssignMovingAvg/ReadVariableOpè
'batch_normalization/AssignMovingAvg/subSub:batch_normalization/AssignMovingAvg/ReadVariableOp:value:0,batch_normalization/moments/Squeeze:output:0*
T0*
_output_shapes
:#2)
'batch_normalization/AssignMovingAvg/subß
'batch_normalization/AssignMovingAvg/mulMul+batch_normalization/AssignMovingAvg/sub:z:02batch_normalization/AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:#2)
'batch_normalization/AssignMovingAvg/mul£
#batch_normalization/AssignMovingAvgAssignSubVariableOp;batch_normalization_assignmovingavg_readvariableop_resource+batch_normalization/AssignMovingAvg/mul:z:03^batch_normalization/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype02%
#batch_normalization/AssignMovingAvg
+batch_normalization/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2-
+batch_normalization/AssignMovingAvg_1/decayæ
4batch_normalization/AssignMovingAvg_1/ReadVariableOpReadVariableOp=batch_normalization_assignmovingavg_1_readvariableop_resource*
_output_shapes
:#*
dtype026
4batch_normalization/AssignMovingAvg_1/ReadVariableOpð
)batch_normalization/AssignMovingAvg_1/subSub<batch_normalization/AssignMovingAvg_1/ReadVariableOp:value:0.batch_normalization/moments/Squeeze_1:output:0*
T0*
_output_shapes
:#2+
)batch_normalization/AssignMovingAvg_1/subç
)batch_normalization/AssignMovingAvg_1/mulMul-batch_normalization/AssignMovingAvg_1/sub:z:04batch_normalization/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:#2+
)batch_normalization/AssignMovingAvg_1/mul­
%batch_normalization/AssignMovingAvg_1AssignSubVariableOp=batch_normalization_assignmovingavg_1_readvariableop_resource-batch_normalization/AssignMovingAvg_1/mul:z:05^batch_normalization/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02'
%batch_normalization/AssignMovingAvg_1
#batch_normalization/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2%
#batch_normalization/batchnorm/add/yÒ
!batch_normalization/batchnorm/addAddV2.batch_normalization/moments/Squeeze_1:output:0,batch_normalization/batchnorm/add/y:output:0*
T0*
_output_shapes
:#2#
!batch_normalization/batchnorm/add
#batch_normalization/batchnorm/RsqrtRsqrt%batch_normalization/batchnorm/add:z:0*
T0*
_output_shapes
:#2%
#batch_normalization/batchnorm/RsqrtÚ
0batch_normalization/batchnorm/mul/ReadVariableOpReadVariableOp9batch_normalization_batchnorm_mul_readvariableop_resource*
_output_shapes
:#*
dtype022
0batch_normalization/batchnorm/mul/ReadVariableOpÕ
!batch_normalization/batchnorm/mulMul'batch_normalization/batchnorm/Rsqrt:y:08batch_normalization/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:#2#
!batch_normalization/batchnorm/mul²
#batch_normalization/batchnorm/mul_1Mulinputs%batch_normalization/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#2%
#batch_normalization/batchnorm/mul_1Ë
#batch_normalization/batchnorm/mul_2Mul,batch_normalization/moments/Squeeze:output:0%batch_normalization/batchnorm/mul:z:0*
T0*
_output_shapes
:#2%
#batch_normalization/batchnorm/mul_2Î
,batch_normalization/batchnorm/ReadVariableOpReadVariableOp5batch_normalization_batchnorm_readvariableop_resource*
_output_shapes
:#*
dtype02.
,batch_normalization/batchnorm/ReadVariableOpÑ
!batch_normalization/batchnorm/subSub4batch_normalization/batchnorm/ReadVariableOp:value:0'batch_normalization/batchnorm/mul_2:z:0*
T0*
_output_shapes
:#2#
!batch_normalization/batchnorm/subÕ
#batch_normalization/batchnorm/add_1AddV2'batch_normalization/batchnorm/mul_1:z:0%batch_normalization/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#2%
#batch_normalization/batchnorm/add_1 
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes
:	#ú*
dtype02
dense/MatMul/ReadVariableOp§
dense/MatMulMatMul'batch_normalization/batchnorm/add_1:z:0#dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿú2
dense/MatMul
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes	
:ú*
dtype02
dense/BiasAdd/ReadVariableOp
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿú2
dense/BiasAddu
activation/ReluReludense/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿú2
activation/Relu¶
4batch_normalization_1/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 26
4batch_normalization_1/moments/mean/reduction_indicesé
"batch_normalization_1/moments/meanMeanactivation/Relu:activations:0=batch_normalization_1/moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	ú*
	keep_dims(2$
"batch_normalization_1/moments/mean¿
*batch_normalization_1/moments/StopGradientStopGradient+batch_normalization_1/moments/mean:output:0*
T0*
_output_shapes
:	ú2,
*batch_normalization_1/moments/StopGradientþ
/batch_normalization_1/moments/SquaredDifferenceSquaredDifferenceactivation/Relu:activations:03batch_normalization_1/moments/StopGradient:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿú21
/batch_normalization_1/moments/SquaredDifference¾
8batch_normalization_1/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2:
8batch_normalization_1/moments/variance/reduction_indices
&batch_normalization_1/moments/varianceMean3batch_normalization_1/moments/SquaredDifference:z:0Abatch_normalization_1/moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	ú*
	keep_dims(2(
&batch_normalization_1/moments/varianceÃ
%batch_normalization_1/moments/SqueezeSqueeze+batch_normalization_1/moments/mean:output:0*
T0*
_output_shapes	
:ú*
squeeze_dims
 2'
%batch_normalization_1/moments/SqueezeË
'batch_normalization_1/moments/Squeeze_1Squeeze/batch_normalization_1/moments/variance:output:0*
T0*
_output_shapes	
:ú*
squeeze_dims
 2)
'batch_normalization_1/moments/Squeeze_1
+batch_normalization_1/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2-
+batch_normalization_1/AssignMovingAvg/decayç
4batch_normalization_1/AssignMovingAvg/ReadVariableOpReadVariableOp=batch_normalization_1_assignmovingavg_readvariableop_resource*
_output_shapes	
:ú*
dtype026
4batch_normalization_1/AssignMovingAvg/ReadVariableOpñ
)batch_normalization_1/AssignMovingAvg/subSub<batch_normalization_1/AssignMovingAvg/ReadVariableOp:value:0.batch_normalization_1/moments/Squeeze:output:0*
T0*
_output_shapes	
:ú2+
)batch_normalization_1/AssignMovingAvg/subè
)batch_normalization_1/AssignMovingAvg/mulMul-batch_normalization_1/AssignMovingAvg/sub:z:04batch_normalization_1/AssignMovingAvg/decay:output:0*
T0*
_output_shapes	
:ú2+
)batch_normalization_1/AssignMovingAvg/mul­
%batch_normalization_1/AssignMovingAvgAssignSubVariableOp=batch_normalization_1_assignmovingavg_readvariableop_resource-batch_normalization_1/AssignMovingAvg/mul:z:05^batch_normalization_1/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype02'
%batch_normalization_1/AssignMovingAvg£
-batch_normalization_1/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2/
-batch_normalization_1/AssignMovingAvg_1/decayí
6batch_normalization_1/AssignMovingAvg_1/ReadVariableOpReadVariableOp?batch_normalization_1_assignmovingavg_1_readvariableop_resource*
_output_shapes	
:ú*
dtype028
6batch_normalization_1/AssignMovingAvg_1/ReadVariableOpù
+batch_normalization_1/AssignMovingAvg_1/subSub>batch_normalization_1/AssignMovingAvg_1/ReadVariableOp:value:00batch_normalization_1/moments/Squeeze_1:output:0*
T0*
_output_shapes	
:ú2-
+batch_normalization_1/AssignMovingAvg_1/subð
+batch_normalization_1/AssignMovingAvg_1/mulMul/batch_normalization_1/AssignMovingAvg_1/sub:z:06batch_normalization_1/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes	
:ú2-
+batch_normalization_1/AssignMovingAvg_1/mul·
'batch_normalization_1/AssignMovingAvg_1AssignSubVariableOp?batch_normalization_1_assignmovingavg_1_readvariableop_resource/batch_normalization_1/AssignMovingAvg_1/mul:z:07^batch_normalization_1/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02)
'batch_normalization_1/AssignMovingAvg_1
%batch_normalization_1/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2'
%batch_normalization_1/batchnorm/add/yÛ
#batch_normalization_1/batchnorm/addAddV20batch_normalization_1/moments/Squeeze_1:output:0.batch_normalization_1/batchnorm/add/y:output:0*
T0*
_output_shapes	
:ú2%
#batch_normalization_1/batchnorm/add¦
%batch_normalization_1/batchnorm/RsqrtRsqrt'batch_normalization_1/batchnorm/add:z:0*
T0*
_output_shapes	
:ú2'
%batch_normalization_1/batchnorm/Rsqrtá
2batch_normalization_1/batchnorm/mul/ReadVariableOpReadVariableOp;batch_normalization_1_batchnorm_mul_readvariableop_resource*
_output_shapes	
:ú*
dtype024
2batch_normalization_1/batchnorm/mul/ReadVariableOpÞ
#batch_normalization_1/batchnorm/mulMul)batch_normalization_1/batchnorm/Rsqrt:y:0:batch_normalization_1/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:ú2%
#batch_normalization_1/batchnorm/mulÐ
%batch_normalization_1/batchnorm/mul_1Mulactivation/Relu:activations:0'batch_normalization_1/batchnorm/mul:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿú2'
%batch_normalization_1/batchnorm/mul_1Ô
%batch_normalization_1/batchnorm/mul_2Mul.batch_normalization_1/moments/Squeeze:output:0'batch_normalization_1/batchnorm/mul:z:0*
T0*
_output_shapes	
:ú2'
%batch_normalization_1/batchnorm/mul_2Õ
.batch_normalization_1/batchnorm/ReadVariableOpReadVariableOp7batch_normalization_1_batchnorm_readvariableop_resource*
_output_shapes	
:ú*
dtype020
.batch_normalization_1/batchnorm/ReadVariableOpÚ
#batch_normalization_1/batchnorm/subSub6batch_normalization_1/batchnorm/ReadVariableOp:value:0)batch_normalization_1/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:ú2%
#batch_normalization_1/batchnorm/subÞ
%batch_normalization_1/batchnorm/add_1AddV2)batch_normalization_1/batchnorm/mul_1:z:0'batch_normalization_1/batchnorm/sub:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿú2'
%batch_normalization_1/batchnorm/add_1§
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource* 
_output_shapes
:
úú*
dtype02
dense_1/MatMul/ReadVariableOp¯
dense_1/MatMulMatMul)batch_normalization_1/batchnorm/add_1:z:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿú2
dense_1/MatMul¥
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes	
:ú*
dtype02 
dense_1/BiasAdd/ReadVariableOp¢
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿú2
dense_1/BiasAdd{
activation_1/ReluReludense_1/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿú2
activation_1/Relu¶
4batch_normalization_2/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 26
4batch_normalization_2/moments/mean/reduction_indicesë
"batch_normalization_2/moments/meanMeanactivation_1/Relu:activations:0=batch_normalization_2/moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	ú*
	keep_dims(2$
"batch_normalization_2/moments/mean¿
*batch_normalization_2/moments/StopGradientStopGradient+batch_normalization_2/moments/mean:output:0*
T0*
_output_shapes
:	ú2,
*batch_normalization_2/moments/StopGradient
/batch_normalization_2/moments/SquaredDifferenceSquaredDifferenceactivation_1/Relu:activations:03batch_normalization_2/moments/StopGradient:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿú21
/batch_normalization_2/moments/SquaredDifference¾
8batch_normalization_2/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2:
8batch_normalization_2/moments/variance/reduction_indices
&batch_normalization_2/moments/varianceMean3batch_normalization_2/moments/SquaredDifference:z:0Abatch_normalization_2/moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	ú*
	keep_dims(2(
&batch_normalization_2/moments/varianceÃ
%batch_normalization_2/moments/SqueezeSqueeze+batch_normalization_2/moments/mean:output:0*
T0*
_output_shapes	
:ú*
squeeze_dims
 2'
%batch_normalization_2/moments/SqueezeË
'batch_normalization_2/moments/Squeeze_1Squeeze/batch_normalization_2/moments/variance:output:0*
T0*
_output_shapes	
:ú*
squeeze_dims
 2)
'batch_normalization_2/moments/Squeeze_1
+batch_normalization_2/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2-
+batch_normalization_2/AssignMovingAvg/decayç
4batch_normalization_2/AssignMovingAvg/ReadVariableOpReadVariableOp=batch_normalization_2_assignmovingavg_readvariableop_resource*
_output_shapes	
:ú*
dtype026
4batch_normalization_2/AssignMovingAvg/ReadVariableOpñ
)batch_normalization_2/AssignMovingAvg/subSub<batch_normalization_2/AssignMovingAvg/ReadVariableOp:value:0.batch_normalization_2/moments/Squeeze:output:0*
T0*
_output_shapes	
:ú2+
)batch_normalization_2/AssignMovingAvg/subè
)batch_normalization_2/AssignMovingAvg/mulMul-batch_normalization_2/AssignMovingAvg/sub:z:04batch_normalization_2/AssignMovingAvg/decay:output:0*
T0*
_output_shapes	
:ú2+
)batch_normalization_2/AssignMovingAvg/mul­
%batch_normalization_2/AssignMovingAvgAssignSubVariableOp=batch_normalization_2_assignmovingavg_readvariableop_resource-batch_normalization_2/AssignMovingAvg/mul:z:05^batch_normalization_2/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype02'
%batch_normalization_2/AssignMovingAvg£
-batch_normalization_2/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2/
-batch_normalization_2/AssignMovingAvg_1/decayí
6batch_normalization_2/AssignMovingAvg_1/ReadVariableOpReadVariableOp?batch_normalization_2_assignmovingavg_1_readvariableop_resource*
_output_shapes	
:ú*
dtype028
6batch_normalization_2/AssignMovingAvg_1/ReadVariableOpù
+batch_normalization_2/AssignMovingAvg_1/subSub>batch_normalization_2/AssignMovingAvg_1/ReadVariableOp:value:00batch_normalization_2/moments/Squeeze_1:output:0*
T0*
_output_shapes	
:ú2-
+batch_normalization_2/AssignMovingAvg_1/subð
+batch_normalization_2/AssignMovingAvg_1/mulMul/batch_normalization_2/AssignMovingAvg_1/sub:z:06batch_normalization_2/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes	
:ú2-
+batch_normalization_2/AssignMovingAvg_1/mul·
'batch_normalization_2/AssignMovingAvg_1AssignSubVariableOp?batch_normalization_2_assignmovingavg_1_readvariableop_resource/batch_normalization_2/AssignMovingAvg_1/mul:z:07^batch_normalization_2/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02)
'batch_normalization_2/AssignMovingAvg_1
%batch_normalization_2/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2'
%batch_normalization_2/batchnorm/add/yÛ
#batch_normalization_2/batchnorm/addAddV20batch_normalization_2/moments/Squeeze_1:output:0.batch_normalization_2/batchnorm/add/y:output:0*
T0*
_output_shapes	
:ú2%
#batch_normalization_2/batchnorm/add¦
%batch_normalization_2/batchnorm/RsqrtRsqrt'batch_normalization_2/batchnorm/add:z:0*
T0*
_output_shapes	
:ú2'
%batch_normalization_2/batchnorm/Rsqrtá
2batch_normalization_2/batchnorm/mul/ReadVariableOpReadVariableOp;batch_normalization_2_batchnorm_mul_readvariableop_resource*
_output_shapes	
:ú*
dtype024
2batch_normalization_2/batchnorm/mul/ReadVariableOpÞ
#batch_normalization_2/batchnorm/mulMul)batch_normalization_2/batchnorm/Rsqrt:y:0:batch_normalization_2/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:ú2%
#batch_normalization_2/batchnorm/mulÒ
%batch_normalization_2/batchnorm/mul_1Mulactivation_1/Relu:activations:0'batch_normalization_2/batchnorm/mul:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿú2'
%batch_normalization_2/batchnorm/mul_1Ô
%batch_normalization_2/batchnorm/mul_2Mul.batch_normalization_2/moments/Squeeze:output:0'batch_normalization_2/batchnorm/mul:z:0*
T0*
_output_shapes	
:ú2'
%batch_normalization_2/batchnorm/mul_2Õ
.batch_normalization_2/batchnorm/ReadVariableOpReadVariableOp7batch_normalization_2_batchnorm_readvariableop_resource*
_output_shapes	
:ú*
dtype020
.batch_normalization_2/batchnorm/ReadVariableOpÚ
#batch_normalization_2/batchnorm/subSub6batch_normalization_2/batchnorm/ReadVariableOp:value:0)batch_normalization_2/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:ú2%
#batch_normalization_2/batchnorm/subÞ
%batch_normalization_2/batchnorm/add_1AddV2)batch_normalization_2/batchnorm/mul_1:z:0'batch_normalization_2/batchnorm/sub:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿú2'
%batch_normalization_2/batchnorm/add_1¦
dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource*
_output_shapes
:	ú*
dtype02
dense_2/MatMul/ReadVariableOp®
dense_2/MatMulMatMul)batch_normalization_2/batchnorm/add_1:z:0%dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_2/MatMul¤
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_2/BiasAdd/ReadVariableOp¡
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_2/BiasAddy
dense_2/SigmoidSigmoiddense_2/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_2/SigmoidÆ
.dense/kernel/Regularizer/Square/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes
:	#ú*
dtype020
.dense/kernel/Regularizer/Square/ReadVariableOp®
dense/kernel/Regularizer/SquareSquare6dense/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	#ú2!
dense/kernel/Regularizer/Square
dense/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2 
dense/kernel/Regularizer/Const²
dense/kernel/Regularizer/SumSum#dense/kernel/Regularizer/Square:y:0'dense/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense/kernel/Regularizer/Sum
dense/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *3Yd<2 
dense/kernel/Regularizer/mul/x´
dense/kernel/Regularizer/mulMul'dense/kernel/Regularizer/mul/x:output:0%dense/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense/kernel/Regularizer/mulÍ
0dense_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource* 
_output_shapes
:
úú*
dtype022
0dense_1/kernel/Regularizer/Square/ReadVariableOpµ
!dense_1/kernel/Regularizer/SquareSquare8dense_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
úú2#
!dense_1/kernel/Regularizer/Square
 dense_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_1/kernel/Regularizer/Constº
dense_1/kernel/Regularizer/SumSum%dense_1/kernel/Regularizer/Square:y:0)dense_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/Sum
 dense_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *3Yd<2"
 dense_1/kernel/Regularizer/mul/x¼
dense_1/kernel/Regularizer/mulMul)dense_1/kernel/Regularizer/mul/x:output:0'dense_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/muln
IdentityIdentitydense_2/Sigmoid:y:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

IdentityÝ	
NoOpNoOp$^batch_normalization/AssignMovingAvg3^batch_normalization/AssignMovingAvg/ReadVariableOp&^batch_normalization/AssignMovingAvg_15^batch_normalization/AssignMovingAvg_1/ReadVariableOp-^batch_normalization/batchnorm/ReadVariableOp1^batch_normalization/batchnorm/mul/ReadVariableOp&^batch_normalization_1/AssignMovingAvg5^batch_normalization_1/AssignMovingAvg/ReadVariableOp(^batch_normalization_1/AssignMovingAvg_17^batch_normalization_1/AssignMovingAvg_1/ReadVariableOp/^batch_normalization_1/batchnorm/ReadVariableOp3^batch_normalization_1/batchnorm/mul/ReadVariableOp&^batch_normalization_2/AssignMovingAvg5^batch_normalization_2/AssignMovingAvg/ReadVariableOp(^batch_normalization_2/AssignMovingAvg_17^batch_normalization_2/AssignMovingAvg_1/ReadVariableOp/^batch_normalization_2/batchnorm/ReadVariableOp3^batch_normalization_2/batchnorm/mul/ReadVariableOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp/^dense/kernel/Regularizer/Square/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp1^dense_1/kernel/Regularizer/Square/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:ÿÿÿÿÿÿÿÿÿ#: : : : : : : : : : : : : : : : : : 2J
#batch_normalization/AssignMovingAvg#batch_normalization/AssignMovingAvg2h
2batch_normalization/AssignMovingAvg/ReadVariableOp2batch_normalization/AssignMovingAvg/ReadVariableOp2N
%batch_normalization/AssignMovingAvg_1%batch_normalization/AssignMovingAvg_12l
4batch_normalization/AssignMovingAvg_1/ReadVariableOp4batch_normalization/AssignMovingAvg_1/ReadVariableOp2\
,batch_normalization/batchnorm/ReadVariableOp,batch_normalization/batchnorm/ReadVariableOp2d
0batch_normalization/batchnorm/mul/ReadVariableOp0batch_normalization/batchnorm/mul/ReadVariableOp2N
%batch_normalization_1/AssignMovingAvg%batch_normalization_1/AssignMovingAvg2l
4batch_normalization_1/AssignMovingAvg/ReadVariableOp4batch_normalization_1/AssignMovingAvg/ReadVariableOp2R
'batch_normalization_1/AssignMovingAvg_1'batch_normalization_1/AssignMovingAvg_12p
6batch_normalization_1/AssignMovingAvg_1/ReadVariableOp6batch_normalization_1/AssignMovingAvg_1/ReadVariableOp2`
.batch_normalization_1/batchnorm/ReadVariableOp.batch_normalization_1/batchnorm/ReadVariableOp2h
2batch_normalization_1/batchnorm/mul/ReadVariableOp2batch_normalization_1/batchnorm/mul/ReadVariableOp2N
%batch_normalization_2/AssignMovingAvg%batch_normalization_2/AssignMovingAvg2l
4batch_normalization_2/AssignMovingAvg/ReadVariableOp4batch_normalization_2/AssignMovingAvg/ReadVariableOp2R
'batch_normalization_2/AssignMovingAvg_1'batch_normalization_2/AssignMovingAvg_12p
6batch_normalization_2/AssignMovingAvg_1/ReadVariableOp6batch_normalization_2/AssignMovingAvg_1/ReadVariableOp2`
.batch_normalization_2/batchnorm/ReadVariableOp.batch_normalization_2/batchnorm/ReadVariableOp2h
2batch_normalization_2/batchnorm/mul/ReadVariableOp2batch_normalization_2/batchnorm/mul/ReadVariableOp2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2`
.dense/kernel/Regularizer/Square/ReadVariableOp.dense/kernel/Regularizer/Square/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp2d
0dense_1/kernel/Regularizer/Square/ReadVariableOp0dense_1/kernel/Regularizer/Square/ReadVariableOp2@
dense_2/BiasAdd/ReadVariableOpdense_2/BiasAdd/ReadVariableOp2>
dense_2/MatMul/ReadVariableOpdense_2/MatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#
 
_user_specified_nameinputs
¨
²
__inference_loss_fn_1_2280063M
9dense_1_kernel_regularizer_square_readvariableop_resource:
úú
identity¢0dense_1/kernel/Regularizer/Square/ReadVariableOpà
0dense_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOp9dense_1_kernel_regularizer_square_readvariableop_resource* 
_output_shapes
:
úú*
dtype022
0dense_1/kernel/Regularizer/Square/ReadVariableOpµ
!dense_1/kernel/Regularizer/SquareSquare8dense_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
úú2#
!dense_1/kernel/Regularizer/Square
 dense_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_1/kernel/Regularizer/Constº
dense_1/kernel/Regularizer/SumSum%dense_1/kernel/Regularizer/Square:y:0)dense_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/Sum
 dense_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *3Yd<2"
 dense_1/kernel/Regularizer/mul/x¼
dense_1/kernel/Regularizer/mulMul)dense_1/kernel/Regularizer/mul/x:output:0'dense_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/mull
IdentityIdentity"dense_1/kernel/Regularizer/mul:z:0^NoOp*
T0*
_output_shapes
: 2

Identity
NoOpNoOp1^dense_1/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2d
0dense_1/kernel/Regularizer/Square/ReadVariableOp0dense_1/kernel/Regularizer/Square/ReadVariableOp"¨L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*ª
serving_default
;
input_10
serving_default_input_1:0ÿÿÿÿÿÿÿÿÿ#;
dense_20
StatefulPartitionedCall:0ÿÿÿÿÿÿÿÿÿtensorflow/serving/predict:Ú°
÷
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer_with_weights-2
layer-4
layer_with_weights-3
layer-5
layer-6
layer_with_weights-4
layer-7
	layer_with_weights-5
	layer-8

	optimizer
trainable_variables
regularization_losses
	variables
	keras_api

signatures
¦_default_save_signature
+§&call_and_return_all_conditional_losses
¨__call__"
_tf_keras_network
"
_tf_keras_input_layer
ì
axis
	gamma
beta
moving_mean
moving_variance
trainable_variables
regularization_losses
	variables
	keras_api
+©&call_and_return_all_conditional_losses
ª__call__"
_tf_keras_layer
½

kernel
bias
trainable_variables
regularization_losses
	variables
	keras_api
+«&call_and_return_all_conditional_losses
¬__call__"
_tf_keras_layer
§
trainable_variables
 regularization_losses
!	variables
"	keras_api
+­&call_and_return_all_conditional_losses
®__call__"
_tf_keras_layer
ì
#axis
	$gamma
%beta
&moving_mean
'moving_variance
(trainable_variables
)regularization_losses
*	variables
+	keras_api
+¯&call_and_return_all_conditional_losses
°__call__"
_tf_keras_layer
½

,kernel
-bias
.trainable_variables
/regularization_losses
0	variables
1	keras_api
+±&call_and_return_all_conditional_losses
²__call__"
_tf_keras_layer
§
2trainable_variables
3regularization_losses
4	variables
5	keras_api
+³&call_and_return_all_conditional_losses
´__call__"
_tf_keras_layer
ì
6axis
	7gamma
8beta
9moving_mean
:moving_variance
;trainable_variables
<regularization_losses
=	variables
>	keras_api
+µ&call_and_return_all_conditional_losses
¶__call__"
_tf_keras_layer
½

?kernel
@bias
Atrainable_variables
Bregularization_losses
C	variables
D	keras_api
+·&call_and_return_all_conditional_losses
¸__call__"
_tf_keras_layer
Ã

Ebeta_1

Fbeta_2
	Gdecay
Hlearning_rate
Iitermmmm$m%m,m-m7m8m?m@mvvvv$v%v,v -v¡7v¢8v£?v¤@v¥"
	optimizer
v
0
1
2
3
$4
%5
,6
-7
78
89
?10
@11"
trackable_list_wrapper
0
¹0
º1"
trackable_list_wrapper
¦
0
1
2
3
4
5
$6
%7
&8
'9
,10
-11
712
813
914
:15
?16
@17"
trackable_list_wrapper
Î

Jlayers
Klayer_regularization_losses
Lnon_trainable_variables
Mmetrics
trainable_variables
Nlayer_metrics
regularization_losses
	variables
¨__call__
¦_default_save_signature
+§&call_and_return_all_conditional_losses
'§"call_and_return_conditional_losses"
_generic_user_object
-
»serving_default"
signature_map
 "
trackable_list_wrapper
':%#2batch_normalization/gamma
&:$#2batch_normalization/beta
/:-# (2batch_normalization/moving_mean
3:1# (2#batch_normalization/moving_variance
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
<
0
1
2
3"
trackable_list_wrapper
°

Olayers
Player_regularization_losses
Qnon_trainable_variables
Rmetrics
trainable_variables
Slayer_metrics
regularization_losses
	variables
ª__call__
+©&call_and_return_all_conditional_losses
'©"call_and_return_conditional_losses"
_generic_user_object
:	#ú2dense/kernel
:ú2
dense/bias
.
0
1"
trackable_list_wrapper
(
¹0"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
°

Tlayers
Ulayer_regularization_losses
Vnon_trainable_variables
Wmetrics
trainable_variables
Xlayer_metrics
regularization_losses
	variables
¬__call__
+«&call_and_return_all_conditional_losses
'«"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
°

Ylayers
Zlayer_regularization_losses
[non_trainable_variables
\metrics
trainable_variables
]layer_metrics
 regularization_losses
!	variables
®__call__
+­&call_and_return_all_conditional_losses
'­"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
*:(ú2batch_normalization_1/gamma
):'ú2batch_normalization_1/beta
2:0ú (2!batch_normalization_1/moving_mean
6:4ú (2%batch_normalization_1/moving_variance
.
$0
%1"
trackable_list_wrapper
 "
trackable_list_wrapper
<
$0
%1
&2
'3"
trackable_list_wrapper
°

^layers
_layer_regularization_losses
`non_trainable_variables
ametrics
(trainable_variables
blayer_metrics
)regularization_losses
*	variables
°__call__
+¯&call_and_return_all_conditional_losses
'¯"call_and_return_conditional_losses"
_generic_user_object
": 
úú2dense_1/kernel
:ú2dense_1/bias
.
,0
-1"
trackable_list_wrapper
(
º0"
trackable_list_wrapper
.
,0
-1"
trackable_list_wrapper
°

clayers
dlayer_regularization_losses
enon_trainable_variables
fmetrics
.trainable_variables
glayer_metrics
/regularization_losses
0	variables
²__call__
+±&call_and_return_all_conditional_losses
'±"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
°

hlayers
ilayer_regularization_losses
jnon_trainable_variables
kmetrics
2trainable_variables
llayer_metrics
3regularization_losses
4	variables
´__call__
+³&call_and_return_all_conditional_losses
'³"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
*:(ú2batch_normalization_2/gamma
):'ú2batch_normalization_2/beta
2:0ú (2!batch_normalization_2/moving_mean
6:4ú (2%batch_normalization_2/moving_variance
.
70
81"
trackable_list_wrapper
 "
trackable_list_wrapper
<
70
81
92
:3"
trackable_list_wrapper
°

mlayers
nlayer_regularization_losses
onon_trainable_variables
pmetrics
;trainable_variables
qlayer_metrics
<regularization_losses
=	variables
¶__call__
+µ&call_and_return_all_conditional_losses
'µ"call_and_return_conditional_losses"
_generic_user_object
!:	ú2dense_2/kernel
:2dense_2/bias
.
?0
@1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
?0
@1"
trackable_list_wrapper
°

rlayers
slayer_regularization_losses
tnon_trainable_variables
umetrics
Atrainable_variables
vlayer_metrics
Bregularization_losses
C	variables
¸__call__
+·&call_and_return_all_conditional_losses
'·"call_and_return_conditional_losses"
_generic_user_object
: (2beta_1
: (2beta_2
: (2decay
: (2learning_rate
:	 (2	Adam/iter
_
0
1
2
3
4
5
6
7
	8"
trackable_list_wrapper
 "
trackable_list_wrapper
J
0
1
&2
'3
94
:5"
trackable_list_wrapper
<
w0
x1
y2
z3"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
(
¹0"
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
.
&0
'1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
(
º0"
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
.
90
:1"
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
N
	{total
	|count
}	variables
~	keras_api"
_tf_keras_metric
b
	total

count

_fn_kwargs
	variables
	keras_api"
_tf_keras_metric
c

total

count

_fn_kwargs
	variables
	keras_api"
_tf_keras_metric
c

total

count

_fn_kwargs
	variables
	keras_api"
_tf_keras_metric
:  (2total
:  (2count
.
{0
|1"
trackable_list_wrapper
-
}	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
/
0
1"
trackable_list_wrapper
.
	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
0
1"
trackable_list_wrapper
.
	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
0
1"
trackable_list_wrapper
.
	variables"
_generic_user_object
,:*#2 Adam/batch_normalization/gamma/m
+:)#2Adam/batch_normalization/beta/m
$:"	#ú2Adam/dense/kernel/m
:ú2Adam/dense/bias/m
/:-ú2"Adam/batch_normalization_1/gamma/m
.:,ú2!Adam/batch_normalization_1/beta/m
':%
úú2Adam/dense_1/kernel/m
 :ú2Adam/dense_1/bias/m
/:-ú2"Adam/batch_normalization_2/gamma/m
.:,ú2!Adam/batch_normalization_2/beta/m
&:$	ú2Adam/dense_2/kernel/m
:2Adam/dense_2/bias/m
,:*#2 Adam/batch_normalization/gamma/v
+:)#2Adam/batch_normalization/beta/v
$:"	#ú2Adam/dense/kernel/v
:ú2Adam/dense/bias/v
/:-ú2"Adam/batch_normalization_1/gamma/v
.:,ú2!Adam/batch_normalization_1/beta/v
':%
úú2Adam/dense_1/kernel/v
 :ú2Adam/dense_1/bias/v
/:-ú2"Adam/batch_normalization_2/gamma/v
.:,ú2!Adam/batch_normalization_2/beta/v
&:$	ú2Adam/dense_2/kernel/v
:2Adam/dense_2/bias/v
ÍBÊ
"__inference__wrapped_model_2278352input_1"
²
FullArgSpec
args 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ö2Ó
B__inference_model_layer_call_and_return_conditional_losses_2279490
B__inference_model_layer_call_and_return_conditional_losses_2279617
B__inference_model_layer_call_and_return_conditional_losses_2279284
B__inference_model_layer_call_and_return_conditional_losses_2279344À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
ê2ç
'__inference_model_layer_call_fn_2278999
'__inference_model_layer_call_fn_2279658
'__inference_model_layer_call_fn_2279699
'__inference_model_layer_call_fn_2279224À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
Þ2Û
P__inference_batch_normalization_layer_call_and_return_conditional_losses_2279719
P__inference_batch_normalization_layer_call_and_return_conditional_losses_2279753´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
¨2¥
5__inference_batch_normalization_layer_call_fn_2279766
5__inference_batch_normalization_layer_call_fn_2279779´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
ì2é
B__inference_dense_layer_call_and_return_conditional_losses_2279801¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ñ2Î
'__inference_dense_layer_call_fn_2279810¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ñ2î
G__inference_activation_layer_call_and_return_conditional_losses_2279815¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ö2Ó
,__inference_activation_layer_call_fn_2279820¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
â2ß
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_2279840
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_2279874´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
¬2©
7__inference_batch_normalization_1_layer_call_fn_2279887
7__inference_batch_normalization_1_layer_call_fn_2279900´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
î2ë
D__inference_dense_1_layer_call_and_return_conditional_losses_2279922¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ó2Ð
)__inference_dense_1_layer_call_fn_2279931¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ó2ð
I__inference_activation_1_layer_call_and_return_conditional_losses_2279936¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ø2Õ
.__inference_activation_1_layer_call_fn_2279941¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
â2ß
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_2279961
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_2279995´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
¬2©
7__inference_batch_normalization_2_layer_call_fn_2280008
7__inference_batch_normalization_2_layer_call_fn_2280021´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
î2ë
D__inference_dense_2_layer_call_and_return_conditional_losses_2280032¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ó2Ð
)__inference_dense_2_layer_call_fn_2280041¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
´2±
__inference_loss_fn_0_2280052
²
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *¢ 
´2±
__inference_loss_fn_1_2280063
²
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *¢ 
ÌBÉ
%__inference_signature_wrapper_2279405input_1"
²
FullArgSpec
args 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
"__inference__wrapped_model_2278352y'$&%,-:798?@0¢-
&¢#
!
input_1ÿÿÿÿÿÿÿÿÿ#
ª "1ª.
,
dense_2!
dense_2ÿÿÿÿÿÿÿÿÿ§
I__inference_activation_1_layer_call_and_return_conditional_losses_2279936Z0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿú
ª "&¢#

0ÿÿÿÿÿÿÿÿÿú
 
.__inference_activation_1_layer_call_fn_2279941M0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿú
ª "ÿÿÿÿÿÿÿÿÿú¥
G__inference_activation_layer_call_and_return_conditional_losses_2279815Z0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿú
ª "&¢#

0ÿÿÿÿÿÿÿÿÿú
 }
,__inference_activation_layer_call_fn_2279820M0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿú
ª "ÿÿÿÿÿÿÿÿÿúº
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_2279840d'$&%4¢1
*¢'
!
inputsÿÿÿÿÿÿÿÿÿú
p 
ª "&¢#

0ÿÿÿÿÿÿÿÿÿú
 º
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_2279874d&'$%4¢1
*¢'
!
inputsÿÿÿÿÿÿÿÿÿú
p
ª "&¢#

0ÿÿÿÿÿÿÿÿÿú
 
7__inference_batch_normalization_1_layer_call_fn_2279887W'$&%4¢1
*¢'
!
inputsÿÿÿÿÿÿÿÿÿú
p 
ª "ÿÿÿÿÿÿÿÿÿú
7__inference_batch_normalization_1_layer_call_fn_2279900W&'$%4¢1
*¢'
!
inputsÿÿÿÿÿÿÿÿÿú
p
ª "ÿÿÿÿÿÿÿÿÿúº
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_2279961d:7984¢1
*¢'
!
inputsÿÿÿÿÿÿÿÿÿú
p 
ª "&¢#

0ÿÿÿÿÿÿÿÿÿú
 º
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_2279995d9:784¢1
*¢'
!
inputsÿÿÿÿÿÿÿÿÿú
p
ª "&¢#

0ÿÿÿÿÿÿÿÿÿú
 
7__inference_batch_normalization_2_layer_call_fn_2280008W:7984¢1
*¢'
!
inputsÿÿÿÿÿÿÿÿÿú
p 
ª "ÿÿÿÿÿÿÿÿÿú
7__inference_batch_normalization_2_layer_call_fn_2280021W9:784¢1
*¢'
!
inputsÿÿÿÿÿÿÿÿÿú
p
ª "ÿÿÿÿÿÿÿÿÿú¶
P__inference_batch_normalization_layer_call_and_return_conditional_losses_2279719b3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ#
p 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ#
 ¶
P__inference_batch_normalization_layer_call_and_return_conditional_losses_2279753b3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ#
p
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ#
 
5__inference_batch_normalization_layer_call_fn_2279766U3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ#
p 
ª "ÿÿÿÿÿÿÿÿÿ#
5__inference_batch_normalization_layer_call_fn_2279779U3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ#
p
ª "ÿÿÿÿÿÿÿÿÿ#¦
D__inference_dense_1_layer_call_and_return_conditional_losses_2279922^,-0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿú
ª "&¢#

0ÿÿÿÿÿÿÿÿÿú
 ~
)__inference_dense_1_layer_call_fn_2279931Q,-0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿú
ª "ÿÿÿÿÿÿÿÿÿú¥
D__inference_dense_2_layer_call_and_return_conditional_losses_2280032]?@0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿú
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 }
)__inference_dense_2_layer_call_fn_2280041P?@0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿú
ª "ÿÿÿÿÿÿÿÿÿ£
B__inference_dense_layer_call_and_return_conditional_losses_2279801]/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ#
ª "&¢#

0ÿÿÿÿÿÿÿÿÿú
 {
'__inference_dense_layer_call_fn_2279810P/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ#
ª "ÿÿÿÿÿÿÿÿÿú<
__inference_loss_fn_0_2280052¢

¢ 
ª " <
__inference_loss_fn_1_2280063,¢

¢ 
ª " »
B__inference_model_layer_call_and_return_conditional_losses_2279284u'$&%,-:798?@8¢5
.¢+
!
input_1ÿÿÿÿÿÿÿÿÿ#
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 »
B__inference_model_layer_call_and_return_conditional_losses_2279344u&'$%,-9:78?@8¢5
.¢+
!
input_1ÿÿÿÿÿÿÿÿÿ#
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 º
B__inference_model_layer_call_and_return_conditional_losses_2279490t'$&%,-:798?@7¢4
-¢*
 
inputsÿÿÿÿÿÿÿÿÿ#
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 º
B__inference_model_layer_call_and_return_conditional_losses_2279617t&'$%,-9:78?@7¢4
-¢*
 
inputsÿÿÿÿÿÿÿÿÿ#
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 
'__inference_model_layer_call_fn_2278999h'$&%,-:798?@8¢5
.¢+
!
input_1ÿÿÿÿÿÿÿÿÿ#
p 

 
ª "ÿÿÿÿÿÿÿÿÿ
'__inference_model_layer_call_fn_2279224h&'$%,-9:78?@8¢5
.¢+
!
input_1ÿÿÿÿÿÿÿÿÿ#
p

 
ª "ÿÿÿÿÿÿÿÿÿ
'__inference_model_layer_call_fn_2279658g'$&%,-:798?@7¢4
-¢*
 
inputsÿÿÿÿÿÿÿÿÿ#
p 

 
ª "ÿÿÿÿÿÿÿÿÿ
'__inference_model_layer_call_fn_2279699g&'$%,-9:78?@7¢4
-¢*
 
inputsÿÿÿÿÿÿÿÿÿ#
p

 
ª "ÿÿÿÿÿÿÿÿÿ®
%__inference_signature_wrapper_2279405'$&%,-:798?@;¢8
¢ 
1ª.
,
input_1!
input_1ÿÿÿÿÿÿÿÿÿ#"1ª.
,
dense_2!
dense_2ÿÿÿÿÿÿÿÿÿ