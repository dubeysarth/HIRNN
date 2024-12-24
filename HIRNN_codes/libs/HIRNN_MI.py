#!/usr/bin/env python
# coding: utf-8

# In[1]:


from keras import initializers, constraints, regularizers
from keras.layers import Layer, Dense, Lambda, Activation
import keras.backend as K
import tensorflow as tf
import numpy as np
from tensorflow.keras import initializers
from tensorflow.keras.layers import concatenate 
from tensorflow.keras.backend import zeros 
from tensorflow.keras.constraints import Constraint


# In[2]:


class MinMaxConstraint(Constraint):
    def __init__(self, min_value, max_value, rate=1.0):
        self.min_value = min_value
        self.max_value = max_value
        self.rate = rate

    def __call__(self, w):
        w_clipped = tf.clip_by_value(w, self.min_value, self.max_value)
        return w + self.rate * (w_clipped - w)

    def get_config(self):
        return {'min_value': self.min_value, 'max_value': self.max_value, 'rate': self.rate}


# In[3]:


class HIRNNLayer(Layer):
    """Implementation of the standard P-RNN layer
    Hyper-parameters
    ----------
    mode: if in "normal", the output will be the generated flow;
          if in "analysis", the output will be a tensor containing all state variables and process variables
    ==========
    Parameters
    ----------
    INSC: interception store capacity (mm) | Range: (0.5, 5) 
    COEFF: maximum infiltration loss      | Range: (50, 400)
    SQ: infiltration loss exponent     | Range: (0, 6) 
    SMSC: soil moisture storage capacity                     | Range: (50, 500) 
    SUB: constant of proportionality in interflow equation  | Range: (0, 1)
    CRAK: constant of proportionality in groundwater recharge equation| Range: (0, 1) 
    RecK:baseflow linear recession parameter| Range: (0.003, 0.3) 
    Kr:reservoir release coefficient| Range: (0.01, 0.1)  
    LG:loss from Groundwater Storage| Range: (0.01, 0.5) 
    LS:loss from Direct Runoff| Range: (0.01, 10) 
    """

    def __init__(self, mode='normal', **kwargs):
        self.mode = mode
        super(HIRNNLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.INSC = self.add_weight(name='INSC', shape=(1,),
                                 initializer=initializers.Constant(value=0.5),
                                 constraint=MinMaxConstraint(min_value=0.1, max_value=1.0, rate=0.9),
                                 trainable=True)
        self.COEFF = self.add_weight(name='COEFF', shape=(1,),
                                    initializer=initializers.Constant(value=0.5),
                                    constraint=MinMaxConstraint(min_value= 1/8, max_value=1.0, rate=0.9),
                                    trainable=True)
        self.SQ =  self.add_weight(name='SQ', shape=(1,),
                                    initializer=initializers.Constant(value=0.5),
                                    constraint=MinMaxConstraint(min_value=0, max_value=1.0, rate=0.9),
                                    trainable=True)
        self.SMSC = self.add_weight(name='SMSC', shape=(1,),
                                   initializer=initializers.Constant(value=0.5),
                                   constraint=MinMaxConstraint(min_value=0.1, max_value=1.0, rate=0.9),
                                   trainable=True)
        self.SUB = self.add_weight(name='SUB', shape=(1,),
                                    initializer=initializers.Constant(value=0.5),
                                    constraint=MinMaxConstraint(min_value=0.0, max_value=1.0, rate=0.9),
                                    trainable=True)
        self.CRAK = self.add_weight(name='CRAK', shape=(1,),
                                    initializer=initializers.Constant(value=0.5),
                                    constraint=MinMaxConstraint(min_value=0.0, max_value=1.0, rate=0.9),
                                    trainable=True)

        self.RecK = self.add_weight(name='RecK', shape=(1,),
                                    initializer=initializers.Constant(value=0.5),
                                    constraint=MinMaxConstraint(min_value=0.01, max_value=1.0, rate=0.9),
                                    trainable=True)
    
        self.Kr = self.add_weight(name='Kr', shape=(1,),
                                    initializer=initializers.Constant(value=0.5),
                                    constraint=MinMaxConstraint(min_value=0.1, max_value=1.0, rate=0.9),
                                    trainable=True)
        self.LG = self.add_weight(name='LG', shape=(1,),
                                    initializer=initializers.Constant(value=0.5),
                                    constraint=MinMaxConstraint(min_value=0.01, max_value=1.0, rate=0.9),
                                    trainable=True)
        self.LS = self.add_weight(name='LS', shape=(1,),
                                    initializer=initializers.Constant(value=0.5),
                                    constraint=MinMaxConstraint(min_value=0.001, max_value=1.0, rate=0.9),
                                    trainable=True)
        super(HIRNNLayer, self).build(input_shape)
        
    def heaviside(self, x):
        """
        A smooth approximation of Heaviside step function
            if x < 0: heaviside(x) ~= 0
            if x > 0: heaviside(x) ~= 1
        """

        return (tf.keras.backend.tanh(10000000 * x) + 1) / 2

    def interception(self, Prec, PET, INSC):
        INSC = INSC*5
        INSC = tf.clip_by_value(INSC, 0.5, 5)
        IMAX = tf.keras.backend.minimum(INSC, PET)
        IMAX = tf.maximum(IMAX, 0.0) 
        # then calculate interception
        INT = tf.keras.backend.minimum(IMAX, Prec)
        INT = tf.maximum(INT, 0.0)
        # calculate runoff after interception
        INR = Prec - INT
        INR = tf.maximum(INR, 0.0)
        return [INT, INR]
    
    def soil_moisture_store(self, SMS1, SMSC):
        SMSC = SMSC*500
        SMSC = tf.clip_by_value(SMSC, 50, 500)   
        SMS1 = tf.keras.backend.minimum(SMS1, SMSC)
        SMS1 = tf.maximum(SMS1, 0.0)
        return [SMS1]
    
    def recharge_calculation(self, SMS1, SMSC, REC, SMF, ETS):
        SMSC = SMSC*500
        SMSC = tf.clip_by_value(SMSC, 50, 500)   
        RECnew  = self.heaviside((SMS1 + SMF - ETS) - SMSC)*(REC + SMS1 + SMF - ETS - SMSC) + self.heaviside(SMSC - (SMS1 + SMF - ETS))*REC
        RECnew = tf.maximum(RECnew, 0.0)
        return [RECnew]
    
    def soil_fluxes(self, PET, INT, INR, COEFF, SQ, SMSC, SUB, CRAK, SMS1):
        COEFF = COEFF*400
        COEFF = tf.clip_by_value(COEFF, 50, 400)
        SQ = SQ*6
        SQ = tf.clip_by_value(SQ, 0, 6)
        SMSC = SMSC*500
        SMSC = tf.clip_by_value(SMSC, 50, 500)
        SUB = SUB*1
        SUB = tf.clip_by_value(SUB, 0, 1)
        CRAK = CRAK*1
        CRAK = tf.clip_by_value(CRAK, 0, 1)
        
        # calculate infiltration 
        RMO = self.heaviside(COEFF*tf.keras.backend.exp(-1*SQ*SMS1/SMSC) - INR)*INR + self.heaviside(INR - COEFF*tf.keras.backend.exp(-1*SQ*SMS1/SMSC))*COEFF*tf.keras.backend.exp(-1*SQ*SMS1/SMSC)
        RMO = tf.maximum(RMO, 0.0)
        # calculate direct runoff after loading to infiltration 
        IRUN = INR - RMO
        IRUN = tf.maximum(IRUN, 0.0)
        # saturation excess runoff and interflow
        SRUN = SUB * (SMS1 / SMSC) * RMO
        SRUN = tf.maximum(SRUN, 0.0)
        # calculate recharge
        REC = CRAK * (SMS1 / SMSC) * (RMO - SRUN)
        REC = tf.maximum(REC, 0.0)
        # calculate infiltration into soil store
        SMF = RMO - SRUN - REC
        SMF = tf.maximum(SMF, 0.0)
        # calculate potential ET (amount of Evap after loses)
        POT = PET - INT
        POT = tf.maximum(POT, 0.0)
        # calculate soil evaporation
        ETS = self.heaviside(10 * SMS1/SMSC - POT)*POT + self.heaviside(POT - 10*SMS1/SMSC)*(10*SMS1/SMSC)
        ETS = tf.maximum(ETS, 0.0)
        
        return [RMO, IRUN, SRUN, REC, SMF, ETS]  

    
    def DR_calculation(self, SRUN, IRUN, Area_ratio, Q_ir, Q_or, LS):
        LS = LS*10
        LS = tf.clip_by_value(LS, 0.01, 10)
        DR = self.heaviside((SRUN + IRUN)*(1 - Area_ratio) + Q_ir + Q_or - LS)*((SRUN + IRUN)*(1 - Area_ratio) + Q_ir + Q_or - LS)
        DR = tf.maximum(DR, 0.0)
        return [DR]
    
    def GD_calculation(self, BAS, Area_ratio):
        GD = (BAS)*(1 - Area_ratio) 
        return [GD]
    
    def groundwaterstore(self, GW1, RecK, LG):  
        LG = LG*0.1
        LG = tf.clip_by_value(LG, 0.001, 0.1)
        RecK = RecK*0.3
        RecK = tf.clip_by_value(RecK, 0.003, 0.3)
        BAS = self.heaviside(GW1)*(RecK*GW1)
        BAS = tf.maximum(BAS, 0.0) 
        GWL1 = GW1 - LG # groundwater after loss
        return [BAS, GWL1]
        
    def Q_calculation(self, DR, GD):
        Q = DR + GD
        Q = tf.maximum(Q, 0.0)      
        return [Q]
    
    def release_calculation(self, RS, SRUN, IRUN, BAS, RSmax, Area_ratio, Kr):
        Kr = Kr*0.1
        Kr = tf.clip_by_value(Kr, 0.01, 0.1)
        Q_or = self.heaviside(RS + (IRUN + SRUN + BAS)*Area_ratio - RSmax)*(RS + (IRUN + SRUN + BAS)*Area_ratio - RSmax) 
        Q_or = tf.maximum(Q_or, 0.0)
        Q_ir = self.heaviside(RS + (IRUN + SRUN + BAS)*Area_ratio - RSmax)*(Kr*(RSmax)) + self.heaviside(RSmax - RS - (IRUN + SRUN + BAS)*Area_ratio)*Kr*(RS)*(tf.keras.backend.pow((RS/RSmax),1.5))
        Q_ir = tf.maximum(Q_ir, 0.0)
        Q_res_in = (IRUN + SRUN + BAS)*Area_ratio
        Q_res_in = tf.maximum(Q_res_in, 0.0)
        return [Q_res_in, Q_or, Q_ir]
        

    def step_do(self, step_in, states):
        SMS1 = states[0][:, 0:1]  # Soil moisture
        GW1 = states[0][:, 1:2]  # Groundwater 
        RS = states[0][:, 2:3] # Reservoir storage

        # Load the current input column
        Prec = step_in[:, 0:1]
        PET = step_in[:, 1:2]  
        RSmax = step_in[:, 2:3] 
        Area_ratio = step_in[:, 3:4] 
    
        [_INT, _INR] = self.interception(Prec, PET, self.INSC)
        [_SMS1] = self.soil_moisture_store(SMS1, self.SMSC)
        [_RMO, _IRUN, _SRUN, _REC, _SMF, _ETS] = self.soil_fluxes(PET, _INT, _INR, self.COEFF, self.SQ, self.SMSC, self.SUB, self.CRAK, _SMS1)   
        _dSMS = _SMF - _ETS  
        next_SMS = _SMS1 + tf.clip_by_value(_dSMS, -1e5, 1e5)

        [_RECnew] = self.recharge_calculation(_SMS1, self.SMSC, _REC, _SMF, _ETS)
        [_BAS, _GWL1] = self.groundwaterstore(GW1, self.RecK, self.LG)
        _dGW = _RECnew - _BAS
        next_GW = _GWL1 + tf.clip_by_value(_dGW, -1e5, 1e5)

        [_Q_res_in, _Q_ir, _Q_or] = self.release_calculation(RS, _SRUN, _IRUN, _BAS, RSmax, Area_ratio, self.Kr)
        _dRS = _Q_res_in - _Q_ir - _Q_or
        next_RS= RS + tf.clip_by_value(_dRS, -1e5, 1e5)
        step_out = concatenate([next_SMS, next_GW, next_RS], axis=1)
        return step_out, [step_out]
    
    def call(self, inputs):
        # Load the input vector
        Prec = inputs[:, :, 0:1]
        PET = inputs[:, :, 1:2]
        RSmax = inputs[:, :, 2:3] 
        Area_ratio = inputs[:, :, 3:4]        
        # Concatenate Prec, PET, RSmax, and Area_ratio into a new input
        new_inputs = concatenate((Prec, PET, RSmax, Area_ratio), axis=-1)
        # Define 3 initial state variables at the beginning
        init_states = [tf.zeros((tf.shape(new_inputs)[0], 3))]
        # Recursively calculate state variables by using RNN
        _, outputs, _ = tf.keras.backend.rnn(self.step_do, new_inputs, init_states)
        SMS1 = outputs[:, :, 0:1]
        GW1 = outputs[:, :, 1:2]
        RS = outputs[:, :, 2:3]
        SMS1 = tf.roll(SMS1, shift=1, axis=1)
        GW1 = tf.roll(GW1, shift=1, axis=1)
        [INT, INR] = self.interception(Prec, PET, self.INSC)
        [SMS1] = self.soil_moisture_store(SMS1, self.SMSC)
        [RMO, IRUN, SRUN, REC, SMF, ETS] = self.soil_fluxes(PET, INT, INR, self.COEFF, self.SQ, self.SMSC, self.SUB, self.CRAK, SMS1)
        [RECnew] = self.recharge_calculation(SMS1, self.SMSC, REC, SMF, ETS)
        [BAS, GWL1] = self.groundwaterstore(GW1, self.RecK, self.LG)
        [Q_res_in, Q_or, Q_ir] = self.release_calculation(RS, SRUN, IRUN, BAS, RSmax, Area_ratio, self.Kr)
        [DR] = self.DR_calculation(SRUN, IRUN, Area_ratio, Q_ir, Q_or, self.LS)
        [GD] = self.GD_calculation(BAS, Area_ratio)
        [Q] = self.Q_calculation(DR, GD)
        
        if self.mode == "normal":
            return Q
        elif self.mode == "analysis": 
            return concatenate([SMS1, GW1, RS, Q], axis=-1)
    def compute_output_shape(self, input_shape):
        if self.mode == "normal":
            return (input_shape[0], input_shape[1], 1)
        elif self.mode == "analysis":
            return (input_shape[0], input_shape[1], 4)


# In[4]:


class ConvLayer(Layer):
    """Implementation of the standard 1D-CNN layer

    Hyper-parameters (same as the Conv1D in https://keras.io/layers/convolutional/)
    ----------
    filters: The dimensionality of the output space (i.e. the number of output filters in the convolution)
    kernel_size: An integer or tuple/list of a single integer, specifying the length of the 1D convolution window
    padding: One of "valid", "causal" or "same"
    seed: Random seed for initialization
    """

    def __init__(self, filters, kernel_size, padding, seed=200, **kwargs):
        self.filters = filters
        self.kernel_size = kernel_size
        self.padding = padding
        self.seed = seed
        super(ConvLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.kernel = self.add_weight(name='kernel',
                                      shape=(self.kernel_size, input_shape[-1], self.filters),
                                      initializer=initializers.random_uniform(seed=self.seed),
                                      trainable=True)

        self.bias = self.add_weight(name='bias',
                                    shape=(self.filters,),
                                    initializer=initializers.Zeros(),
                                    trainable=True)

        super(ConvLayer, self).build(input_shape)

    def call(self, inputs):

        outputs = tf.keras.backend.conv1d(inputs, self.kernel, strides=1, padding=self.padding)
        outputs = tf.keras.backend.elu(tf.keras.backend.bias_add(outputs, self.bias))
        return outputs

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], self.filters)


class ScaleLayer(Layer):
    """
    Scale the inputs with the mean activation close to 0 and the standard deviation close to 1
    """

    def __init__(self, **kwargs):
        super(ScaleLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        super(ScaleLayer, self).build(input_shape)

    def call(self, inputs):
        met = inputs[:, :, :-1]
        flow = inputs[:, :, -1:]

        self.met_center = tf.keras.backend.mean(met, axis=-2, keepdims=True)
        self.met_scale = tf.keras.backend.std(met, axis=-2, keepdims=True)
        self.met_scaled = (met - self.met_center) / self.met_scale

        return concatenate([self.met_scaled, flow], axis=-1)
    def compute_output_shape(self, input_shape):
        return input_shape


# In[ ]:




