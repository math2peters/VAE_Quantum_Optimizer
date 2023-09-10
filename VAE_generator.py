import numpy as np
from scipy.special import jv, airy
import matplotlib.pyplot as plt
from keras.utils import Sequence
from scipy.ndimage import gaussian_filter1d
import qutip
import numpy as np
import scipy.interpolate as interpolate
import matplotlib
import matplotlib.pylab as plt
from qutip import basis, mesolve, destroy
from torch.utils.data import Dataset
import torch
import pickle
from qutip import mesolve, Options


# random functions used to train the model
def gaussian(t, t0, A, sigma, C):
    out = A*np.exp(-(t-t0)**2/(2*sigma**2)) + C
    # out /= np.max(abs(out))
    return out

def sinc(t, t0, A, w, C):
    out = A*np.sin(w*(t-t0))/(w*(t-t0)) + C
    # out /= np.max(out)
    return out

def sin(t, t0, A, w, C):
    out = A*np.sin(w*(t-t0)) + C
    # out /= np.max(out)
    return out

def exp(t, t0, A, a):
    out = A*np.exp(-a*(t-t0))
    # out /= np.max(out)
    return out

def noise(t, correlation_length, filter_size, C, **kwargs):
    

    gaussian_kernel = gaussian_filter1d(np.ones((filter_size,))/filter_size, sigma=correlation_length)

    # Convolve the array with the Gaussian kernel
    convolved_array = np.convolve(np.random.normal(size=len(t)), gaussian_kernel, mode='same') + C
    
    return convolved_array

def linear(t, m, C):
    return m*t + C 


class VAEDataGeneratorKeras(Sequence):
    def __init__(self, array_size, num_samples, batch_size, shared_data_list=None, save_data=False):
        """initialize the data generator

        """
        #self.shared_data_list = shared_data_list
        self.shared_data_list = shared_data_list
        self.num_samples = num_samples # number of samples to generate
        self.batch_size = batch_size # minibatch size
        self.array_size = array_size # number of samples for a single prediction for NN
        self.max_time = 1 # max time in the simulation
        self.t_list = np.linspace(0,self.max_time, array_size) # list of times for the simulation
        self.t_list_scale = 5
        self.best_cost = 0
        self.output_scale = 2 # scale the output of the NN before running through qutip
        self.data_list = []
        self.cost_list = []
        self.get_cost_value = True
        self.save_data = save_data
    


    def __len__(self):
        return int(np.ceil(self.num_samples / self.batch_size))

    def __getitem__(self, idx):
        start_index = idx * self.batch_size
        end_index = min((idx + 1) * self.batch_size, self.num_samples)
        

        x_train, y_train = self.generate_data()
        
        if self.save_data:
            self.shared_data_list.append(list(np.array(list(x_train.T)+list(y_train.reshape((1, -1)))).T))

        return [x_train, y_train], x_train
    

    
    def generate_data(self):
        # generate a random array of integers to determine which function to use
        random_array = np.random.randint(0, 18, size=self.batch_size)
        
        training_list = []
        cost_list = []
        
        for integer in random_array:
            C = (np.random.random_sample(5)-.5)*2
            P = (np.random.random_sample(5)-.5)**2*20
            t0 = np.random.random_sample(5)*self.max_time
            A = np.sqrt(np.random.random_sample(5))
            sigma = np.random.random_sample(5)**2*self.max_time+np.random.random_sample(5)**3
            w = 2*np.pi*np.random.random_sample(5)**2*self.max_time*5+self.max_time
            m = (np.random.random_sample(5)-.5)*3
            a = (np.random.random_sample(5)-.5)*np.random.random_sample(5)**2*10
            
            combo_val = np.random.random_sample()
            
            if  integer == 0:
                data = gaussian(self.t_list, t0[0], A[0], sigma[0], C[0])
                
            if integer == 1:
                data = sinc(self.t_list, t0[0], A[0], w[0], C[0]/5)
                
            if integer == 2:
                data = sin(self.t_list, t0[0], A[0], w[0], C[0]/10)
                
            if integer == 3:
                data = linear(self.t_list, m[0], C[0])
                
            if integer == 4:
                data1 = sinc(self.t_list, t0[0], A[0], w[0], C[0]/10)
                data2 = gaussian(self.t_list, t0[1], A[1], sigma[1], C[0])
                data = data1*combo_val + data2*(1-combo_val)
                
            if integer == 5:
                data1 = sin(self.t_list, t0[0], A[0], w[0], C[0])
                data2 = gaussian(self.t_list, t0[1], A[1], sigma[1], C[1])
                data = data1*combo_val + data2*(1-combo_val)
                
            if integer == 6:
                data1 = linear(self.t_list, m[0], C[0])
                data2 = gaussian(self.t_list, t0[1], A[1], sigma[1], C[1])
                data = data1*combo_val + data2*(1-combo_val)
                
            
            if integer == 7:
                data1 = sin(self.t_list, t0[0], A[0], w[0], 0)
                data2 = linear(self.t_list, m[1], C[1])
                data = data1*combo_val + data2*(1-combo_val)
                
            if integer == 8:
                data1 = sinc(self.t_list, t0[0], A[0], w[0], 0)
                data2 = gaussian(self.t_list, t0[1], A[1], sigma[1], 0)
                data3 = linear(self.t_list, m[2], C[2])
                data = data1*combo_val/2 + data2*(1-combo_val)/2 + data3/2
                
            if integer == 9:
                data1 = sin(self.t_list, t0[0], A[0], w[0], C[0]/10)
                data2 = sinc(self.t_list, t0[1], A[1], w[1], C[1]/10)
                data = data1*combo_val + data2*(1-combo_val)
                
            if integer == 10:
                exp_t0 = np.cos(np.pi*np.random.random_sample())**2 # to keep the exponential a bit closer to the edges
                data = sin(self.t_list, t0[0], A[0], w[0], C[0]/10) * exp(self.t_list, exp_t0*self.max_time, 1, a[1])
                
            if integer == 11:
                data1 = sin(self.t_list, t0[0], A[0], w[0], C[0]/10)
                data2 = gaussian(self.t_list, t0[1], A[1], sigma[1], 0)
                data = data1*data2
                
            if integer == 12:
                data1 = sinc(self.t_list, t0[0], A[0], w[0], C[0]/10)
                data2 = gaussian(self.t_list, t0[1], A[1], sigma[1], 0)
                data = data1*data2
                
            if integer == 13:
                data = sin(self.t_list**2, t0[0], A[0], w[0], C[0]/10)
                
            if  integer == 14:
                data = gaussian(self.t_list, t0[0], A[0], sigma[0], 0)
                
            if integer == 15:
                data = P[2]*self.t_list**2 + P[1]*self.t_list+ P[0]
                
            if integer == 16:
                data = P[3]*self.t_list**3+ P[2]*self.t_list**2 + P[1]*self.t_list+ P[0]
                
            if integer == 17:
                data = (P[4]*self.t_list**4 + P[3]*self.t_list**3+ P[2]*self.t_list**2 + P[1]*self.t_list+ P[0])*gaussian(self.t_list, t0[0], A[0], sigma[0], 0)
            
            if np.max(abs(data)) == 0:
                data = (P[4]*self.t_list**4 + P[3]*self.t_list**3+ P[2]*self.t_list**2 + P[1]*self.t_list+ P[0])
            data = data / np.max(abs(data))  

            augmentation_probability = 0.02
            if np.random.random_sample() < augmentation_probability:
                data[0:int(np.random.random_sample()*len(data)*0.25)] = np.random.random_sample()*.01-.005
            
            if np.random.random_sample() < augmentation_probability:
                data[-int(np.random.random_sample()*len(data)*0.25):] = np.random.random_sample()*.01-.005
                
            if np.random.random_sample() < augmentation_probability/2:
                rand1 = int(np.random.random_sample()*len(data)*0.5)+len(data)//2
                rand2 = int(np.random.random_sample()*len(data)*0.5)+len(data)//2
                data[min(rand1, rand2):max(rand1, rand2)] = np.random.random_sample()*.01-.005
                
            if np.random.random_sample() > augmentation_probability:
                data  *= 2*(np.random.rand(1) - .5)
            
            else:
                if np.random.random_sample() < .5:
                    data = np.clip(data, min(data), (max(data)-min(data)*np.random.random_sample()+min(data)))
                else:
                    data = np.clip(data, (max(data)-min(data))*np.random.random_sample()*.2+min(data), max(data))
              
            training_list.append(data)
            if self.get_cost_value:
                cost = self.get_cost(data)
            else:
                cost = -1
            cost_list.append(cost)

        
        self.data_list.append(data)
        self.cost_list.append(cost)
        return np.array(training_list), np.array(cost_list)
            
    def get_cost(self, function_vals, return_full_values=False):
        # make a qutip hamiltonian and see how the function evolves the population in |1>

        # have to interpolate to make mesolve happy
        f = interpolate.interp1d(self.t_list*self.t_list_scale, function_vals*self.output_scale, fill_value="extrapolate")
        g = lambda t, args: f(t)
        
        # two level hamiltonian. H1 is multipled by the function vals at each time step
        H0 = -0.5 * 1 * qutip.operators.sigmaz()
        H1 = qutip.operators.sigmax()

        H = [H0, [H1, g]]

        psi0 = basis(2, 0)
        proj1 = qutip.ket2dm(qutip.ket("1"))
        # solve the hamiltonian for the times given in t_list, there is a collapse operator that causes damping from |1> to |0>
        try:
            result = mesolve(H, psi0,self.t_list,[destroy(2)*np.sqrt(6)], e_ops=[proj1]) 
        except Exception as e:
            error_message = f"An error occurred during the mesolve operation: {str(e)}"
            print(error_message)
            return 0

        if not return_full_values:
            return result.expect[0][-1]
        else:
            return result
        
        # # two level hamiltonian. H1 is multipled by the function vals at each time step
        # H0 = qutip.Qobj([[0, 0.0, 0.0], [0.0, 1, 0.0], [0.0, 0.0, .5]])
        # H1 = qutip.Qobj([[0, 1, 0.0], [1, 0, 0.0], [0.0, 0.0, 0]])

        # H = [H0, [H1, g]]

        # psi0 = basis(3, 0)
        # proj1 = qutip.ket2dm(qutip.ket("1"))
        # # solve the hamiltonian for the times given in t_list, there is a collapse operator that causes damping from |1> to |0>
        # result = mesolve(H, psi0,self.t_list,[destroy(2)*np.sqrt(.25)], e_ops=[proj1]) 

                

    def on_epoch_end(self):
        pass
    
# class VAEDataGenerator(Dataset):
#     def __init__(self, array_size, num_samples, batch_size):
#         # Initialization logic remains largely the same
#         self.num_samples = num_samples
#         self.batch_size = batch_size
#         self.array_size = array_size
#         self.max_time = 1
#         self.t_list = np.linspace(0, self.max_time, array_size)
#         self.best_cost = 0
#         self.output_scale = 2
#         self.data_list = []
#         self.cost_list = []

#     def __len__(self):
#         return int(np.ceil(self.num_samples / self.batch_size))

#     def __getitem__(self, idx):
#         batch_data, target_data = self.generate_data()
#         weights = target_data
#         weights = np.clip(weights, 0.25, 1)
#         #print(torch.tensor(batch_data, dtype=torch.float).shape)
#         return torch.tensor(batch_data, dtype=torch.float), torch.tensor(batch_data, dtype=torch.float) #, torch.tensor(weights, dtype=torch.float)
    
#     # def __iter__(self):

#     #     batch_data, target_data = self.generate_data()
#     #     weights = target_data
#     #     weights = np.clip(weights, 0.25, 1)
#     #     #print(torch.tensor(batch_data, dtype=torch.float).shape)
#     #     return torch.tensor(batch_data, dtype=torch.float), torch.tensor(batch_data, dtype=torch.float) #, torch.tensor(weights, dtype=torch.float)

#     def generate_data(self):
#         # generate a random array of integers to determine which function to use
#         random_array = np.random.randint(0, 10, size=self.batch_size)
        
#         training_list = []
#         cost_list = []

#         for integer in random_array:
#             C = (np.random.random_sample()-.5)
#             t0 = np.random.random_sample()*self.max_time
#             A = np.random.random_sample()
#             sigma = np.random.random_sample()*self.max_time*2
#             w = np.random.random_sample()*self.max_time*4
#             m = (np.random.random_sample()-.5)*3
#             C = (np.random.random_sample()-.5)
#             combo_val = np.random.random_sample()
            
#             if  integer == 0:
#                 data = gaussian(self.t_list, t0, A, sigma, C)
                
#             if integer == 1:
#                 data = sinc(self.t_list, t0, A, w, C)
                
#             if integer == 2:
#                 data = sin(self.t_list, t0, A, w, C)
                
#             if integer == 3:
#                 data = linear(self.t_list, m, C)
                
#             if integer == 4:
#                 data1 = sinc(self.t_list, t0, A, w, C)
#                 data2 = gaussian(self.t_list, t0, A, sigma, C)
#                 data = data1*combo_val + data2*(1-combo_val)
                
#             if integer == 5:
#                 data1 = sin(self.t_list, t0, A, w, C)
#                 data2 = gaussian(self.t_list, t0, A, sigma, C)
#                 data = data1*combo_val + data2*(1-combo_val)
                
#             if integer == 6:
#                 data1 = linear(self.t_list, m, C)
#                 data2 = gaussian(self.t_list, t0, A, sigma, C)
#                 data = data1*combo_val + data2*(1-combo_val)
                
            
#             if integer == 7:
#                 data1 = sin(self.t_list, t0, A, w, C)
#                 data2 = linear(self.t_list, m, C)
#                 data = data1*combo_val + data2*(1-combo_val)
                
#             if integer == 8:
#                 data1 = sinc(self.t_list, t0, A, w, C)
#                 data2 = gaussian(self.t_list, t0, A, sigma, C)
#                 data3 = linear(self.t_list, m, C)
#                 data = data1*combo_val/2 + data2*(1-combo_val)/2 + data3/2
                
#             if integer == 9:
#                 data1 = sin(self.t_list, t0, A, w, C)
#                 data2 = sinc(self.t_list, t0, A, w, C)
#                 data = data1*combo_val + data2*(1-combo_val)
            
                
            
            
#             # if integer == 0:
#             #     while cost < .4 :
#             #         t0 = random.uniform(0, self.max_time)
#             #         A = random.uniform(-1, -1)
#             #         sigma = random.uniform(0.1, self.max_time/2)
#             #         C = random.uniform(-1, 1)
#             #         data = gaussian(self.t_list, t0, A, sigma, C)
#             #         data = np.clip(data, -1, 1)
#             #         cost = self.get_cost(data)
#             #data=self.t_list
#             # clip the data to be between -1 and 1 (we'll eventually scale the gan generator output to be between -1 and 1)
#             data = np.clip(data, -1, 1)        
#             training_list.append(data)
#             cost = self.get_cost(data)
#             cost_list.append(cost)
        
#             # if cost > self.best_cost:
#             #     self.best_cost = cost
#             #     #print("\nNew best cost: ", self.best_cost)
#             #     self.best_data = data
                
        
#         # self.data_list.append(data)
#         # self.cost_list.append(cost)
#         return np.array(training_list), np.array(cost_list)

#     def get_cost(self, function_vals, return_full_values=False):
#         # make a qutip hamiltonian and see how the function evolves the population in |1>

#         # have to interpolate to make mesolve happy
#         f = interpolate.interp1d(self.t_list, function_vals*self.output_scale, fill_value="extrapolate")
#         g = lambda t, args: f(t)
        
#         # two level hamiltonian. H1 is multipled by the function vals at each time step
#         H0 = -0.5 * 1 * qutip.operators.sigmaz()
#         H1 = qutip.operators.sigmax()

#         H = [H0, [H1, g]]

#         psi0 = basis(2, 0)
#         proj1 = qutip.ket2dm(qutip.ket("1"))
#         # solve the hamiltonian for the times given in t_list, there is a collapse operator that causes damping from |1> to |0>
#         result = mesolve(H, psi0,self.t_list,[destroy(2)*np.sqrt(.25)], e_ops=[proj1]) 

#         if not return_full_values:
#             return result.expect[0][-1]
#         else:
#             return result




#     def generate_pickle(self, file_name):
    
#         x_train_list = []
#         y_train_list = []
#         for i in range(self.num_samples):
#             if i % 100 == 0:
#                 print("At {} of {}".print(i, self.num_samples))
#             x_train, y_train = self.generate_data()
#             x_train_list.append(x_train)
#             y_train_list.append(y_train)
        
#         with open(file_name, 'wb') as file:
#             # Use pickle.dump() to save the data to the file
#             pickle.dump([x_train_list, y_train_list], file)