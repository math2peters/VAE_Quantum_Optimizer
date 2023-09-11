import matplotlib.pyplot as plt
import numpy as np
from tensorflow import keras
import tensorflow as tf
from data_generator import VAEDataGeneratorKeras
import platform
import sys
import pickle
from VAE_optimizer_architecture import VAE, Sampling
from scipy.optimize import minimize


np.random.seed(0)
tf.random.set_seed(0)

def decode_latent_space(latent_variables, vae):
    latent_variables_tensor = tf.convert_to_tensor([latent_variables], dtype=tf.float32)
    decoded_output = vae.decoder(latent_variables_tensor)
    return decoded_output.numpy()[0][:, 0]

def encode_latent_space(input_array, vae):
    # Make sure the input_array has the right shape (1, input_size, 1)
    input_array = np.array(input_array).reshape(-1, vae.input_size, 1)
    # Use the encoder of the VAE to predict the mean and log variance
    mean, log_var = vae.encoder.predict(input_array, verbose=0)
    # Create an instance of the Sampling layer
    sampling_layer = Sampling()
    # Call the sampling layer with the mean and log variance to get the sampled latent variable
    sampled_latent_variable = sampling_layer([mean, log_var])
    #  sampled latent variable
    return sampled_latent_variable.numpy()

def get_latent_mean_var(input_array, vae):
    # Make sure the input_array has the right shape (1, input_size, 1)
    input_array = np.array(input_array).reshape(-1, vae.input_size, 1)
    # Use the encoder of the VAE to predict the mean and log variance
    mean, log_var = vae.encoder.predict(input_array, verbose=0)
    # Create an instance of the Sampling layer
    # Return the mean,  variance
    return mean, tf.exp(log_var)

def population_prediction_function(latent_variables, vae):
    latent_variables = np.array([latent_variables])  # Ensure the shape is compatible
    return np.float64(vae.population_predictor_network.predict(latent_variables, verbose=0)[0][0])

def population_prediction_function_with_gradient(latent_variables, vae, with_penalty=False, found_minima=[], penalty_scaling=1):
    latent_variables_tensor = tf.Variable([latent_variables], dtype=tf.float32)
    with tf.GradientTape() as tape:
        population_prediction = vae.population_predictor_network(latent_variables_tensor)
    gradient = tape.gradient(population_prediction, latent_variables_tensor)
    population_prediction_function_result = -np.float64(population_prediction.numpy()[0][0])
    gradient_function_result = -np.float64(gradient.numpy()[0])
    if with_penalty:
        population_prediction_function_result =population_prediction_function_result+penalty(latent_variables, found_minima, penalty_scaling=penalty_scaling)
        gradient_function_result =gradient_function_result+penalty(latent_variables, found_minima, penalty_scaling=penalty_scaling)
        
        
    return population_prediction_function_result, gradient_function_result

def penalty(x, found_minima, penalty_factor=1, penalty_scaling=5):
    penalty_value = 0
    for minimum in found_minima:
        distance = np.linalg.norm(x - np.array(minimum))

        penalty_value += min(1, penalty_factor / distance**2 * np.exp(-penalty_scaling*distance))
        # penalty_scaling controls the strength of the penalty
    return penalty_value

def find_minima(latent_dim, cfwg, vae, num_minima, penalty_scaling, stochasticity=1, initial_points_starter = [], min_value=-20, max_value=+20):
    # Define the bounds of your latent space
    bounds = [(min_value, max_value) for _ in range(latent_dim)]

    # Store the minima
    minima = []
    initial_points = (np.random.rand(num_minima, latent_dim)-.5)*stochasticity
    for i, val in enumerate(initial_points_starter):
        initial_points[i] = val
    # Perform the optimization starting from different initial points
    for initial_point in initial_points:
        # add gpt penalty code
        result = minimize(cfwg, initial_point, method='L-BFGS-B', jac=True, bounds=bounds, args=(vae, True, minima,penalty_scaling))
        minima.append(result.x)
        
        print(cfwg(result.x, vae, True, minima,penalty_scaling))
        
    return minima



if __name__ == '__main__':

    input_size = 64
    latent_dim = 8
    batch_size=128
    use_saved = True
    

    with open('data_v3.pkl', 'rb') as file:
        # Use pickle.load() to load the data from the file
        pickle_data = pickle.load(file)
        pickle_data = np.array(pickle_data)
        pickle_data = pickle_data.reshape(-1, 65)
        print(pickle_data.shape)
        reconstruction_train = pickle_data[:, :-1]
        population_train = pickle_data[:, -1]

    validation_generator = VAEDataGeneratorKeras(array_size=input_size, num_samples=512, batch_size=batch_size)
    validation_generator.get_population_value = True
    
    print(f"Python Platform: {platform.platform()}")
    print(f"Tensor Flow Version: {tf.__version__}")
    print()
    print(f"Python {sys.version}")
    gpu = len(tf.config.list_physical_devices('GPU'))>0
    print("GPU is", "available" if gpu else "NOT AVAILABLE")

    vae = VAE(input_size=input_size, latent_dim=latent_dim)
    # vae.encoder.summary()
    # vae.decoder.summary()    

    lr_decay = keras.optimizers.schedules.ExponentialDecay(1e-1, 1, .15)
    optimizer = keras.optimizers.legacy.Adam(amsgrad=True, learning_rate=lr_decay)

    vae.vae_model.compile(optimizer)
    #vae.vae_model.summary()

    indices = np.where(population_train < 0.1)
    low_population_train = population_train[indices]
    low_reconstruction_train = reconstruction_train[indices]
    # Generate random indices
    random_indices = np.random.choice(len(low_population_train), size=128, replace=False)

    # Select elements from both arrays using the same indices
    low_population_train = low_population_train[random_indices]
    low_reconstruction_train = low_reconstruction_train[random_indices]

    
    print("Max fed population is {:.4f}".format(max(low_population_train)))
    print("Avg fed population is {:.4f}".format(np.mean(low_population_train)))
    print("Number of VAE-trained samples with populations > 0.250: {:.6f}".format(len(np.where(population_train > .25)[0])/len(population_train)))
    print("Number of VAE-trained samples with populations > 0.225: {:.6f}".format(len(np.where(population_train > .225)[0])/len(population_train)))
    print("Number of VAE-trained samples with populations > 0.200: {:.6f}".format(len(np.where(population_train > .200)[0])/len(population_train)))

    
    vae.vae_model.load_weights("VAEGO_no_population.h5")
    vae.vae_model.get_layer('vae_loss_layer').gamma = 1
    vae.vae_model.get_layer('vae_loss_layer').beta = 0
    vae.vae_model.get_layer('vae_loss_layer').alpha = 0
    vae.set_trainable_layers(['population_predictor_network'])
    
    vae.vae_model.compile(optimizer)
    vae.vae_model.fit(x=[low_reconstruction_train, low_population_train],
            y=low_reconstruction_train,
            sample_weight=np.clip(1/(1-np.array(low_population_train))**10, 1, 25),
            validation_data=validation_generator,
            steps_per_epoch=1,
            epochs=16,
            use_multiprocessing=False)
    
        
    reconstruction_list = []
    population_list = []
    for i in range(len(low_population_train)):
        population_list.append(low_population_train[i])
        reconstruction_list.append(low_reconstruction_train[i])
        
    vae.vae_model.get_layer('vae_loss_layer').gamma = 1
    vae.vae_model.get_layer('vae_loss_layer').alpha = 1
    #vae.vae_model.get_layer('vae_loss_layer').reg = 0
    vae.vae_model.set_trainable = True
    
    #lr_decay = keras.optimizers.schedules.ExponentialDecay(1e-3, 2, .99)
    optimizer = keras.optimizers.legacy.Adam(amsgrad=True, learning_rate=1e-4)
    vae.vae_model.compile(optimizer)
    
    initial_points_starter_list = []
    
    plt.ion()  # Turn on interactive mode
    fig, axes = plt.subplots(4, 1, figsize=(12, 10))  # Create 4 subplots

    for i in range (100):
        largest_population_indices = np.argsort(np.array(population_list))[-4:]
        initial_points_starter_list = np.array(encode_latent_space(np.array(reconstruction_list)[largest_population_indices], vae))
        minima = find_minima(latent_dim, population_prediction_function_with_gradient, vae, num_minima=8, 
                             initial_points_starter=initial_points_starter_list, penalty_scaling=min(5+i/5, 8), stochasticity=10)

        decoded_latent = [decode_latent_space(np.array(m), vae) for m in minima]
        [reconstruction_list.append(j) for j in decoded_latent]
        [population_list.append(validation_generator.get_population(j)) for j in decoded_latent]
    
        
        vae.vae_model.fit(x=[np.array(reconstruction_list), np.array(population_list)],
                y=np.array(reconstruction_list),
                #sample_weight=np.clip(1/(1-abs(np.array(population_list)))**10, 1, 25),
                batch_size=len(population_list),
                steps_per_epoch=1,
                epochs=1,
                verbose=0)
        
        if i % 5 == 0:
            print("Iteration {}".format(i))
            print("Largest Population so far: {:.4f}\n".format(max(population_list)))
        
        # Plotting cost_function(m, vae)
        plot_c_list = []
        for m_value, label in zip(minima, range(len(minima))):
            plot_c_list.append(np.abs(population_prediction_function(m_value, vae)))
        axes[0].clear()
        axes[0].plot(plot_c_list, label='predicted')
        axes[0].plot(list(population_list[-len(minima):]), label='measured')
        axes[0].legend()
        axes[0].set_title('Population Function')

        # Plotting reconstruction_list[-len(m):]
        axes[1].clear()
        reconstructions = reconstruction_list[-len(minima):]
        for recon, label in zip(reconstructions, range(len(minima))):
            axes[1].plot(recon, label=f'm_{label}')
        axes[1].set_title('Reconstructions')

        # Plotting m mean + var
        axes[2].clear()
        axes[3].clear()
        for m_value in range(len(minima)):
            decoded_latent[m_value]
            #print(get_latent_mean_var(decoded_latent[m_value], vae))
            mean, variance = get_latent_mean_var(decoded_latent[m_value], vae)
            #print(mean)
            axes[2].plot(*mean, label=f'm_{m_value}')
            axes[3].plot(*variance, label=f'm_{m_value}')
        axes[2].set_title('Mean')
        axes[3].set_title('Variance')

        plt.tight_layout()
        plt.draw()  # Redraw the plots
        plt.pause(0.001)  # Pause to update the plots
    

    plt.close()
    plt.ioff()
    minima = find_minima(latent_dim, population_prediction_function_with_gradient, vae, num_minima=8, penalty_scaling=4, stochasticity=.1)
    plt.figure(figsize=(12, 8))
    for i in range(len(minima)):
        
        plt.subplot(4, 2, i+1)
        pred_y = decode_latent_space(minima[i], vae)
        population = validation_generator.get_population(pred_y)
        print("Latent space is: {}".format(minima[i]))
        print("Predicted population is: {}".format(abs(population_prediction_function(minima[i], vae))))
        print("Population is: {}".format(abs(population)))
        print()
        x = np.linspace(0, validation_generator.max_time, len(pred_y))
        plt.scatter(x, pred_y)
    
    plt.tight_layout()
    plt.show()
    
    
    plt.plot(population_list)
    plt.tight_layout()
    plt.show()
    
    index = np.argmax(population_list)
    print(max(population_list))
    plt.plot(reconstruction_list[index])
    plt.show()

