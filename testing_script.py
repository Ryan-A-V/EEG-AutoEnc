#imports
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
import scipy.signal as sig
import etc.helper as helper
import sys
import numpy as np
import importlib
from copy import deepcopy
from pathlib import Path


""" Run user_provided models with selected EEG and audio data. Generate plots
based on results.
"""

#options
ENVELOPE_OPTIONS = ['env', 'envelope']
SPECTRO_OPTIONS = ['spec', 'spectrogram']
MODE = "segments"
NUM_SEGMENTS = 1000
SECONDS = 10
RANDOM_STATE = 5
TRAIN_SIZE = 0.7
TEST_SIZE = 0.3
SPEC=100
LATENT_SPACES = [30] #[5, 10, 15, 20, 25, 30]
EPOCHS = 100

if __name__ == "__main__":

    #run if user doesn't provide arguments
    if len(sys.argv) == 1:
        print("USAGE: testing_script.py model_name [other_model_name ...] epochs")
        sys.exit()


    #Subject numbers and experiment
    sub = "sub-28"
    exp = "fixthemix"
    eeg_file = f"data/{sub}_task-{exp}_eegprep.vhdr"
    audio_file = f'data/{sub}_task-{exp}_aud.flac'
    
    #set sample rate. This is the value that the audio data gets resampled to
    sample_rate = 250

    #Load data in
    raw = helper.load_eeg(eeg_file)
    eeg = raw[0].get_data()
    events, event_dict = raw[1:]
    audio, audio_samp = helper.load_audio(audio_file)

    events = [18194, 21455, 24784, 27956, 31145, 33639, 37277, 40288, 44117, 46688, 49050, 51076, 53301, 56163, 59405, 63455, 66834, 79477, 83823, 87931,
            91525, 95086, 99359, 102772, 106184, 109425, 111685, 114611, 118441, 122981, 126463, 132195, 144537, 148171, 152369, 156285, 160349,
            162746, 165174, 167852, 169994, 173607, 176532, 180735, 182988, 185354, 188076, 191113, 202124, 205760, 207668, 211343, 214218, 217331, 219557,
            223642, 226545, 229466, 233711, 237012, 240139, 251742, 254136, 257780, 260465, 263313, 266339, 269752, 273907, 276216, 279055, 281751, 284651,
            289103, 300750, 303385, 305890, 308957, 311250, 313988, 316042, 319696, 323227, 326030, 328762, 331563, 335040, 337760, 343389, 355806, 358673,
            362703, 365792, 368546, 371806, 375306, 378165, 382086, 383906, 386516, 390838, 394648, 398692, 401453, 407008, 417847, 
            421406, 424928, 428115, 430842, 433495, 436684, 439429, 442036, 445459, 447736, 450474, 453479, 457217, 459527, 461458]
    
    
    #format the saved figures
    plt.gcf().set_size_inches(9, 6)

    #latent space dimensions to try
    latents = LATENT_SPACES
    EPOCHS = int(sys.argv[-1])

    #test data index to display reconstruction for
    test_ind = 0
    #keep track of best loss per model
    best_model_loss = []
    #model names for plotting
    labels = []

    for mod in sys.argv[1:-1]:
        #import model here
        print(f"MODEL {mod}\n")
        model_lib = importlib.import_module(f'models.{mod}', '.')
        test_losses = []

        #Train on each latent space and then plot loss over epochs
        for latent in latents:
            print(f"LATENT SPACE {latent} FOR MODEL {mod}\n")
            model = model_lib.Autoencoder(latent, TRAIN_SIZE, TEST_SIZE, EPOCHS, RANDOM_STATE)
            Path(f"figs/{model.model_name}").mkdir(parents=True, exist_ok=True)

            model.process_data(deepcopy(eeg), deepcopy(audio), sample_rate, events)
            model.train()
            model.plot_losses()
            plt.close()
            
            test_losses.append(model.test_loss)

            #model.visualize_activations()
            
        #plot test loss here
        plt.figure()
        plt.title(f"Average Model Loss Over Test Data")
        plt.scatter(latents, test_losses)
        plt.xlabel("Dimension of Latent Space")
        plt.ylabel("Average MSE")
        plt.savefig(f"figs/Model_{mod}_Test_Loss.png")
        

        #find the smallest loss and choose that as the best latent space for the model
        test_losses.sort()
        best_model_loss.append(test_losses[-1])
        labels.append(model.model_name)
        plt.close()
 
    plt.figure()
    plt.plot(audio)
    plt.savefig("figs/original_wave")
    print("Audio: ", audio, audio.shape)
    #plot comparison of test loss for every model trained
    plt.figure()
    plt.title(f"Average Model Loss Over Test Data For Each Model (Best)")
    plt.scatter(labels, best_model_loss)
    plt.xlabel("Model Name")
    plt.ylabel("Average MSE")
    plt.savefig("figs/Model_Comparison.png")
    #plt.show()
    plt.close()
    # python testing_script.py model1 10
