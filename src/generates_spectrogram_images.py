import input_data

if __name__ == '__main__':
  filename = "/home/rqa/git/A-Convolutional-Neural-Network-Approach-for-Speech-Quality-Assessment/database/speech_dataset/EXP1_CODED_FILES/a3.wav"
  outputname = "/home/rqa/git/A-Convolutional-Neural-Network-Approach-for-Speech-Quality-Assessment/database/speech_dataset/EXP1_CODED_FILES/a3.png"
  input_data.save_spectrogram_image(filename, outputname)