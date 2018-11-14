import save_data

if __name__ == '__main__':
  inputdir = "/home/rqa/git/A-Convolutional-Neural-Network-Approach-for-Speech-Quality-Assessment/database/speech_dataset/EXP1_SPECTROGRAM_TEST"
  outputdir = "/home/rqa/git/A-Convolutional-Neural-Network-Approach-for-Speech-Quality-Assessment/database/speech_dataset/EXP1_SPECTROGRAM_IMAGES"
  save_data.save_spectrogram_images(inputdir, outputdir)

  # inputdir = "/home/rqa/git/database_tmp/speech_dataset/EXP3_CODED_FILES"
  # outputdir = "/home/rqa/git/database_tmp/speech_dataset/EXP3_SPECTROGRAM_MAT"
  # save_data.save_spectrogram_mat(inputdir, outputdir)