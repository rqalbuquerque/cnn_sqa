import os
import glob
import threading
import train
import config

if __name__ == '__main__':
  configurations_dir = "/home/rqa/renato/git/A-Convolutional-Neural-Network-Approach-for-Speech-Quality-Assessment/configurations"

  if os.path.isdir(configurations_dir):
    os.chdir(configurations_dir)
    files = glob.glob("*.json")
    for filename in files:
      print('----------------------------------------------')
      print('Running configuration: ' + filename)
      print('----------------------------------------------')
      FLAGS, _unparsed = config.set_flags(config.read_config(filename))
      processThread = threading.Thread(target=train.main, args=[[FLAGS, os.path.splitext(filename)[0]]])
      processThread.start()
      processThread.join()
      print('----------------------------------------------')
      print('Ending configuration: ' + filename)
      print('----------------------------------------------')
      print('')
