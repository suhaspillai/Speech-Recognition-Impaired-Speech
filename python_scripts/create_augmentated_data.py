import os
import sys
import subprocess
from shutil import copyfile

def augment_data(source_path,dest_path,map_sox_options, tot_wav):
  count = 0
  for k in map_sox_options:
    for fname in tot_wav:
        fname_list = fname.split('_')
        path = source_path
        sox_option = k.split('_')[0]
        sox_ratio = str(map_sox_options[k])
        f_str  = sox_option+'_'+sox_ratio
        fname_list.insert(-1,f_str)
        new_fname = '_'.join(fname_list)
        d_wav = dest_path + '/' + new_fname
        d_txt = dest_path + '/' + new_fname.replace('.wav','.txt')
        if os.path.isdir(path):
            s_wav = path + '/' + fname
            s_txt = s_wav.replace('.wav','.txt')
            if os.path.exists(s_wav):
              if sox_option=='speed' or sox_option=='tempo' :
                subprocess.check_output(['sox',s_wav,d_wav,sox_option,sox_ratio]) 
                copyfile(s_txt,d_txt)
                count = count + 1 
              elif sox_option=='amplify':
                amp = str(map_sox_options[k] )
                subprocess.check_output(['sox','-v',amp,s_wav,d_wav])
                copyfile(s_txt,d_txt)
                count = count + 1

  print 'Total new speech files created = %d' % (count)


def main(argv):
  source_path = argv[0]
  dest_path = argv[1]
  map_sox_options = {}
  map_sox_options['speed_0.9']=0.9
  map_sox_options['speed_1.1']= 1.1
  map_sox_options['tempo_0.9'] = 0.9
  map_sox_options['tempo_1.1']= 1.1
  map_sox_options['amplify_2.0'] = 2.0
  tot_wav =[]

  for fname in os.listdir(source_path):
    if fname.endswith('.wav'):
      tot_wav.append(fname)

  print ('Copying Normal Files !!!')  
  for fname in tot_wav:
    copyfile(source_path+'/'+fname,dest_path+'/'+fname)
    copyfile(source_path+'/'+fname.replace('.wav','.txt'),dest_path+'/'+fname.replace('.wav','.txt'))
  
  # call function for augmenting data
  augment_data(source_path,dest_path,map_sox_options, tot_wav)
  print ('!!! Augmentation Done !!!')


if __name__=='__main__':
  main(sys.argv[1:])  

  
