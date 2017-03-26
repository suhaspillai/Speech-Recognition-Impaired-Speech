import os
from shutil import copyfile
import sys



def main(argv):
    print ('Data creation started')     
    source_path = argv[0]
    destination_path = argv[1]
    test_speaker = argv[2]
    tot_speakers = os.listdir(source_path)
    corrupted_files = open('corrupt_files.txt','w')
    flag_not_corrupted = True
    for speaker in tot_speakers:
        if speaker!=test_speaker:
            split = 'train'
        else:
            split = 'test'

        session_path = source_path + '/' + speaker + '/'
        session_folders = os.listdir(session_path)
        for folder in session_folders:
            if folder.startswith('Session'):
                sub_path = session_path + folder + '/'
                if os.path.isdir(sub_path + 'wav_arrayMic'):
                    arr_mic_path = sub_path + 'wav_arrayMic' + '/'
                    head_mic_path = sub_path + 'wav_headMic' + '/'
                    files = os.listdir(arr_mic_path)
                    for speech_file in files:
                        wav_file = arr_mic_path + speech_file
                        text_file = sub_path + 'prompts' + '/' + speech_file.replace('wav','txt')
                        if os.path.exists(wav_file) and os.path.exists(text_file):
                           f_read = open(text_file)
    			   for line in f_read:
                               if line.find('.jpg')>0 or line.find('[')>0 or line.find(']')>0:
                                   corrupted_files.write(text_file+'\n')
                                   flag_not_corrupted = False
                                   break       
                            
                           if flag_not_corrupted:
                               new_wav_file = speaker + '_' + folder + '_' + 'arrayMic' + '_' + speech_file
                               new_text_file = new_wav_file.replace('wav','txt')
                               d_path = destination_path + '/' + split + '/'
                               copyfile(wav_file,d_path + speech_file)
                               copyfile(text_file, d_path + speech_file.replace('wav','txt'))
                               os.rename(d_path + speech_file, d_path + new_wav_file)
                               os.rename(d_path  + speech_file.replace('wav','txt'), d_path + new_text_file)

                        wav_file_head = head_mic_path + speech_file
                        if os.path.exists(wav_file_head) and os.path.exists(text_file) and flag_not_corrupted:
                            new_wav_file_head = speaker + '_' + folder + '_' + 'headMic' + '_' + speech_file
                            new_text_file_head = new_wav_file_head.replace('wav','txt')
                            copyfile(wav_file_head,d_path + speech_file)
                            copyfile(text_file, d_path + speech_file.replace('wav','txt'))
                            os.rename(d_path + speech_file, d_path + new_wav_file_head)
                            os.rename(d_path  + speech_file.replace('wav','txt'), d_path + new_text_file_head)

                        flag_not_corrupted = True
    corrupted_files.close()            
    print ('!!! Done !!!')               


if __name__=='__main__':
    main(sys.argv[1:])
