import os
import shutil

mapping = {
    'd3': '-12',
    'e3': '-11',
    'f3': '-10',
    'g3': '-9',
    'a3': '-8',
    'b3': '-7',
    'c4': '-6',
    'd4': '-5',
    'e4': '-4',
    'f4': '-3',
    'g4': '-2',
    'a4': '-1',
    'b4': '0',
    'c5': '1',
    'd5': '2',
    'e5': '3',
    'f5': '4',
    'g5': '5',
    'a5': '6',
    'b5': '7',
    'c6': '8',
    'd6': '9',
    'e6': '10',
    'f6': '11',
    'g6': '12'
}

mapping_duration = {
    '1': 'w',
    '2': 'h',
    '4': 'q',
    '8': 'e',
    '16': 's'
}

def make_directory(name):
    try:
        os.mkdir(name)
    except:
        print(f'dir "{name}" already exists')

def map_files_to_dataset(given_directory):
    make_directory('data_verovio')
    note_folders = {}
    for file in os.listdir(given_directory):
        if file == 'a3_4.png':
            print('here')
        if file.endswith('png'):
            if file == 'a3_4.png':
                print('here inside')
            note_key = str(file.split('_')[0].strip())
            note_duration = file.split('_')[1].replace('.png','').strip()
            folder_name = mapping[note_key]

            print('note key: ',note_key)
            print('note dur: ',note_duration)
            print('mapped dur: ',mapping_duration[str(note_duration)])
            print('folder name: ',folder_name)
            if str(folder_name) not in note_folders:
                make_directory('data_verovio/'+folder_name)
                note_folders[folder_name] = []

            duration = mapping_duration[str(note_duration)]
            total_images_in_folder = len(note_folders[folder_name])
            final_name = f'data_verovio/{folder_name}/{duration}{folder_name}-{total_images_in_folder+1}.png'
            print(f'created file {final_name}')
            shutil.copy(f'{given_directory}/{file}', f'{final_name}')

map_files_to_dataset('/Users/adrianseguralorente/cv-project/dataset_generation/extract_verovio_durations/data_images_verovio/')