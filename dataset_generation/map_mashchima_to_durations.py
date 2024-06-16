from argparse import ArgumentParser
import os

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--folder_with_data',type=str,required=True,help="Path to folder containing dataset with mashchima files")
    parser.add_argument('--destination_folder',type=str,required=True,help="Path to folder containing dataset with mashchima files")

    