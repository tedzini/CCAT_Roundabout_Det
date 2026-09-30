#!/disk1/home/trucker/miniconda3/envs/YOLOv11/bin/python3
import json
import datetime
from pathlib import Path
import argparse
import inspect
import numpy as np
import pandas as pd
from scipy.optimize import minimize
import cvzone
from typing import List
import struct
from collections import defaultdict

import ctypes

from TMmodules import TM_trackdata as td

###############################################################################
##                                                                           ##
##   class TM_trackdata                                                      ##
##       creates a panda structured database of object tracking data         ##
##                                                                           ##  
###############################################################################
class TM_trackDataRecords:
    bad = False
    def __init__(self, path, filename ):
        if filename is None or filename==[]:
            print('TM_trackDataRecords: filename is not assigned or empty!')
            self.bad = True
            return
        if path is None:
            self.fullfile = filename
        else:
            self.fullfile = Path(path) / filename

        prefix,suffix = fullfile_str.split('.')
        print(suffix)
        if suffix.lower() != 'bin':
            return
 
        # parse binary file:
        try:
            self.fileptr = open(self.fullfile, 'rb')
            # determine data model from header
            header =  self.fileptr.read(struct.calcsize(td.HEADER_FORMAT_SDSM))
            header_str = header.decode('ascii')
            self.col_names = header_str.split(',')
            self.data_format = td.DATA_FORMAT_SDSM
            
            if 'speed' not in self.col_names and 'heading' not in self.col_names:
                self.fileptr.seek(0)
                header = self.fileptr.read(struct.calcsize(td.HEADER_FORMAT))
                header_str = header.decode('ascii')
                self.col_names = header_str.split(',')
                self.data_format = td.DATA_FORMAT
            self.track_data_size = struct.calcsize(self.data_format)
            
            print(f'field names: {self.col_names}')

        except FileNotFoundError:
            # This block runs if the file is not found
            print(f"Error: The file, {self.fullfile}, could not be found.")
        except PermissionError:
            # This block runs if you don't have adequate access rights
            print("Error: Permission denied. Check file permissions.")
        except Exception as e:
            # This block handles any other unexpected errors
            print(f"An unexpected error occurred: {e}")

    def ReadData(self):
        if self.bad:
            print("Unable to read data because an invalid file or path was specified!\n")
            return None
            
        # if file suffix = "feather", try loading the feather file instead of directly
        # reading the raw binary data...
        fullfile_str = str(self.fullfile)
        print(fullfile_str)
        prefix,suffix = fullfile_str.split('.')
        print(suffix)
        if suffix.lower() == 'feather':
            print("Reading Apache Feather formatted dataframe file...")
            self.dataf = pd.read_feather(self.fullfile)
            self.col_names = self.dataf.columns.tolist()
            print("Done")
            return self.dataf

        print("Reading binary data...")
        self.dataf = [] # pd.DataFrame(self.col_names)
        cnt=0
        char_cnt=0
        reckey =td.recdict
        self.dataf = pd.DataFrame(columns=self.col_names)
#        while cnt < 16000:
        datarec_list = []
        while True:
            cnt +=1
            packed_data = self.fileptr.read(self.track_data_size)
            if not packed_data:
                break
            datarecord = struct.unpack(self.data_format, packed_data)
            datarecord = list(datarecord)
            datarecord[reckey['class']] = datarecord[reckey['class']].decode('ascii').strip('\00')
            datarecord[reckey['inbounds']] = datarecord[reckey['inbounds']].decode('ascii').strip('\00')
            #self.dataf.loc[len( self.dataf)] = datarecord
            datarec_list.append(datarecord)

            #print(f"{cnt:08d}")
            if cnt % 1000 == 0:
                print(f"{cnt:,}", end="\r")
        print("\nCreating data frame!")
        self.dataf = pd.DataFrame(datarec_list, columns=self.col_names)
        print("Done!")
        return self.dataf

    def SaveData(self, filename_IN=''):
        if len(filename_IN) > 0:
            df_filename = filename_IN
        else:
            # use input file name prefix
            fullfile_str = str(self.fullfile)
            prefix,suffix = fullfile_str.split('.')
            df_filename = prefix + '.feather'
        
        # compression level 1 (fastest) - 22 (highest compression)
        self.dataf.to_feather(df_filename, compression="zstd", compression_level=5)

###############################################################################
##                                                                           ##
##                        main                                               ##
##                                                                           ##  
###############################################################################
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Vehicle and pedestrian classes trajectory extractor")
    parser.add_argument('--trajectorydata_path', '-p', type=str, default='', help="path to binary or feather file containing trajectory data")
    parser.add_argument('--skip_speed', type=bool, default=False, help="skip processing speed and heading data if not needed")
    parser.add_argument('--trajectory_file', '-t', type=str, 
                        help="file name of the binary formatted data, or pandas .feather format file containing the trajectory data.")
    args = parser.parse_args()

    # input all the track data to convert to feather format, and optionally process speed and heading data if they are present in the track data file.  
    if args.trajectorydata_path == '' and args.trajectory_file == '':
        print("Error: no trajectory data file specified, exiting...")
        sys.exit()

    tracking_data = TM_trackDataRecords(args.trajectorydata_path, args.trajectory_file)
    datarecords = tracking_data.ReadData()
    tracking_data.SaveData()
    if 'speed' not in tracking_data.col_names and 'heading' not in tracking_data.col_names:
        print("speed and heading data not found in the track data file, skipping processing of speed and heading data...")
    print("Done.")

