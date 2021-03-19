#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
sys.path.append('/media/ext4_storage2/NLP/Modules')
import os
program_folder = '/media/ext4_storage2/NLP'
year='2020'
google_drive_folder_ID = ''


os.system(f'gnome-terminal -- python {program_folder}/Modules/DOWNLOAD.py --year={year} --diretorio={program_folder} --folder_ID={google_drive_folder_ID}')
os.system(f'gnome-terminal -- python {program_folder}/Modules/FIREFOX_W.py')
