#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Add custom configs and default values"""


def read_secret_txt(file_path):
    """Read secret variables from file and return as dict"""
    variables = {}
    try:
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and '=' in line:
                    key, value = line.split('=', 1)
                    variables[key.strip()] = value.strip()
    except FileNotFoundError:
        print(f"Warning: {file_path} not found")
    return variables


def add_custom_config(_C):
    # Add your own customized configs.
    _C.CUSTOM = "Add your own customized configs here"
    _C.secret_txt = "/scratch/Video_Understanding/GraSP/TAPIS/.secret/.export_vars.txt"
    
    secrets = read_secret_txt(_C.secret_txt)
        # Add each secret variable to config
    for key, value in secrets.items():
        if value.lower() in ['true', 'false']:
            value = value.lower() == 'true'  # Convert to boolean
        setattr(_C, key, value)
    
    return _C
